# -*- coding: utf-8 -*-
"""v0.93 主臂/对照臂——自教学微调循环（LoRA Qwen2.5-0.5B——F1/F3/J1 意图损失）

【预注册（先写后跑——评审 1-7 吸收）】
- 采样：temp 0.8 / top_k=None / top_p=None / repetition_penalty=1.0 / max_new 300 / min 100——
  prompt = 训练 6 中文主题（prompts.json lang=zh id 0-5）
- 等价性（评审 3）：lora_dropout=0.0 + attention_dropout=0.0 ⇒ 采样与教师强制前向逐 token 一致
- 批调度 [LM, LM, Intent] × 16/循环 × 16 循环 = 768 步（512 LM + 256 Intent）
  ——LM 批 = 4×512 token 人类语料 NLL；Intent 批 = 2 段自生成（无 LM 损失）
- 损失：LM NLL + λ_eff·(L_α+L_ratio+L_jump)——平方 hinge/软计数 k=5（评审 4）——
  λ=0.1 固定；梯度守卫 λ_eff=0.1·min(1, 2·g_LM/g_Intent)（运行均值 β=0.9——每 12 步双反传刷新）；
  全局 clip 1.0
- 每循环：真管线指标（CPU——评审 6）10 段 + 代理-真实缺口 + held-out PPL +
  可读性代理（重复 4-gram 比/句元长均值——评审 7）+ 隐状态范数漂移
- G2 @ 循环 8：G1 严格——≥1 指标向目标移动 ≥20% 缺口 ∧ 代理-真实误差 ≤0.5；
  G1 降级——≥2 指标 ∧ 误差 ≤0.3（评审 1）
- 停止：G2 停 / 连续 4 循环无进展 / 语言门（PPL>25% 或 重复4gram↑>50%∧PPL>15%）/ 循环上限
- 刷新（评审 2）：偶数循环追加当前策略新鲜配对（上限 20K FIFO）+ 代理小 lr 重训 100 步（1e-5）
- mode='control'：批调度 [LM,LM,LM]（Intent 槽位放 LM 批）——同 768 步/批大小/lr/LoRA——
  唯一差异无意图损失（评审 5）
- 检查点每 192 步；NaN 守卫；种子 42

产出：data/dim_analysis/lora_run/（adapter_*、cycle_log.jsonl、heartbeat.json）
"""
import sys, json, time, argparse, os
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
if '--cpu' in sys.argv:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / 'data' / 'dim_analysis'
RUN = OUT / 'lora_run'
QWEN = 'Qwen/Qwen2.5-0.5B'
BGE = 'BAAI/bge-small-zh-v1.5'
SEED = 42
LR, WARMUP, TOTAL = 1e-4, 30, 768
LAMBDA = 0.1
MAX_TOK = 512
LM_BS = 2  # LM 批大小（batch 2×512×2 批/步 = 2048 token/步——预注册口径一致——
# 显存友好：双批图保留 ~4GB + 意图批 ~0.6GB < 12GB）
N_CYCLES, CYCLE_STEPS = 16, 48
N_SAMPLE = 32
SAMPLE_MAX, SAMPLE_MIN = 300, 100
REFRESH_EVERY, REFRESH_STEPS, REFRESH_LR = 1, 200, 3e-5  # 增强（v0.93 诊断：代理密度在模型分布失真——
# gap 1.0——每循环刷新 + 更高 lr + 更多步——文档化变更）
BUF_CAP = 20000
CKPT_EVERY = 192

from subclause_structure import QUOTE_RE, SUB_BOUNDARY, _SENT_END
from train_reg_common import get_D_shared, load_disc, fp_from_bge
from train_reg_common import alpha_metrics, ratio_of, density_hard
from intent_loss import intent_losses


def split_clauses_offsets(text):
    quote_ranges = [(m.start(), m.end()) for m in QUOTE_RE.finditer(text)]
    subs, cur, cur_start, i, n = [], "", None, 0, len(text)
    while i < n:
        in_q = any(s <= i < e for s, e in quote_ranges)
        if in_q:
            if cur.strip():
                subs.append((cur.strip(), cur_start, i))
                cur, cur_start = "", None
            for s, e in quote_ranges:
                if s <= i < e:
                    sub = text[i:e].strip()
                    if len(sub) >= 2:
                        st = i + text[i:e].find(sub)
                        subs.append((sub, st, st + len(sub)))
                    i = e
                    break
            continue
        ch = text[i]
        if ch in SUB_BOUNDARY or ch in _SENT_END:
            if cur.strip():
                subs.append((cur.strip(), cur_start, i))
            cur, cur_start = "", None
        else:
            if cur_start is None:
                cur_start = i
            cur += ch
        i += 1
    if cur.strip():
        subs.append((cur.strip(), cur_start, n))
    return subs


class Runner:
    def __init__(self, mode, cycles, cycle_steps, smoke=False):
        self.mode, self.cycles, self.cycle_steps, self.smoke = mode, cycles, cycle_steps, smoke
        # torch 2.13 语义：CUDA_VISIBLE_DEVICES='' 下 is_available() 仍 True——按 device_count 判
        self.device = 'cuda' if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else 'cpu'
        print(f'设备: {self.device}（cuda 可见数 {torch.cuda.device_count() if torch.cuda.is_available() else 0}）', flush=True)
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        self.step = 0
        self.g_lm, self.g_it = 1.0, 1.0

        import transformers
        import sentence_transformers
        from peft import LoraConfig, get_peft_model
        from surrogate_train import Surrogate
        self.tok = transformers.AutoTokenizer.from_pretrained(QWEN)
        self.tok.padding_side = 'left'  # decoder-only 批量生成必须 left-pad
        dtype = torch.float16 if self.device == 'cuda' else torch.float32
        from train_reg_common import load_causal_lm
        base = load_causal_lm(QWEN, dtype, self.device)
        lora = LoraConfig(r=8, lora_alpha=16, target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
                          lora_dropout=0.0, bias='none', task_type='CAUSAL_LM')
        self.model = get_peft_model(base, lora)
        # 不用 gradient checkpointing（torch 2.1 + tie-embedding 断链——梯度 None 崩溃——
        # 显存靠 batch 2×512×2 批（每步 2048 token——与原预注册一致）控制）
        self.model.print_trainable_parameters()
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=LR)
        self.lr_sched = torch.optim.lr_scheduler.LambdaLR(self.opt, self._lr_fn)
        self.D = torch.as_tensor(get_D_shared(), dtype=torch.float32).to(self.device)

        # 代理（冻结）
        self.sur = Surrogate().to(self.device)
        self.sur.load_state_dict(torch.load(OUT / 'surrogate.pt', map_location=self.device))
        sc = json.loads((OUT / 'surrogate_scaler.json').read_text(encoding='utf-8'))
        self.h_mean = torch.as_tensor(sc['h_mean'], dtype=torch.float32).to(self.device)
        self.h_std = torch.as_tensor(sc['h_std'], dtype=torch.float32).to(self.device)
        self.f_mean = torch.as_tensor(sc['f_mean'], dtype=torch.float32).to(self.device)
        self.f_std = torch.as_tensor(sc['f_std'], dtype=torch.float32).to(self.device)
        for p in self.sur.parameters():
            p.requires_grad_(False)
        self.sur.eval()

        # 真管线（CPU——评审 6）
        self.disc = load_disc('cpu')
        self.enc = sentence_transformers.SentenceTransformer(BGE, device='cpu')

        # 语料（预 tokenize——消除训练循环内 CPU tokenize 等待——GPU 吞吐优化）
        lm = json.loads((OUT / 'train_lm_corpus.json').read_text(encoding='utf-8'))
        self.lm_segs = [s['text'] for s in lm['segments']]
        self.lm_tok = [self.tok(t, return_tensors='pt', truncation=True, max_length=512)
                       for t in self.lm_segs]
        prompts = json.loads((BASE / 'data' / 'bilingual_test' / 'prompts.json').read_text(encoding='utf-8'))
        zh = [t['topic'] for t in prompts['topics'] if t.get('lang') == 'zh']
        self.train_topics, self.eval_topics = zh[:6], zh[6:10]
        bil = BASE / 'data' / 'bilingual_test' / 'human_zh'
        docs = sorted(bil.glob('*.txt'))
        self.ppl_docs = [d.read_text(encoding='utf-8', errors='replace') for d in docs[3:6]]

        # 刷新缓冲（初始 = pairs 训练子集）
        d = np.load(OUT / 'pairs.npz')
        tr = ~d['val']
        self.buf_h = [d['h'][tr]]
        self.buf_f = [d['fp'][tr]]

        RUN.mkdir(exist_ok=True)
        self.log = open(RUN / f'cycle_log_{mode}.jsonl', 'a', encoding='utf-8')
        self.ppl0 = self.ppl()
        # 可读性基准 = 未训练模型自身生成样本（修复：原合成重复文本基准使门失效）
        if self.mode == 'main':
            base_texts = [s[2] for s in self.sample_batch(self.train_topics[:4])]
            self.rd0 = self.readability(base_texts)
        else:
            self.rd0 = {'rep4_ratio': 0.05, 'unit_len_mean': 20.0}
        print(f'基线 PPL {self.ppl0:.3f} rd0 {self.rd0}', flush=True)

    def _lr_fn(self, step):
        if step < WARMUP:
            return step / WARMUP
        p = (step - WARMUP) / max(TOTAL - WARMUP, 1)
        return 0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * min(p, 1.0)))

    # ---------- 采样（批量——GPU 吞吐优化）----------
    @torch.no_grad()
    def sample_batch(self, topics):
        """一次 generate 批量采样（batch=len(topics)——GPU 满载——
        eval 包裹（dropout=0 下数学等价——等价性断言不受影响）"""
        self.model.eval()
        enc = self.tok(list(topics), return_tensors='pt', padding=True).to(self.device)
        out = self.model.generate(**enc, max_new_tokens=SAMPLE_MAX, min_new_tokens=SAMPLE_MIN,
                                  do_sample=True, temperature=0.8, top_k=None, top_p=None,
                                  repetition_penalty=1.0, pad_token_id=self.tok.eos_token_id)
        self.model.train()
        results = []
        for k, t in enumerate(topics):
            # prompt 长度用 attention_mask（pad_token_id==eos 时 pad 计数会把生成里的 EOS 误算）
            plen = int(enc['attention_mask'][k].sum())
            new = out[k][plen:].tolist()
            text = self.tok.decode(new, skip_special_tokens=True)
            tok_lens = [len(self.tok.decode([x], skip_special_tokens=True)) for x in new]
            results.append((enc['input_ids'][k][:plen].tolist(), new, text, tok_lens))
        return results

    # ---------- 教师强制（字符级精确对齐）----------
    def teacher_forward(self, prompt_ids, gen_ids, gen_text, tok_lens, grad=False):
        """前向 prompt+gen 全序列——单元↔token 映射 = 字符级对齐：
        re-encode(gen_text) 的 offset_mapping 给出 (token→char)——每个 char 位置经
        tok_lens 累积映射回 gen_ids 索引 → h 位置 gstart+j。与 id 序列是否幂等解耦——
        等价性（评审 3）在字符空间精确成立。"""
        ids = torch.as_tensor([prompt_ids + gen_ids], device=self.device)
        assert ids.shape[1] <= MAX_TOK + 50, f'超长: {ids.shape[1]}'
        with torch.set_grad_enabled(grad):
            out = self.model(ids, output_hidden_states=True)
        enc = self.tok(gen_text, return_offsets_mapping=True, add_special_tokens=False)
        off = enc['offset_mapping']
        # char→gen_ids 索引映射（累积）
        cum = np.cumsum(tok_lens).tolist()
        n_gen = len(gen_ids)

        def char_to_idx(c):
            """字符位置 → gen_ids 索引（首个累积超 c 的）"""
            lo, hi = 0, n_gen
            while lo < hi:
                mid = (lo + hi) // 2
                if cum[mid] > c:
                    hi = mid
                else:
                    lo = mid + 1
            return min(lo, n_gen - 1)

        units = split_clauses_offsets(gen_text)
        texts = [u for u, _, _ in units]
        h = out.hidden_states[-1][0]
        gstart = len(prompt_ids)
        hs = []
        for u, s, e in units:
            toks = []
            for k, (ts, te) in enumerate(off):
                if ts == te:
                    continue
                if s <= ts < e:
                    j = char_to_idx(ts)
                    toks.append(gstart + j)
            hs.append(h[toks].mean(0) if toks else h[0] * 0)
        return ids, texts, (torch.stack(hs) if hs else None)

    # ---------- LM 批 ----------
    def lm_batch(self, bs=LM_BS):
        """预 tokenize 批（8×512——吞吐优化——预注册 2048→4096 tok/步偏差声明）"""
        rng = np.random.default_rng(int(time.time() * 1000) % 2 ** 31)
        idx = rng.choice(len(self.lm_tok), bs, replace=False)
        batch = [self.lm_tok[i] for i in idx]
        maxlen = max(b['input_ids'].shape[1] for b in batch)
        ids = torch.full((bs, maxlen), self.tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((bs, maxlen), dtype=torch.long)
        for k, b in enumerate(batch):
            n = b['input_ids'].shape[1]
            ids[k, :n] = b['input_ids']
            att[k, :n] = b['attention_mask']
        return ids.to(self.device), att.to(self.device)

    def lm_loss(self, ids, att):
        out = self.model(ids, attention_mask=att)
        lg = out.logits[..., :-1, :].reshape(-1, out.logits.shape[-1])
        lab = ids[..., 1:].reshape(-1)
        m = att[..., 1:].reshape(-1).bool()
        return F.cross_entropy(lg[m], lab[m])

    # ---------- 意图批（可微——代理在图中）----------
    def intent_losses_batch(self, samples):
        tot = []
        for prompt_ids, gen_ids, text, tok_lens in samples:
            ids, texts, hs = self.teacher_forward(prompt_ids, gen_ids, text, tok_lens, grad=True)
            if hs is None or len(hs) < 3:
                continue
            z = (hs - self.h_mean) / self.h_std
            pf = self.sur(z) * self.f_std + self.f_mean
            tot.append(sum(intent_losses(pf, self.D)))
        if not tot:
            return None  # 无效批（调用处兜底——避免无图 zeros 断链）
        return torch.stack(tot).mean()

    # ---------- 梯度范数测量（每 12 步——运行均值刷新）----------
    def measure_norms(self, l_lm, l_it):
        """双反传测各自梯度范数——两图都保留（第三次合成反传还要用）"""
        self.opt.zero_grad()
        l_lm.backward(retain_graph=True)
        g1 = sum(p.grad.norm().item() ** 2 for p in self.model.parameters()
                 if p.grad is not None) ** 0.5
        self.opt.zero_grad()
        l_it.backward(retain_graph=True)
        g2 = sum(p.grad.norm().item() ** 2 for p in self.model.parameters()
                 if p.grad is not None) ** 0.5
        self.opt.zero_grad()
        return g1, g2

    # ---------- 真管线指标（CPU）----------
    def real_metrics(self, texts):
        Dn = get_D_shared()
        out = []
        for t in texts:
            units = [u for u, _, _ in split_clauses_offsets(t)]
            if len(units) < 3:
                out.append(None)
                continue
            embs = self.enc.encode(units, normalize_embeddings=True, batch_size=64,
                                   show_progress_bar=False)
            fp = fp_from_bge(embs, self.disc)
            a_abs, _, _ = alpha_metrics(fp, Dn)
            Df = fp[1:] - fp[:-1]
            r = ratio_of(Df, Dn)[0]
            out.append({'alpha_abs': a_abs, 'ratio': r, 'density': density_hard(fp), 'n': len(units)})
        return out

    def surrogate_pred_metrics(self, samples):
        Dn = get_D_shared()
        out = []
        for prompt_ids, gen_ids, text, tok_lens in samples:
            _, _, hs = self.teacher_forward(prompt_ids, gen_ids, text, tok_lens, grad=False)
            if hs is None or len(hs) < 3:
                out.append(None)
                continue
            with torch.no_grad():
                z = (hs - self.h_mean) / self.h_std
                pf = (self.sur(z) * self.f_std + self.f_mean).cpu().numpy()
            a_abs, _, _ = alpha_metrics(pf, Dn)
            Df = pf[1:] - pf[:-1]
            r = ratio_of(Df, Dn)[0]
            out.append({'alpha_abs': a_abs, 'ratio': r, 'density': density_hard(pf),
                        'hid_norm': float(np.linalg.norm(hs.cpu().numpy(), axis=1).mean())})
        return out

    # ---------- 可读性代理（评审 7）----------
    @staticmethod
    def readability(texts):
        n4 = rep4 = 0
        lens = []
        for t in texts:
            units = [u for u, _, _ in split_clauses_offsets(t)]
            lens += [len(u) for u in units]
            grams = set()
            for i in range(len(t) - 3):
                g = t[i:i + 4]
                if g in grams:
                    rep4 += 1
                else:
                    grams.add(g)
            n4 += max(len(t) - 3, 0)
        return {'rep4_ratio': rep4 / max(n4, 1), 'unit_len_mean': float(np.mean(lens)) if lens else 0.0}

    # ---------- PPL ----------
    @torch.no_grad()
    def ppl(self):
        nll = n = 0
        for doc in self.ppl_docs:
            for i in range(0, len(doc), 1500):
                p = doc[i:i + 1500]
                ids = self.tok(p, return_tensors='pt').to(self.device)
                if ids['input_ids'].shape[1] < 5:
                    continue
                out = self.model(ids['input_ids'], attention_mask=ids['attention_mask'])
                lg = out.logits[..., :-1, :].reshape(-1, out.logits.shape[-1])
                lab = ids['input_ids'][..., 1:].reshape(-1)
                m = ids['attention_mask'][..., 1:].reshape(-1).bool()
                nll += float(F.cross_entropy(lg[m], lab[m], reduction='sum'))
                n += int(m.sum())
        return float(np.exp(nll / max(n, 1)))

    # ---------- 刷新（评审 2：追加 + 小 lr 重训）----------
    def refresh_surrogate(self, fresh):
        from surrogate_train import Surrogate
        for h, fp in fresh:
            self.buf_h.append(h)
            self.buf_f.append(fp)
        H = np.concatenate(self.buf_h)
        F = np.concatenate(self.buf_f)
        if len(H) > BUF_CAP:
            H, F = H[-BUF_CAP:], F[-BUF_CAP:]
            self.buf_h, self.buf_f = [H], [F]
        X = (H - self.h_mean.cpu().numpy()) / self.h_std.cpu().numpy()
        Z = (F - self.f_mean.cpu().numpy()) / self.f_std.cpu().numpy()
        net = Surrogate().to(self.device)
        net.load_state_dict(torch.load(OUT / 'surrogate.pt', map_location=self.device))
        opt = torch.optim.AdamW(net.parameters(), lr=REFRESH_LR)
        net.train()
        fm, fs = self.f_mean.to(self.device), self.f_std.to(self.device)
        for _ in range(REFRESH_STEPS):
            i = np.random.default_rng().choice(len(X), 256, replace=False)
            x = torch.as_tensor(X[i], dtype=torch.float32).to(self.device)
            zt = torch.as_tensor(Z[i], dtype=torch.float32).to(self.device)
            ft = torch.as_tensor(F[i], dtype=torch.float32).to(self.device)
            pz = net(x)
            pr = pz * fs + fm
            loss = ((pz - zt) ** 2).mean() + 0.3 * (1 - (pr * ft).sum(1) /
                    (pr.norm(2, 1) * ft.norm(2, 1) + 1e-9)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        self.sur.load_state_dict(net.state_dict())
        for p in self.sur.parameters():
            p.requires_grad_(False)
        self.sur.eval()
        torch.save(net.state_dict(), OUT / 'surrogate.pt')

    # ---------- 主循环 ----------
    def run(self):
        g1 = json.loads((OUT / 'gate_g1.json').read_text(encoding='utf-8'))['g1']
        strict = g1 == '严格通过'
        g2_met, g2_err = (1, 0.5) if strict else (2, 0.3)
        baseline, best, no_gain = None, None, 0
        lam_eff = LAMBDA
        hid0 = None
        for cyc in range(1, self.cycles + 1):
            cyc_t0 = time.time()
            if self.mode == 'main':
                n_s = 8 if self.smoke else N_SAMPLE
                topics = [self.train_topics[k % 6] for k in range(n_s)]
                samples = self.sample_batch(topics)
                if self.smoke:
                    samples = (samples * 3)[:self.cycle_steps // 3 * 2]
            else:
                samples = []
            for g in range(self.cycle_steps // 3):
                if self.mode == 'control':
                    ids1, att1 = self.lm_batch()
                    ids2, att2 = self.lm_batch()
                    ids3, att3 = self.lm_batch()
                    l_tot = (self.lm_loss(ids1, att1) + self.lm_loss(ids2, att2) +
                             self.lm_loss(ids3, att3)) / 3
                    self.opt.zero_grad()
                    l_tot.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.opt.step()
                    self.lr_sched.step()
                    self.step += 1
                else:
                    ids1, att1 = self.lm_batch()
                    ids2, att2 = self.lm_batch()
                    l_lm = (self.lm_loss(ids1, att1) + self.lm_loss(ids2, att2)) / 2
                    l_it = self.intent_losses_batch(samples[2 * g:2 * g + 2])
                    if l_it is None:
                        # 意图批无效（生成段过短）——只做 LM 步
                        self.opt.zero_grad()
                        l_lm.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.opt.step()
                        self.lr_sched.step()
                        self.step += 1
                        continue
                    if not (torch.isfinite(l_lm) and torch.isfinite(l_it)):
                        print('NaN 守卫：损失非有限——停止', flush=True)
                        sys.exit(9)
                    # 双反传梯度合成（各图反传一次即释放——无 retain_graph——显存友好——
                    # 每步精确测 g_lm/g_it——梯度守卫运行均值更新）
                    self.opt.zero_grad()
                    l_lm.backward()
                    ng1 = sum(p.grad.norm().item() ** 2 for p in self.model.parameters()
                              if p.grad is not None) ** 0.5
                    g1 = {id(p): p.grad.clone() for p in self.model.parameters()
                          if p.grad is not None}
                    self.opt.zero_grad()
                    l_it.backward()
                    ng2 = sum(p.grad.norm().item() ** 2 for p in self.model.parameters()
                              if p.grad is not None) ** 0.5
                    self.g_lm = 0.9 * self.g_lm + 0.1 * ng1
                    self.g_it = 0.9 * self.g_it + 0.1 * max(ng2, 1e-9)
                    lam_eff = LAMBDA * min(1.0, 2.0 * self.g_lm / max(self.g_it, 1e-9))
                    for p in self.model.parameters():
                        if p.grad is not None and id(p) in g1:
                            p.grad = g1[id(p)] + lam_eff * p.grad
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.opt.step()
                    self.lr_sched.step()
                    self.step += 1
                if self.step % CKPT_EVERY == 0 and not self.smoke:
                    self.model.save_pretrained(RUN / f'adapter_{self.mode}_step{self.step}')
            # 循环结束指标
            entry = {'cycle': cyc, 'step': self.step, 'lam_eff': lam_eff,
                     'g_lm': self.g_lm, 'g_it': self.g_it,
                     'secs': round(time.time() - cyc_t0, 1)}
            if self.mode == 'main':
                rl = self.real_metrics([s[2] for s in samples[:10]])
                pr = self.surrogate_pred_metrics(samples[:10])
                rl_ok = [r for r in rl if r]
                pr_ok = [p for p in pr if p]
                real = {k: float(np.mean([r[k] for r in rl_ok])) for k in ['alpha_abs', 'ratio', 'density']}
                pred = {k: float(np.mean([p[k] for p in pr_ok])) for k in ['alpha_abs', 'ratio', 'density']}
                gap = {k: abs(pred[k] - real[k]) / abs(real[k]) if real.get(k) else None for k in real}
                hid_mean = float(np.mean([p['hid_norm'] for p in pr_ok]))
                if hid0 is None:
                    hid0 = hid_mean
                entry.update({'real': real, 'pred': pred, 'gap': gap, 'hid_drift': hid_mean / hid0})
                if baseline is None:
                    baseline, best = dict(real), dict(real)
                move = {}
                for k in real:
                    tgt = {'alpha_abs': 1.4746, 'ratio': 0.0863, 'density': 12.08}[k]
                    base = baseline.get(k, tgt)
                    if abs(tgt - base) > 1e-9:
                        move[k] = (real[k] - base) / (tgt - base)
                entry['move'] = move
                if any(move.get(k, 0) >= 0.05 for k in real):
                    no_gain = 0
                else:
                    no_gain += 1
                tgt_map = {'alpha_abs': 1.4746, 'ratio': 0.0863, 'density': 12.08}
                for k in real:
                    if abs(real[k] - tgt_map[k]) < abs(best[k] - tgt_map[k]):
                        best[k] = real[k]
            entry['ppl'] = self.ppl()
            entry['readability'] = self.readability([s[2] for s in samples[:10]]) if samples else self.rd0
            self.log.write(json.dumps(entry, ensure_ascii=False) + '\n')
            self.log.flush()
            print(json.dumps(entry, ensure_ascii=False), flush=True)
            (RUN / 'heartbeat.json').write_text(json.dumps({'cycle': cyc, 'step': self.step}))
            # G2 @ 循环 8
            if self.mode == 'main' and cyc == 8 and not self.smoke:
                met_ok = sum(1 for k in move if move[k] >= 0.2) >= g2_met
                err_ok = all(v is not None and v <= g2_err for v in gap.values())
                if not (met_ok and err_ok):
                    print(f'G2 未过（move {move} gap {gap}）——保存权重后停止', flush=True)
                    self.model.save_pretrained(RUN / f'adapter_final_{self.mode}_g2fail')
                    self.log.write(json.dumps({'g2': 'FAIL', 'move': move, 'gap': gap}) + '\n')
                    self.log.flush()
                    sys.exit(3)
                print(f'G2 通过（move {move} gap {gap}）', flush=True)
            # 停止规则
            if self.mode == 'main' and no_gain >= 4:
                print('早停：连续 4 循环无进展', flush=True)
                break
            ppl_r = entry['ppl']
            if ppl_r > 1.25 * self.ppl0:
                print('语言门：PPL>25%——停止', flush=True)
                break
            if entry['readability']['rep4_ratio'] > 1.5 * self.rd0['rep4_ratio'] and ppl_r > 1.15 * self.ppl0:
                print('语言门：重复 4-gram↑>50% ∧ PPL>15%——停止', flush=True)
                break
            # 刷新（偶数循环）
            if self.mode == 'main' and cyc % REFRESH_EVERY == 0 and not self.smoke:
                fresh = []
                for prompt_ids, gen_ids, text, tok_lens in samples:
                    _, _, hs = self.teacher_forward(prompt_ids, gen_ids, text, tok_lens, grad=False)
                    if hs is None:
                        continue
                    units = [u for u, _, _ in split_clauses_offsets(text)]
                    if len(units) < 3 or len(hs) != len(units):
                        continue
                    embs = self.enc.encode(units, normalize_embeddings=True, batch_size=64,
                                           show_progress_bar=False)
                    fresh.append((hs.cpu().numpy(), fp_from_bge(embs, self.disc)))
                if fresh:
                    self.refresh_surrogate(fresh)
                    print(f'刷新完成（缓冲 {sum(len(x) for x in self.buf_h)} 对）', flush=True)
        self.model.save_pretrained(RUN / f'adapter_final_{self.mode}')
        self.log.write(json.dumps({'done': True, 'mode': self.mode, 'step': self.step}) + '\n')
        self.log.flush()
        print(f'完成：{self.mode} {self.step} 步', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['main', 'control'], default='main')
    ap.add_argument('--cycles', type=int, default=N_CYCLES)
    ap.add_argument('--cycle-steps', type=int, default=CYCLE_STEPS)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--cpu', action='store_true')
    a = ap.parse_args()
    if a.cpu:
        import os as _os
        _os.environ['CUDA_VISIBLE_DEVICES'] = ''
    r = Runner(a.mode, a.cycles, a.cycle_steps, a.smoke)
    r.run()


if __name__ == '__main__':
    main()
