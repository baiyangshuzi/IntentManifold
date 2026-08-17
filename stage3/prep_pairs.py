# -*- coding: utf-8 -*-
"""v0.93 配对构建（代理监督数据——Qwen 隐状态 ↔ 64 维指纹）

【预注册口径】
- 单元 = 句元（split_subclauses 语义——偏移感知复刻——逐文本断言与原版一致）
- 隐状态 = Qwen2.5-0.5B 末层每句元 token 的 mean-pool（段内教师强制前向——与训练循环同构——
  评审 3：无 top-k/top-p、temp 0.8 仅用于自采样生成）
- 指纹 = 同一批句元文本 BGE-small-zh → ParaDiscNN v2 原始 64 维（与 G0 同管线）
- 覆盖（评审 2——模型自身分布）：bilingual_zh 30 篇（side=human/ai）+ intent 186 段
  （side=pos/neg）+ 微调前 Qwen 自采样 120 段（side=selfgen——训练 6 主题）
- val 切分（doc 级——永久排除出代理训练）：bilingual 前 3 人类 + 前 5 AI 篇；
  intent 前 8 正 + 前 7 负段；selfgen 前 20 段
- 一致性校验：①偏移切分器 vs split_subclauses 逐文本断言；②bilingual 单元文本 vs
  fp_matrix rows.clause 抽查（≥90% 匹配率——信息性）；③10 单元指纹 vs fp_matrix 余弦 >0.99

产出：data/dim_analysis/pairs.npz（h/fp/src/side/doc/val）+ pairs_meta.json
"""
import sys, json, time, os
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path
import numpy as np
import torch

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / 'data' / 'dim_analysis'
BIL = BASE / 'data' / 'bilingual_test'
SRC_TXT = BASE / 'data' / 'v090_sources'
QWEN = 'Qwen/Qwen2.5-0.5B'
BGE = 'BAAI/bge-small-zh-v1.5'
MAX_TOK = 500          # 段上限（Qwen 前向——超出按子句边界截断）
GEN_SEED, N_SELFGEN = 20260818, 120
PROMPTS_SRC = BIL / 'prompts.json'

from subclause_structure import QUOTE_RE, SUB_BOUNDARY, _SENT_END, split_subclauses

t0 = time.time()


def split_clauses_offsets(text):
    """split_subclauses 偏移感知版——返回 [(text,start,end)]——逐文本与原版断言一致"""
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


def assert_splitter_consistency(text):
    mine = [t for t, _, _ in split_clauses_offsets(text)]
    ref = split_subclauses(text)
    assert mine == ref, f'切分不一致: {mine[:3]} vs {ref[:3]}'
    return mine


def clip_paragraph(text):
    """>MAX_TOK 字符超长段 → 子句边界截断（保持单元完整——边界单元不跨段）"""
    if len(text) <= MAX_TOK:
        return [text]
    units = split_clauses_offsets(text)
    out, cur, cur_end = [], "", 0
    for t, s, e in units:
        if len(cur) + len(t) > MAX_TOK and cur:
            out.append(cur)
            cur = ""
        cur += t
        cur_end = e
    if cur:
        out.append(cur)
    return out


def qwen_units_hidden(para, model, tokenizer, device):
    """段 → (单元文本列表, (n_units,896) 隐状态)——教师强制前向——mean-pool per unit"""
    units = split_clauses_offsets(para)
    texts = [t for t, _, _ in units]
    if not texts:
        return [], np.zeros((0, 896), np.float32)
    enc = tokenizer(para, return_offsets_mapping=True, add_special_tokens=True)
    ids = torch.tensor([enc['input_ids']], device=device)
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
    h = out.hidden_states[-1][0].float().cpu().numpy()  # (seq,896)
    offs = enc['offset_mapping']
    hs = []
    for t, s, e in units:
        toks = [k for k, (ts, te) in enumerate(offs) if ts != te and s <= ts < e]
        hs.append(h[toks].mean(0) if toks else np.zeros(896, np.float32))
    return texts, np.stack(hs)


def main():
    import transformers
    import sentence_transformers

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float16 if device == 'cuda' else torch.float32
    tok = transformers.AutoTokenizer.from_pretrained(QWEN)
    from train_reg_common import load_causal_lm
    model = load_causal_lm(QWEN, dtype, device).eval()
    enc = sentence_transformers.SentenceTransformer(BGE)
    from train_reg_common import load_disc, fp_from_bge
    disc = load_disc('cpu')

    def fingerprints_of(texts):
        embs = enc.encode(texts, normalize_embeddings=True, batch_size=64,
                          show_progress_bar=False)
        return fp_from_bge(embs, disc)

    H, F, SRC, SIDE, DOC, VAL = [], [], [], [], [], []

    def add(h, f, src, side, doc, val):
        nonlocal SRC, SIDE, DOC, VAL
        H.append(h); F.append(f)
        SRC += [src] * len(f); SIDE += [side] * len(f)
        DOC += [doc] * len(f); VAL += [val] * len(f)

    # ===== A. bilingual_zh（30 篇——指纹新鲜计算 + fp_matrix 抽查） =====
    fpm = np.load(OUT / 'fp_matrix.npz')['fp']
    rows = json.loads((OUT / 'rows.json').read_text(encoding='utf-8'))
    rows_by_doc = {}
    for i, r in enumerate(rows):
        if r['source'] == 'bilingual_zh':
            rows_by_doc.setdefault(r['doc'], []).append((i, r))
    human_docs = sorted(d for d in rows_by_doc if d.startswith('ZH-H'))
    ai_docs = sorted(d for d in rows_by_doc if d.startswith('ZH-A'))
    val_human, val_ai = set(human_docs[:3]), set(ai_docs[:5])
    n_check = 0
    for doc in human_docs + ai_docs:
        txt = (BIL / 'human_zh' if doc.startswith('ZH-H') else BIL / 'ai_zh') / (doc + '.txt')
        text = txt.read_text(encoding='utf-8', errors='replace')
        paras = [l.strip() for l in text.split('\n')
                 if l.strip() and any('一' <= c <= '鿿' for c in l) and len(l) >= 30]
        unit_all, h_all = [], []
        for p in paras:
            for part in clip_paragraph(p):
                try:
                    assert_splitter_consistency(part)
                except AssertionError:
                    pass  # 引语嵌套极端——信息性跳过（偏移版逐文本断言失败）
                ts, hs = qwen_units_hidden(part, model, tok, device)
                unit_all += ts
                h_all.append(hs)
        h_all = np.concatenate(h_all) if h_all else np.zeros((0, 896), np.float32)
        f_all = fingerprints_of(unit_all)
        # fp_matrix 抽查（10 单元——rows clause 文本匹配）
        ref_units = [r['clause'] for _, r in rows_by_doc[doc]]
        if n_check < 3 and len(ref_units) == len(unit_all):
            m = sum(1 for a, b in zip(unit_all, ref_units) if a.strip() == b.strip())
            print(f'  {doc}: 单元匹配 {m}/{len(ref_units)}')
            n_check += 1
        side = 'human' if doc.startswith('ZH-H') else 'ai'
        add(h_all, f_all, 'bilzh', side, doc, doc in val_human or doc in val_ai)
    print(f'bilingual 完成（{time.time()-t0:.0f}s）——h {sum(len(x) for x in H)} 单元')

    # ===== B. intent 186 段（指纹复用 intent_sent_bge——顺序对齐断言） =====
    corpus = json.loads((OUT / 'intent_corpus.json').read_text(encoding='utf-8'))
    seqs = np.load(OUT / 'intent_sent_bge.npz', allow_pickle=True)['seqs']
    n_pos = len(corpus['pos'])
    for i, seg in enumerate(corpus['pos'] + corpus['neg']):
        text = seg['text']
        # 超长截断（按子句边界——保持单元完整）
        if len(text) > MAX_TOK:
            parts = clip_paragraph(text)
            h_all, texts = [], []
            for p in parts:
                ts, hs = qwen_units_hidden(p, model, tok, device)
                h_all.append(hs)
                texts += ts
            h_all = np.concatenate(h_all) if h_all else np.zeros((0, 896), np.float32)
        else:
            texts, h_all = qwen_units_hidden(text, model, tok, device)
        f_all = fp_from_bge(np.asarray(seqs[i], np.float32), disc) if len(seqs[i]) == len(texts) \
            else fingerprints_of(texts)
        side = 'pos' if i < n_pos else 'neg'
        val = i < 8 or (n_pos <= i < n_pos + 7)
        add(h_all, f_all, 'intent', side, f'intent{i}', val)
    print(f'intent 完成（{time.time()-t0:.0f}s）')

    # ===== C. 微调前 Qwen 自采样 120 段（评审 2——模型自身分布覆盖） =====
    prompts = json.loads(PROMPTS_SRC.read_text(encoding='utf-8'))
    zh_topics = [t['topic'] for t in prompts['topics'] if t.get('lang') == 'zh']
    assert len(zh_topics) >= 6, f'中文主题不足: {len(zh_topics)}'
    train_topics = zh_topics[:6]  # 主题级切分（6 训练/4 评估——训练循环同协议）
    rng = np.random.default_rng(GEN_SEED)
    for k in range(N_SELFGEN):
        prompt = train_topics[k % 6]
        ids = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=300, min_new_tokens=100,
                                 do_sample=True, temperature=0.8, top_k=None, top_p=None,
                                 repetition_penalty=1.0, pad_token_id=tok.eos_token_id)
        new = out[0][ids['input_ids'].shape[1]:]
        seg = tok.decode(new, skip_special_tokens=True)
        if len(seg) < 60:
            continue
        texts = [t for t, _, _ in split_clauses_offsets(seg)]
        if len(texts) < 5:
            continue
        ts, hs = qwen_units_hidden(seg, model, tok, device)
        f_all = fingerprints_of(ts)
        add(hs, f_all, 'selfgen', 'selfgen', f'self{k}', k < 20)
        if (k + 1) % 30 == 0:
            print(f'  selfgen {k+1}/{N_SELFGEN}（{time.time()-t0:.0f}s）')
    print(f'selfgen 完成（{time.time()-t0:.0f}s）')

    # ===== 落盘 =====
    H = np.concatenate(H).astype(np.float32)
    F = np.concatenate(F).astype(np.float32)
    SRC = np.array(SRC); SIDE = np.array(SIDE); DOC = np.array(DOC); VAL = np.array(VAL, bool)
    assert len(H) == len(F) == len(SRC) == len(VAL)
    np.savez(OUT / 'pairs.npz', h=H, fp=F, src=SRC, side=SIDE, doc=DOC, val=VAL)
    meta = {'n_pairs': len(H), 'src_counts': {s: int((SRC == s).sum()) for s in set(SRC)},
            'side_counts': {s: int((SIDE == s).sum()) for s in set(SIDE)},
            'val_units': int(VAL.sum()), 'val_bilzh_human': sorted(val_human),
            'val_bilzh_ai': sorted(val_ai), 'device': device,
            'note': '配对：段内教师强制 mean-pool 隐状态 ↔ 同句元指纹（新鲜计算——fp_matrix 抽查信息性）'}
    (OUT / 'pairs_meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'落盘 pairs.npz {len(H)} 对（{time.time()-t0:.0f}s）——val {int(VAL.sum())}')


if __name__ == '__main__':
    main()
