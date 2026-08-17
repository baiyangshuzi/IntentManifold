# -*- coding: utf-8 -*-
"""v0.93 G3 主判——held-out 评估（真管线——配对检验——三态判定）

【预注册（先写后跑）】
- 模型：基线（未微调 Qwen2.5-0.5B）/ 主臂（adapter_final_main）/ 对照臂（adapter_final_control）
- 生成：held-out 4 中文主题（prompts.json lang=zh id 6-9）× 3 段 = 12 段/模型——
  temp 0.8 / no top-k/p / max 300 / min 100（与训练同参数——种子 1000+100t+k 复现）
- 指标：真管线（CPU BGE+ParaDiscNN）per-段 α mean_abs / ratio / density + 语言
  （PPL 同 3 篇 held-out 人类文档 + 可读性代理：重复 4-gram 比/句元长均值）
- 配对检验：per-段配对（12 对）——主臂 vs 基线（单侧 Wilcoxon：α↑、ratio↓、density↑）——
  对照臂 vs 基线（归因——评审 5）——主臂 vs 对照臂（增量）
- 三态判定：
  成立 = α↑ p<0.05 ∧ (ratio↓ p<0.10 ∨ density↑ p<0.10) ∧ 语言未退化 ∧ 终代理-真实缺口 ≤0.5 ∧ G1 严格
  部分 = ≥1 指标显著向目标 ∧ 无反向显著 ∧ 语言未退化（G1 降级时封顶"部分"）
  否定 = 全零/反向显著/语言退化
- 语言退化 = PPL 升 >25% ∨ (重复 4-gram↑>50% ∧ PPL>15%)——同训练语言门

产出：data/dim_analysis/eval_g3.json
"""
import sys, json, time, os
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path
import numpy as np
from scipy import stats

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / 'data' / 'dim_analysis'
RUN = OUT / 'lora_run'
QWEN = 'Qwen/Qwen2.5-0.5B'
BGE = 'BAAI/bge-small-zh-v1.5'
N_PER_TOPIC = 3
MAX_NEW, MIN_NEW = 300, 100

from train_reg_common import get_D_shared, load_disc, fp_from_bge
from train_reg_common import alpha_metrics, ratio_of, density_hard
from lora_train import split_clauses_offsets


def main():
    t0 = time.time()
    import torch
    import torch.nn.functional as F
    import transformers
    import sentence_transformers
    from peft import PeftModel

    dev = 'cuda' if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else 'cpu'
    tok = transformers.AutoTokenizer.from_pretrained(QWEN)
    dtype = torch.float16 if dev == 'cuda' else torch.float32
    disc = load_disc('cpu')
    enc = sentence_transformers.SentenceTransformer(BGE, device='cpu')
    D = get_D_shared()

    prompts = json.loads((BASE / 'data' / 'bilingual_test' / 'prompts.json').read_text(encoding='utf-8'))
    zh = [t['topic'] for t in prompts['topics'] if t.get('lang') == 'zh']
    eval_topics = zh[6:10]
    bil = BASE / 'data' / 'bilingual_test' / 'human_zh'
    docs = sorted(bil.glob('*.txt'))
    ppl_docs = [d.read_text(encoding='utf-8', errors='replace') for d in docs[3:6]]

    def make_model(adapter=None):
        from train_reg_common import load_causal_lm
        m = load_causal_lm(QWEN, dtype, dev).eval()
        if adapter:
            m = PeftModel.from_pretrained(m, RUN / adapter)
        return m

    def generate(model, prompt, seed):
        torch.manual_seed(seed)
        ids = tok(prompt, return_tensors='pt').to(dev)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=MAX_NEW, min_new_tokens=MIN_NEW,
                                 do_sample=True, temperature=0.8, top_k=None, top_p=None,
                                 repetition_penalty=1.0, pad_token_id=tok.eos_token_id)
        return tok.decode(out[0][ids['input_ids'].shape[1]:], skip_special_tokens=True)

    def seg_metrics(text):
        units = [u for u, _, _ in split_clauses_offsets(text)]
        if len(units) < 3:
            return None
        embs = enc.encode(units, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
        fp = fp_from_bge(embs, disc)
        a_abs, _, _ = alpha_metrics(fp, D)
        Df = fp[1:] - fp[:-1]
        r = ratio_of(Df, D)[0]
        return {'alpha_abs': a_abs, 'ratio': r, 'density': density_hard(fp), 'n': len(units)}

    def ppl_of(model):
        nll = n = 0
        for doc in ppl_docs:
            for i in range(0, len(doc), 1500):
                ids = tok(doc[i:i + 1500], return_tensors='pt').to(dev)
                if ids['input_ids'].shape[1] < 5:
                    continue
                with torch.no_grad():
                    out = model(ids['input_ids'], attention_mask=ids['attention_mask'])
                lg = out.logits[..., :-1, :].reshape(-1, out.logits.shape[-1])
                lab = ids['input_ids'][..., 1:].reshape(-1)
                m = ids['attention_mask'][..., 1:].reshape(-1).bool()
                nll += float(F.cross_entropy(lg[m], lab[m], reduction='sum'))
                n += int(m.sum())
        return float(np.exp(nll / max(n, 1)))

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

    arms = {}
    for name, adapter in [('baseline', None), ('main', 'adapter_final_main'),
                          ('control', 'adapter_final_control')]:
        m = make_model(adapter)
        segs, texts = [], []
        for ti, topic in enumerate(eval_topics):
            for k in range(N_PER_TOPIC):
                t = generate(m, topic, seed=1000 + 100 * ti + k)
                texts.append(t)
                segs.append(seg_metrics(t))
        valid = [x for x in segs if x]
        arms[name] = {'_seg': segs, 'texts': texts, 'n_valid': len(valid),
                      'metrics': {k: float(np.mean([v[k] for v in valid]))
                                  for k in ['alpha_abs', 'ratio', 'density']},
                      'ppl': ppl_of(m),
                      'readability': readability(texts)}
        print(f'{name}: {arms[name]["metrics"]} ppl {arms[name]["ppl"]:.3f} '
              f'rd {arms[name]["readability"]}', flush=True)
        del m
        torch.cuda.empty_cache()

    def paired(arm_a, arm_b, key, direction):
        seg_a, seg_b = arms[arm_a]['_seg'], arms[arm_b]['_seg']
        va = [x[key] for x in seg_a if x]
        vb = [x[key] for x in seg_b if x]
        if len(va) < 6 or len(vb) < 6:
            return None
        alt = 'greater' if direction == 'up' else 'less'
        w = stats.wilcoxon(va, vb, alternative=alt)
        return {'n': len(va), 'mean_a': float(np.mean(va)), 'mean_b': float(np.mean(vb)),
                'p': float(w.pvalue)}

    pairs = {}
    for (a, b) in [('main', 'baseline'), ('control', 'baseline'), ('main', 'control')]:
        pairs[f'{a}_vs_{b}'] = {k: paired(a, b, k, d) for k, d in
                                [('alpha_abs', 'up'), ('ratio', 'down'), ('density', 'up')]}
        print(f'{a} vs {b}:', {k: (v['p'] if v else None) for k, v in pairs[f"{a}_vs_{b}"].items()})

    g1 = json.loads((OUT / 'gate_g1.json').read_text(encoding='utf-8'))['g1']
    strict = g1 == '严格通过'
    mb = pairs['main_vs_baseline']
    alpha_ok = bool(mb['alpha_abs'] and mb['alpha_abs']['p'] < 0.05 and
                    mb['alpha_abs']['mean_a'] > mb['alpha_abs']['mean_b'])
    sec_ok = any(v and v['p'] < 0.10 and
                 ((k == 'ratio' and v['mean_a'] < v['mean_b']) or
                  (k == 'density' and v['mean_a'] > v['mean_b']))
                 for k, v in [('ratio', mb['ratio']), ('density', mb['density'])])
    counter = any(v and v['p'] < 0.10 and
                  ((k == 'alpha_abs' and v['mean_a'] < v['mean_b']) or
                   (k == 'ratio' and v['mean_a'] > v['mean_b']) or
                   (k == 'density' and v['mean_a'] < v['mean_b']))
                  for k, v in mb.items())
    ppl_up = arms['main']['ppl'] > 1.25 * arms['baseline']['ppl']
    rep_up = arms['main']['readability']['rep4_ratio'] > 1.5 * arms['baseline']['readability']['rep4_ratio']
    ppl_up15 = arms['main']['ppl'] > 1.15 * arms['baseline']['ppl']
    lang_bad = ppl_up or (rep_up and ppl_up15)
    final_gap = None
    cl = [json.loads(l) for l in (RUN / 'cycle_log_main.jsonl').read_text(encoding='utf-8').splitlines()
          if json.loads(l).get('gap')]
    if cl:
        gv = [v for v in cl[-1]['gap'].values() if v is not None]
        final_gap = max(gv) if gv else None
    gap_ok = final_gap is not None and final_gap <= 0.5
    if alpha_ok and sec_ok and not lang_bad and gap_ok and strict:
        verdict = '成立'
    elif (alpha_ok or sec_ok) and not counter and not lang_bad:
        verdict = '部分' + ('' if strict else '（G1 降级封顶）')
    else:
        verdict = '否定'
    print(f'判定: {verdict}（alpha_ok {alpha_ok} sec_ok {sec_ok} counter {counter} '
          f'lang_bad {lang_bad} gap_ok {gap_ok} final_gap {final_gap}）')

    out = {'verdict': verdict,
           'arms': {k: {kk: vv for kk, vv in v.items() if kk not in ('_seg', 'texts')}
                    for k, v in arms.items()},
           'pairs': pairs, 'final_gap': final_gap, 'g1': g1,
           'language': {'ppl_up_25': ppl_up, 'rep4_up_50_and_ppl15': rep_up and ppl_up15,
                        'main_ppl': arms['main']['ppl'], 'base_ppl': arms['baseline']['ppl']}}
    (OUT / 'eval_g3.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'落盘 eval_g3.json（{time.time()-t0:.0f}s）')


if __name__ == '__main__':
    main()
