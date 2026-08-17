# -*- coding: utf-8 -*-
"""v0.93 G0 门——intent 语料句元指纹 pos(73) vs neg(113) 三指标方向与 bilingual 锚一致性

【预注册判据】≥2/3 方向与 bilingual 锚一致（pos ᾱ > neg ᾱ、pos ratio < neg ratio、
pos density > neg density——人类侧方向）→ 通过；否则 LM 流只留 bilingual 人类篇+毛选，
intent 段仅用于代理训练（声明限制仍继续）。

数据：intent_sent_bge.npz（186 句元级 BGE 序列——63 正序）→ ParaDiscNN v2 → 64 维原始指纹
→ per-doc 三指标（α mean_abs / ratio_of / density——p90=3.2286）。
"""
import sys, json, time
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path
import numpy as np
from scipy import stats
from train_reg_common import (OUT, load_disc, get_D_shared, fp_from_bge,
                              alpha_metrics, ratio_of, density_hard)

t0 = time.time()
disc = load_disc('cpu')
D = get_D_shared()
d = np.load(OUT / 'intent_sent_bge.npz', allow_pickle=True)
seqs = d['seqs']
corpus = json.loads((OUT / 'intent_corpus.json').read_text(encoding='utf-8'))
n_pos = len(corpus['pos'])
y = np.array([1] * n_pos + [0] * (len(seqs) - n_pos))

met = []
for i, seq in enumerate(seqs):
    fp = fp_from_bge(seq, disc)
    a_abs, a_sgn, na = alpha_metrics(fp, D)
    r, jpu, jppu = ratio_of(fp[1:] - fp[:-1], D) if len(fp) >= 2 else (0.0, 0.0, 0.0)
    den = density_hard(fp)
    met.append({'i': i, 'y': int(y[i]), 'n': len(seq), 'alpha_abs': a_abs,
                'alpha_sgn': a_sgn, 'ratio': r, 'density': den})

valid = [m for m in met if m['n'] >= 2]
pos = [m for m in valid if m['y'] == 1]
neg = [m for m in valid if m['y'] == 0]
print(f'G0: {len(pos)} 正 / {len(neg)} 负（n≥2——{time.time()-t0:.0f}s）')

res = {}
for key, human_dir in [('alpha_abs', 'up'), ('ratio', 'down'), ('density', 'up')]:
    vp = np.array([m[key] for m in pos])
    vn = np.array([m[key] for m in neg])
    u = stats.mannwhitneyu(vp, vn, alternative='two-sided')
    direction = 'pos>neg' if vp.mean() > vn.mean() else 'pos<neg'
    match = (direction == 'pos>neg') == (human_dir == 'up')
    res[key] = {'pos_mean': float(vp.mean()), 'neg_mean': float(vn.mean()),
                'direction': direction, 'human_dir': human_dir, 'match': match,
                'U_p': float(u.pvalue)}
    print(f'  {key}: pos {vp.mean():.4f} vs neg {vn.mean():.4f} — {direction} '
          f'(期望 {human_dir}) match={match} p={u.pvalue:.3e}')

matches = sum(1 for k in res if res[k]['match'])
gate = matches >= 2
verdict = f'G0 通过（{matches}/3 方向与 bilingual 锚一致）' if gate else \
    f'G0 降级（{matches}/3——LM 流只留 bilingual 人类篇+毛选——声明限制仍继续）'
print(verdict)

out = {'gate': bool(gate), 'matches': matches, 'metrics': res, 'verdict': verdict,
       'n_pos': len(pos), 'n_neg': len(neg),
       'anchors': {'alpha': '1.4746 vs 1.2989', 'ratio': '0.0863 vs 0.1048',
                   'density': '12.08 vs 7.74'}}
(OUT / 'gate_g0.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
print('落盘 gate_g0.json ✓')
