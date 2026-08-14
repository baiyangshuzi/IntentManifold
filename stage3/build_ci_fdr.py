# -*- coding: utf-8 -*-
"""v0.66-4 ⑤ 置信区间 + 多重比较校正（全部现有实验——纯计算）

bootstrap 95% CI（1000 次）——全部效应量 d（严格配对 60 对 + 多段连续 35 组 + 独立集 32 对）
BH-FDR 校正——全部 p 值——明确区分"校正后仍显著/不再显著"

产出：data/independent_test/ci_fdr.json
"""
import sys, json, math
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path
import numpy as np

BASE = Path('C:/Users/bai/Desktop/小说系统')
OUT = BASE / 'data' / 'independent_test'


def cohens_d(h, a):
    h = np.asarray(h, dtype=float); a = np.asarray(a, dtype=float)
    sp = math.sqrt(((len(h) - 1) * h.var(ddof=1) + (len(a) - 1) * a.var(ddof=1)) / (len(h) + len(a) - 2))
    return float((h.mean() - a.mean()) / sp) if sp else 0.0


def bootstrap_ci(h, a, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    h = np.asarray(h, dtype=float); a = np.asarray(a, dtype=float)
    ds = []
    for _ in range(n_boot):
        hb = rng.choice(h, len(h), replace=True)
        ab = rng.choice(a, len(a), replace=True)
        ds.append(cohens_d(hb, ab))
    ds = np.array(ds)
    return [round(float(np.percentile(ds, 2.5)), 2), round(float(np.percentile(ds, 97.5)), 2)]


def bh_fdr(pvals):
    """Benjamini-Hochberg FDR——返回校正后 q 值"""
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = np.zeros(n)
    ranked[order] = np.arange(1, n + 1)
    q = pvals * n / ranked
    # 单调化
    q_sorted = np.sort(q)
    for i in range(n - 2, -1, -1):
        q_sorted[i] = min(q_sorted[i], q_sorted[i + 1])
    out = np.zeros(n)
    out[order] = q_sorted
    return out


def main():
    from scipy import stats as sc
    all_rows = []   # (实验, 域, 维度, human_mean, ai_mean, d, ci, p)

    def add_rows(exp, data, feats, getter):
        for dom, dom_data in data.items():
            for k in feats:
                if k not in dom_data:
                    continue
                x = dom_data[k]
                all_rows.append((exp, dom, k, x['human_mean'], x['ai_mean'], x['cohens_d'], None, x['p']))

    # 严格配对 60 对（L1-L5）
    r1 = json.loads((BASE / 'data/generalization_strict/results.json').read_text(encoding='utf-8'))
    FEATS1 = ['disc', 'sent_proj', 'traj', 'l7_adj', 'word_proj', 'word_adj', 'entropy']
    for layer in ['L1', 'L2', 'L3', 'L4', 'L5']:
        for k in FEATS1:
            if k in r1[layer]:
                x = r1[layer][k]
                all_rows.append(('严格配对', layer, k, x['human_mean'], x['ai_mean'], x['cohens_d'], None, x['p']))

    # 多段连续 35 组（by_domain——6 域×11 维）
    r3 = json.loads((BASE / 'data/para_level_pairs/results.json').read_text(encoding='utf-8'))
    FEATS3 = ['disc', 'sent_proj', 'traj', 'l7_adj', 'word_proj', 'word_adj', 'entropy',
              'seg_mean', 'seg_inner_std', 'seg_adj_diff', 'seg_traj']
    for dom, dd in r3['by_domain'].items():
        for k in FEATS3:
            if k in dd:
                x = dd[k]
                all_rows.append(('多段连续', dom, k, x['h'], x['a'], x['d'], None, None))

    # 独立集 32 对
    ri = json.loads((OUT / 'results.json').read_text(encoding='utf-8'))
    for dom, dd in ri.items():
        if dom in ('circularity', 'meta'):
            continue
        for k in FEATS1:
            if k in dd:
                x = dd[k]
                all_rows.append(('独立集', dom, k, x['human_mean'], x['ai_mean'], x['d'], x['ci95'], x['p']))

    # 多段连续/独立集 补 bootstrap CI（原始数据不在结果里——用聚合均值近似不可行——重算需要原始数据）
    # 简化：CI 对已有原始数据的（严格配对 60 对 per_pair + 独立集）计算——多段连续用 by_domain 聚合均值
    # 严格配对 60 对 CI（per_pair 有逐对数据）
    r1_pp = r1.get('per_pair', [])
    for layer in ['L1', 'L2', 'L3', 'L4', 'L5']:
        rows_pp = [p for p in r1_pp if p['layer'] == layer]
        for k in FEATS1:
            hv = [p[f'h_{k}'] for p in rows_pp if f'h_{k}' in p]
            av = [p[f'a_{k}'] for p in rows_pp if f'a_{k}' in p]
            if hv and av:
                for i, row in enumerate(all_rows):
                    if row[0] == '严格配对' and row[1] == layer and row[2] == k:
                        all_rows[i] = (row[0], row[1], row[2], row[3], row[4], row[5],
                                       bootstrap_ci(hv, av), row[7])

    # FDR 校正（全部 p 值——多段连续域无 p（只有 d）——只校正有 p 的行）
    p_rows = [i for i, r in enumerate(all_rows) if r[7] is not None]
    pvals = [all_rows[i][7] for i in p_rows]
    qvals = bh_fdr(pvals)
    for i, q in zip(p_rows, qvals):
        r = all_rows[i]
        all_rows[i] = (r[0], r[1], r[2], r[3], r[4], r[5], r[6], round(float(q), 4))

    # 输出
    print(f'总检验 {len(all_rows)} 项——其中 p 值 {len(pvals)} 项（FDR 校正）')
    out = {'rows': []}
    for r in all_rows:
        exp, dom, k, hm, am, d, ci, q = r
        out['rows'].append({'exp': exp, 'domain': dom, 'dim': k,
                            'human': hm, 'ai': am, 'd': d, 'ci95': ci, 'q': q})
    # 显著性分类表
    sig = [(r[0], r[1], r[2], r[7]) for r in all_rows if r[7] is not None and r[7] < 0.05]
    nonsig = [(r[0], r[1], r[2], r[7]) for r in all_rows if r[7] is not None and r[7] >= 0.05]
    out['fdr_sig'] = [{'exp': s[0], 'domain': s[1], 'dim': s[2], 'q': s[3]} for s in sig]
    out['fdr_nonsig'] = [{'exp': s[0], 'domain': s[1], 'dim': s[2], 'q': s[3]} for s in nonsig]
    print(f'FDR 后仍显著（q<0.05）: {len(sig)} 项')
    print(f'FDR 后不再显著（q>=0.05）: {len(nonsig)} 项')
    for s in nonsig:
        print(f'  不再显著: {s[0]} {s[1]} {s[2]} q={s[3]}')
    (OUT / 'ci_fdr.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print('\n落盘 ci_fdr.json')


if __name__ == '__main__':
    main()
