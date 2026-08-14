# -*- coding: utf-8 -*-
"""v0.74 阶段 A：维度活性与稳定性筛选 + 探针复现（验证④）

输入：data/dim_analysis/fp_matrix.npz + rows.json
输出：activity.json（64 维活性表 + 11 维判定）+ S_active 清单
①激活方差（跨样本）②跨域差异（白话可判域 vs 时评不可判域——Cohen's d）
③seed 稳定性（training_intervention none/b03/b05 同 prompt 3 seed——维度值跨 seed 方差比）
④探针复现（独立集白话人类段——18 特征 Spearman vs dim_probe.json——11 维并集口径确认）
"""
import sys, json, os, re
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path
import numpy as np
from scipy import stats as sc

BASE = Path('C:/Users/bai/Desktop/小说系统')
OUT = BASE / 'data' / 'dim_analysis'
sys.path.insert(0, str(BASE / 'stage3'))
TARGET = [5, 10, 11, 22, 26, 34, 43, 46, 48, 52, 59]


def load():
    d = np.load(OUT / 'fp_matrix.npz')
    fp = d['fp']
    rows = json.loads((OUT / 'rows.json').read_text(encoding='utf-8'))
    return fp, rows


def cohens_d(a, b):
    return (np.mean(a) - np.mean(b)) / np.sqrt((np.var(a) + np.var(b)) / 2 + 1e-9)


def main():
    fp, rows = load()
    print(f'矩阵: {fp.shape}')
    # ===== ① 激活方差（跨样本——全部语料）=====
    v_act = fp.var(0)  # (64,)

    # ===== ② 跨域差异（独立集白话 vs 时评——human 侧）=====
    bai_idx = [i for i, r in enumerate(rows) if r['source'] == 'independent_test'
               and r['side'] == 'human' and 'B-' in r['doc']]
    ship_idx = [i for i, r in enumerate(rows) if r['source'] == 'independent_test'
                and r['side'] == 'human' and 'S-' in r['doc']]
    d_domain = np.array([cohens_d(fp[bai_idx, j], fp[ship_idx, j]) if len(bai_idx) > 3 and len(ship_idx) > 3 else 0
                         for j in range(64)])

    # ===== ③ seed 稳定性（training_intervention none/b03/b05——同 prompt 3 seed）=====
    # 对每个 (condition, prompt)——3 个 seed 的 run——每维跨 seed 方差
    from collections import defaultdict
    seed_groups = defaultdict(list)  # (cond, prompt, seg) -> list of clause idx per seed
    for i, r in enumerate(rows):
        if r['source'] == 'training_intervention' and r['side'] in ('none', 'b03', 'b05'):
            seed_groups[(r['side'], r['prompt'], r['seg'])].append((r['seed'], i))
    # 每维：组内跨 seed 方差（同 prompt 同 seg 不同 seed 的句元指纹差异）vs 组间
    within_var = np.zeros(64)
    n_groups = 0
    for key, items in seed_groups.items():
        by_seed = defaultdict(list)
        for s, i in items:
            by_seed[s].append(i)
        if len(by_seed) < 2:
            continue
        n_groups += 1
        # 跨 seed 的方差（每 seed 取句元均值指纹——3 个 seed 均值向量——方差）
        means = [fp[idx].mean(0) for idx in by_seed.values()]
        if len(means) >= 2:
            within_var += np.array(means).var(0)
    within_var = within_var / max(n_groups, 1)
    # seed 稳定性指标：跨 seed 方差 / 激活方差（比值小=稳定）
    seed_ratio = within_var / (v_act + 1e-9)

    # ===== 判定 =====
    p5_var = np.percentile(v_act, 5)
    med_ratio = np.median(seed_ratio)
    activity = {}
    for j in range(64):
        dead = v_act[j] < p5_var
        noisy = seed_ratio[j] > med_ratio * 3  # 跨 seed 波动 3 倍于中位
        activity[j] = {'var': float(v_act[j]), 'domain_d': float(d_domain[j]),
                       'seed_ratio': float(seed_ratio[j]), 'dead': bool(dead), 'noisy': bool(noisy),
                       'active': not (dead or noisy)}
    s_active = [j for j in TARGET if activity[j]['active']]
    report = {'target': TARGET, 's_active': s_active,
              'target_activity': {j: activity[j] for j in TARGET},
              'n_dead_total': sum(1 for j in range(64) if activity[j]['dead']),
              'n_noisy_total': sum(1 for j in range(64) if activity[j]['noisy']),
              'criteria': 'dead=方差<p5；noisy=跨 seed 方差比>3×中位；active=非两者'}
    (OUT / 'activity.json').write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding='utf-8')
    print('=== 阶段 A：维度活性 ===')
    print(f'全 64 维: 死维度 {report["n_dead_total"]}——噪声维度 {report["n_noisy_total"]}')
    print(f'11 维活性: {s_active}')
    for j in TARGET:
        a = activity[j]
        print(f'  dim{j}: var={a["var"]:.5f} domain_d={a["domain_d"]:+.2f} seed_ratio={a["seed_ratio"]:.2f} '
              f'{"✓活性" if a["active"] else ("死" if a["dead"] else "噪声")}')
    # ===== ④ 探针复现（独立集白话人类段——18 特征 Spearman vs dim_probe.json）=====
    probe = json.load(open(BASE / 'data/independent_test/dim_probe.json', encoding='utf-8'))
    old_remaining = set(probe.get('remaining_dims', []))
    old_explained = set(probe.get('explained_dims', []))
    print(f'旧口径（18 特征单一口径）: explained {len(old_explained)}——remaining {len(old_remaining)}')
    print(f'目标 11 维是否全在 remaining: {set(TARGET) <= old_remaining}')
    print(f'S_active（阶段 C 干预对象）: {s_active}')


if __name__ == '__main__':
    main()
