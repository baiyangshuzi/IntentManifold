# -*- coding: utf-8 -*-
"""v0.74 阶段 D：跨域/跨模型维度一致性（离线）

①中文侧：bilingual zh（human 10 vs AI 20）——11 维人机差异 d
②双模型：independent_test（DS vs Qwen vs human——32 对三侧）——维度级人机差异方向一致性
③限制标注：英文无判别器（bge 嵌入级——非同一实体——仅报告方向）
"""
import sys, json, os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path
import numpy as np
from scipy import stats as sc

BASE = Path(os.environ.get('INTENT_DYNAMICS_BASE', Path(__file__).resolve().parent.parent))
OUT = BASE / 'data' / 'dim_analysis'
sys.path.insert(0, str(BASE / 'stage3'))
TARGET = [5, 10, 11, 22, 26, 34, 43, 46, 48, 52, 59]


def cohens_d(a, b):
    return (np.mean(a) - np.mean(b)) / np.sqrt((np.var(a) + np.var(b)) / 2 + 1e-9)


def main():
    d = np.load(OUT / 'fp_matrix.npz')
    fp = d['fp']
    rows = json.loads((OUT / 'rows.json').read_text(encoding='utf-8'))

    # ① bilingual zh：human vs AI——11 维人机差异
    hu = [i for i, r in enumerate(rows) if r['source'] == 'bilingual_zh' and r['side'] == 'human']
    ai = [i for i, r in enumerate(rows) if r['source'] == 'bilingual_zh' and r['side'] == 'ai']
    print(f'bilingual zh: human 句元 {len(hu)}——AI {len(ai)}')
    d1 = {}
    for j in TARGET:
        dj = cohens_d(fp[hu, j], fp[ai, j])
        _, p = sc.ttest_ind(fp[hu, j], fp[ai, j], equal_var=False)
        d1[j] = {'d': round(float(dj), 3), 'p': round(float(p), 4),
                 'human': round(float(fp[hu, j].mean()), 4), 'ai': round(float(fp[ai, j].mean()), 4)}
    print('① bilingual zh 11 维人机差异:')
    for j in TARGET:
        v = d1[j]
        print(f'  dim{j}: human={v["human"]:.4f} ai={v["ai"]:.4f} d={v["d"]:+.2f} p={v["p"]:.3f}')

    # ② independent_test 双模型（B 白话——human vs DS vs Qwen）
    res = {}
    for side in ('human', 'ai', 'qwen'):
        idx = [i for i, r in enumerate(rows) if r['source'] == 'independent_test'
               and r['side'] == side and 'B-' in r['doc']]
        res[side] = idx
    print(f'independent_test B 侧句元: human {len(res["human"])}/DS {len(res["ai"])}/Qwen {len(res["qwen"])}')
    d2 = {}
    for j in TARGET:
        d_h_ds = cohens_d(fp[res['human'], j], fp[res['ai'], j])
        d_h_q = cohens_d(fp[res['human'], j], fp[res['qwen'], j])
        d2[j] = {'d_human_vs_DS': round(float(d_h_ds), 3), 'd_human_vs_Qwen': round(float(d_h_q), 3),
                 'dir_consistent': bool((d_h_ds > 0) == (d_h_q > 0))}
    n_cons = sum(1 for j in TARGET if d2[j]['dir_consistent'])
    print(f'② 双模型方向一致: {n_cons}/11——不一致维度: {[j for j in TARGET if not d2[j]["dir_consistent"]]}')
    for j in TARGET:
        v = d2[j]
        print(f'  dim{j}: DS d={v["d_human_vs_DS"]:+.2f} Qwen d={v["d_human_vs_Qwen"]:+.2f} '
              f'{"一致" if v["dir_consistent"] else "✗不一致"}')

    out = {'bilingual_zh': d1, 'dual_model': d2,
           'n_dir_consistent': n_cons,
           'english_note': '英文无判别器（bge 嵌入级——非同一实体）——维度级跨语言不可直接比较——仅作方向参考',
           'criteria': 'D1：跨模型方向一致 → 深层组织特征'}
    (OUT / 'cross_validate.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print('落盘 cross_validate.json ✓')


if __name__ == '__main__':
    main()
