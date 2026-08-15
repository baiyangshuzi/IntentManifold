# -*- coding: utf-8 -*-
"""v0.85-4 区间映射稳健性验证（支线 1——纯离线）

问题：区间映射（数字→小/中/大）可能引入新的人为结构——多法一致才确定
"运算符语义独立几何结构"稳定。

三法对照（同真词四类/占位词四类——seed 2026 同一批数据）：
A. 区间映射（基准——切点 [33,66]）
B. 切点敏感性：更细（[25,50,75] 四档）/更粗（[50] 两档）/切点偏移（[40,70]）
C. 数字归一化嵌入：数字替换为 [0,1] 归一化小数表示（"数字 0.23 加号 数字 0.45 等于 数字 0.68"——
   保留数值信息但消除量级差异——不同于区间映射的离散化）

判据：三种方法下 Δsil 均 > 0.10 且真词 sil > 0.25 → 运算符语义独立结构稳健
"""
import sys, json
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path
import numpy as np
import re

BASE = Path('C:/Users/bai/Desktop/小说系统')
OUT = BASE / 'data' / 'dim_analysis'
sys.path.insert(0, str(BASE / 'stage3'))
from engine_math_geometry import (gen_pairs, make_structured, fingerprints_of,
                                  l2norm, sil_acc, permutation_test, OPS,
                                  RANDOM_PH, SEED, N_PER)


def bin_at(text, cuts):
    """数字→档位词（切点 cuts 列表——分 len(cuts)+1 档）"""
    names = ['极', '小', '中', '大', '巨'][:len(cuts) + 1]
    def rep(m):
        v = int(m.group())
        for i, c in enumerate(cuts):
            if v <= c:
                return names[i]
        return names[-1]
    return re.sub(r'\d+', rep, text)


def norm01(text, vmax):
    """数字→归一化小数（v/vmax 保留数值信息消除量级）"""
    def rep(m):
        v = int(m.group())
        return f'{v / vmax:.2f}'
    return re.sub(r'\d+', rep, text)


def main():
    import os
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    from para_dimensions import load_models
    enc, disc = load_models('cpu')

    rng = np.random.default_rng(SEED)
    pairs = gen_pairs(rng)
    true_texts, true_labels = make_structured(pairs, OPS)
    ph_words = dict(zip(OPS.keys(), RANDOM_PH))
    ph_texts, ph_labels = make_structured(pairs, ph_words)

    methods = {
        'A_区间映射基准([33,66])': lambda t: bin_at(t, [33, 66]),
        'B1_四档([25,50,75])': lambda t: bin_at(t, [25, 50, 75]),
        'B2_两档([50])': lambda t: bin_at(t, [50]),
        'B3_切点偏移([40,70])': lambda t: bin_at(t, [40, 70]),
        'C_归一化小数(vmax=200)': lambda t: norm01(t, 200),
        'C2_归一化小数(vmax=1000)': lambda t: norm01(t, 1000),
    }
    print('===== v0.85-4 区间映射稳健性（三法对照） =====')
    results = {}
    for name, fn in methods.items():
        mt = [fn(t) for t in true_texts]
        mp = [fn(t) for t in ph_texts]
        F_mt = fingerprints_of(mt, enc, disc)
        F_mp = fingerprints_of(mp, enc, disc)
        st_t = sil_acc(F_mt, true_labels)
        st_p = sil_acc(F_mp, ph_labels)
        d_sil = st_t['sil'] - st_p['sil']
        perm = permutation_test(F_mt, true_labels, n_perm=500)
        ok = st_t['sil'] > 0.25 and d_sil > 0.10 and perm['p'] < 0.05
        results[name] = {'true_sil': round(st_t['sil'], 3), 'ph_sil': round(st_p['sil'], 3),
                         'delta_sil': round(float(d_sil), 3), 'perm_p': round(perm['p'], 4),
                         'pass': bool(ok)}
        print(f'  {name}: 真词 sil={st_t["sil"]:.3f} vs 占位词 {st_p["sil"]:.3f}——'
              f'Δsil={d_sil:+.3f} 置换 p={perm["p"]:.4f}——{"PASS" if ok else "FAIL"}')

    n_pass = sum(1 for v in results.values() if v['pass'])
    print(f'\n三法一致判定: {n_pass}/{len(results)} PASS——'
          f'{"稳健（运算符语义独立结构多法一致）" if n_pass >= 4 else "不稳定（需进一步审查）"}')

    (OUT / 'math_robust.json').write_text(
        json.dumps({'methods': results, 'n_pass': n_pass}, ensure_ascii=False, indent=1), encoding='utf-8')
    print('落盘 math_robust.json ✓')


if __name__ == '__main__':
    main()
