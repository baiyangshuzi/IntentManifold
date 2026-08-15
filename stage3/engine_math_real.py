# -*- coding: utf-8 -*-
"""v0.85-5 真实推导文本检验（支线 2——纯离线）

从模板算式走向真实推导文本——检验运算类型差异是否保持：
A. 应用题（加/减/乘/除——数量关系句——12 条/类——生成规则化）
B. 竖式计算描述（进位/借位过程——12 条/类——规则化）
C. 方程推导（一元方程步骤——加/乘型各 12 条——两类型检验）

方法：区间映射（[33,66] 基准口径）→ bge → fingerprint → sil/acc vs 占位词基线
（同文本数字——运算符词换随机拼音——Δ 判据——同 v0.85 主判）
"""
import sys, json, re
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path
import numpy as np

BASE = Path('C:/Users/bai/Desktop/小说系统')
OUT = BASE / 'data' / 'dim_analysis'
sys.path.insert(0, str(BASE / 'stage3'))
from engine_math_geometry import fingerprints_of, sil_acc, permutation_test, RANDOM_PH
from engine_math_robust import bin_at


def gen_applications(rng, n=12):
    """应用题（数量关系句——加/减/乘/除）——返回 {op: [文本]}"""
    items = {'加': [], '减': [], '乘': [], '除': []}
    names = ['小明', '小红', '小华', '小丽', '小刚']
    while len(items['加']) < n:
        a, b = rng.integers(2, 50, 2)
        items['加'].append(f'{rng.choice(names)}有 {a} 个苹果，又买来 {b} 个，现在一共有 {a + b} 个苹果。')
    while len(items['减']) < n:
        a, b = rng.integers(2, 50, 2)
        if a > b:
            items['减'].append(f'{rng.choice(names)}有 {a} 本书，借出 {b} 本，还剩 {a - b} 本书。')
    while len(items['乘']) < n:
        a, b = rng.integers(2, 12, 2)
        items['乘'].append(f'每排有 {a} 个座位，共 {b} 排，一共有 {a * b} 个座位。')
    while len(items['除']) < n:
        a, b = rng.integers(2, 12, 2)
        items['除'].append(f'{a * b} 颗糖平均分给 {b} 个小朋友，每人分得 {a} 颗糖。')
    return items


def gen_vertical(rng, n=12):
    """竖式计算描述（进位/借位）——加/乘两型"""
    items = {'加法竖式': [], '乘法竖式': []}
    while len(items['加法竖式']) < n:
        a, b = rng.integers(23, 99, 2)
        s = str(a + b)
        items['加法竖式'].append(
            f'列竖式计算 {a} 加 {b}：先加个位 {a % 10} 加 {b % 10} 得 {(a % 10 + b % 10) % 10}'
            f'，写 {(a % 10 + b % 10) % 10} 进 {(a % 10 + b % 10) // 10}，'
            f'再加十位得 {s}。')
    while len(items['乘法竖式']) < n:
        a, b = rng.integers(12, 49, 2)
        items['乘法竖式'].append(
            f'列竖式计算 {a} 乘 {b}：先用 {b % 10} 乘 {a} 得 {a * (b % 10)}，'
            f'再用 {b // 10} 乘 {a} 得 {a * (b // 10)}，最后相加得 {a * b}。')
    return items


def gen_equations(rng, n=12):
    """方程推导（一元——加/乘两型）"""
    items = {'加法方程': [], '乘法方程': []}
    while len(items['加法方程']) < n:
        x = rng.integers(1, 20)
        b = rng.integers(2, 30)
        items['加法方程'].append(
            f'解方程：x 加 {b} 等于 {x + b}。两边同时减 {b}，得 x 等于 {x}。')
    while len(items['乘法方程']) < n:
        x = rng.integers(1, 12)
        a = rng.integers(2, 9)
        items['乘法方程'].append(
            f'解方程：{a} 乘 x 等于 {a * x}。两边同时除以 {a}，得 x 等于 {x}。')
    return items


def to_lists(d):
    texts, labels = [], []
    for i, (k, v) in enumerate(d.items()):
        for t in v:
            texts.append(t)
            labels.append(i)
    return texts, np.array(labels)


def main():
    import os
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    from para_dimensions import load_models
    enc, disc = load_models('cpu')
    rng = np.random.default_rng(2026)

    print('===== v0.85-5 真实推导文本检验（支线 2） =====')
    all_res = {}
    for name, gen in (('应用题', gen_applications), ('竖式', gen_vertical), ('方程', gen_equations)):
        d = gen(rng)
        texts, labels = to_lists(d)
        # 区间映射（基准口径）
        btexts = [bin_at(t, [33, 66]) for t in texts]
        # 占位词基线：运算符词换随机拼音（同文本同数字）
        reps = {'加': RANDOM_PH[0], '减': RANDOM_PH[1], '乘': RANDOM_PH[2], '除': RANDOM_PH[3],
                '加法竖式': RANDOM_PH[0], '乘法竖式': RANDOM_PH[2],
                '加法方程': RANDOM_PH[0], '乘法方程': RANDOM_PH[2]}
        ptexts = []
        for t in texts:
            for k, v in reps.items():
                t = t.replace(k, v)
            ptexts.append(t)
        ptexts = [bin_at(t, [33, 66]) for t in ptexts]
        F_t = fingerprints_of(btexts, enc, disc)
        F_p = fingerprints_of(ptexts, enc, disc)
        st_t = sil_acc(F_t, labels)
        st_p = sil_acc(F_p, labels)
        d_sil = st_t['sil'] - st_p['sil']
        perm = permutation_test(F_t, labels, n_perm=500)
        ok = st_t['sil'] > 0.25 and d_sil > 0.10 and perm['p'] < 0.05
        all_res[name] = {'true_sil': round(st_t['sil'], 3), 'ph_sil': round(st_p['sil'], 3),
                         'delta_sil': round(float(d_sil), 3), 'acc': round(st_t['acc_lda'], 3),
                         'perm_p': round(perm['p'], 4), 'pass': bool(ok), 'n': len(texts)}
        print(f'  {name}（n={len(texts)}）: 真词 sil={st_t["sil"]:.3f} vs 占位词 {st_p["sil"]:.3f}——'
              f'Δsil={d_sil:+.3f} acc={st_t["acc_lda"]:.3f} 置换 p={perm["p"]:.4f}——'
              f'{"PASS" if ok else "FAIL"}')

    n_pass = sum(1 for v in all_res.values() if v['pass'])
    print(f'\n真实推导文本判定: {n_pass}/{len(all_res)} PASS——'
          f'{"运算类型差异在真实文本保持" if n_pass >= 2 else "真实文本运算类型差异弱（模板效应？）"}')
    (OUT / 'math_real.json').write_text(json.dumps(all_res, ensure_ascii=False, indent=1), encoding='utf-8')
    print('落盘 math_real.json ✓')


if __name__ == '__main__':
    main()
