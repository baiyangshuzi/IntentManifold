# -*- coding: utf-8 -*-
"""v0.89 数学推理对错判别力探测（语义层支线——纯离线——零 GPU——评审 4 条冻结版）

用户方案（v0.88 否定后的架构主张）：BGE/语言编码器与数学逻辑编码器是两类不同结构——
应分离。四步路线第一步 = 特征融合判别器。本实验测量"文本表面表征（bge 语义/风格指纹/
手写结构）对数学推理对错的可判别力上界"——双向预注册（FAIL 预期支持架构分离——PASS 进路线②）。

【预注册判据表（先写后跑——Plan 评审 + 用户评审 4 条冻结）】

| # | 判据 | 判定线 | 角色 |
|---|------|--------|------|
| M-M1 | F_all 按对 LOO acc（LR——train 折内标准化——主 acc 嵌套选 C） | acc > 0.60 且 全局标签置换 1000（固定 C=1.0）p < 0.05 | 主判 |
| M-M3 | 正确值池重分配控制（仅 M-M1 PASS 时——决定性） | acc_m3 ≥ acc_m1 − 0.05 → 关系性；acc_m3 ≤ 0.55 → 值边缘泄漏（归否定）；中间 → 混合 | 条件门 |
| M-M2 | 特征分解 | 四组 acc + Δf 报告（报告臂置换 300——预算控制） | 报告 |
| M-M4 | 对内差向量集中度 κ | 报告（与主判同向才引用） | 报告 |
| 稳健臂 | MLP 5 折 / KNN-1 | 与主判同向才引用 | 报告 |

评审冻结：
①样本量 = 3 组 × 32 对（96 对）：A 组 4 运算各 8 对 / B 组 4 运算各 8 对 / C 组 2 类各 16 对；
②错误注入五约束：e≠c、|e−c|∈[1, max(2,c//10)]、同位数、**同奇偶**（防 75% 表面 cue）、
  **同大小档**（防档位边界跨档泄漏——跨档比例 0 断言落盘）；
③LR 主 acc 嵌套选 C（train 折内 3-fold 网格 {0.01,0.1,1,10}）；置换 null 固定 C=1.0
  （声明：无信号下 acc 对 C 近似不变）；P2 末固定 C=1.0 重算主 acc 对照（差 >0.03 标记"参数敏感"）；
④KNN-1 冻结：测试对两样本分别对训练集 188 条做 cosine 1-NN——硬标签——配对得分
  s_i = 1 if (p_c, p_e) == (0, 1) else 0；
⑤方程模板去推导句（断言式——防内部一致性 cue）；竖式组排除（末位一致性 cue——理由写入报告）；
⑥M-M3 决定性控制 = 错误值从同组正确值池无放回抽取（对别的操作数是对的、对这对是错的）；
⑦对内单 token 差异硬断言（结果槽位定位替换——禁 str.replace 改值）；方向平衡（组内 16/16）；
⑧F_struct 红线：禁"结果与操作数关系"派生特征（=手写算术规则——循环论证）——11 维；
⑨reasoning_test 迁移臂排除（转路线②）；⑩结论限定功率范围（θ≤0.65——不声称绝对无信号）。

裁定阶梯：M-M1 FAIL → REJECTED（"无 ≥0.65 可读判别信号——文本表面表征不承载结果-操作数
运算关系——架构分离主张获反向支持——与 v0.88 Δg 显著为负一致"）；
M-M1 PASS ∧ M-M3 关系性 → PASS（"存在非值边缘统计规律——进路线② CodeBERT"）；
M-M3 值边缘 → 泄漏否定；M-M3 混合 → PASS-弱+声明。
"""
import sys, json, os, re, time
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path('C:/Users/bai/Desktop/小说系统')
OUT = BASE / 'data' / 'dim_analysis'
PAPER = Path('C:/Users/bai/Desktop/AB系统论文储备')
sys.path.insert(0, str(BASE / 'stage3'))

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

SEED = 20260817
SEED_LOO = SEED + 1
SEED_PERM = SEED + 2
SEED_M3 = SEED + 3
SEED_BOOT = SEED + 4
N_PER_GROUP = 32          # 3 组 × 32 对 = 96 对
N_PER_CLASS = 8           # A/B 组每类 8 对
OP_ORDER = ['add', 'sub', 'mul', 'div']
BIN_CUTS = (33, 66)
N_PERM_MAIN = 1000
N_PERM_REPORT = 300
N_JOBS = min(16, os.cpu_count() - 1)   # 电力充足——高效利用资源（分块置换并行）


def bin_of(v):
    """大小档（数值版——口径同 engine_math_geometry.bin_digits：[33,66]）"""
    return 0 if v <= BIN_CUTS[0] else (1 if v <= BIN_CUTS[1] else 2)


def error_set(c):
    """预注册注入约束：e≥0、e≠c、|e−c|∈[1,max(2,c//10)]、同位数、同奇偶、同大小档
    返回 (E, up, down)"""
    r = max(2, c // 10)
    E, up, down = [], [], []
    for e in range(max(0, c - r), c + r + 1):
        if e == c:
            continue
        if len(str(e)) != len(str(c)):
            continue
        if e % 2 != c % 2:
            continue
        if bin_of(e) != bin_of(c):
            continue
        E.append(e)
        (up if e > c else down).append(e)
    return E, up, down


def feasible(c):
    E, up, down = error_set(c)
    return len(E) >= 2 and len(up) >= 1 and len(down) >= 1


def gen_gated(rng, n, sampler, seen=None):
    """可行性闸门：每对 (a,b,c) 要求 error_set(c) 双向可行——否则重采样——上限 500；
    seen=(a,b) 组合去重（C 组短模板空间小——有放回抽样碰撞率高——防重复文本）"""
    if seen is None:
        seen = set()
    out, tries = [], 0
    while len(out) < n and tries < 500:
        tries += 1
        row = sampler(rng)
        key = (row[0], row[1])
        if key in seen:
            continue
        if feasible(row[2]):
            seen.add(key)
            out.append(row)
    assert len(out) == n, f'可行性闸门 500 次内未达成（已得 {len(out)}/{n}）'
    return out


def _sampler(r, a_range, b_range, op):
    """通用采样器：(a, b, 结果位)——结果位 = op 计算结果（sub 用 A−B——约束 A≥B+2 由 b 上限保证）"""
    a = int(r.integers(*a_range))
    b = int(r.integers(*b_range))
    c = {'add': a + b, 'sub': a - b, 'mul': a * b, 'div': a // b}[op]
    return (a, b, c)


def _div_sampler(r):
    """除法：A = B·q（q 结果位）——4 ≤ A < 120——feasible(q) 检查——上限 500 防死循环"""
    for _ in range(500):
        b = int(r.integers(2, 10))
        q = int(r.integers(2, 30))
        a = b * q
        if 4 <= a < 120 and feasible(q):
            return (a, b, q)
    raise RuntimeError('_div_sampler 500 次未找到可行对')


def gen_structured(rng, n=N_PER_CLASS):
    """组 A：算式转写（约束同 engine_math_geometry.gen_pairs + c≥2 + sub A>B+2）"""
    out, seen = {}, set()
    out['add'] = gen_gated(rng, n, lambda r: _sampler(r, (2, 60), (2, 60), 'add'), seen)
    out['sub'] = gen_gated(rng, n, lambda r: _sampler(r, (4, 60), (2, 40), 'sub'), seen)
    out['mul'] = gen_gated(rng, n, lambda r: _sampler(r, (2, 30), (2, 30), 'mul'), seen)
    out['div'] = gen_gated(rng, n, _div_sampler, seen)
    return out


def _b_sampler(r, a_range, b_range, op):
    """B 组显式采样器（先 int() 再赋值——防 walrus 绑定 np.int64）"""
    a = int(r.integers(*a_range))
    b = int(r.integers(*b_range))
    c = {'add': a + b, 'sub': a - b, 'mul': a * b}[op]
    return (a, b, c)


def _b_div_sampler(r):
    """除：结果位 = a（句末"每人分得"）——句首 a*b 固定（真值 a）"""
    a = int(r.integers(2, 12))
    b = int(r.integers(2, 12))
    return (a, b, a)


def gen_applications_pairs(rng, n=N_PER_CLASS):
    """组 B：应用题（复刻 engine_math_real.gen_applications 数值域——返回 (a,b,r) 元组——
    r=结果位（除法的结果位是 a——句末"每人分得"））"""
    names = ['小明', '小红', '小华', '小丽', '小刚']
    out, seen = {}, set()
    out['add'] = gen_gated(rng, n, lambda r: _b_sampler(r, (2, 50), (2, 50), 'add'), seen)
    out['sub'] = gen_gated(rng, n, lambda r: _b_sampler(r, (4, 50), (2, 46), 'sub'), seen)
    out['mul'] = gen_gated(rng, n, lambda r: _b_sampler(r, (2, 12), (2, 12), 'mul'), seen)
    out['div'] = gen_gated(rng, n, _b_div_sampler, seen)
    return out, names


def _c_add_sampler(r):
    x = int(r.integers(2, 20))
    b = int(r.integers(2, 30))
    return (x, b, x)


def _c_mul_sampler(r):
    x = int(r.integers(2, 12))
    a = int(r.integers(2, 9))
    return (x, a, x)


def gen_equations_pairs(rng, n=16):
    """组 C：方程（新断言式模板——去推导句——评审点 5）——结果位 x∈[2,20)/[2,12)"""
    out, seen = {}, set()
    out['add_eq'] = gen_gated(rng, n, _c_add_sampler, seen)
    out['mul_eq'] = gen_gated(rng, n, _c_mul_sampler, seen)
    return out


def tpl_A(op, a, b, r):
    opw = {'add': '加号', 'sub': '减号', 'mul': '乘号', 'div': '除号'}[op]
    return f'数字 {a} {opw} 数字 {b} 等于 数字 {r}'


def tpl_B(op, name, a, b, r):
    if op == 'add':
        return f'{name}有 {a} 个苹果，又买来 {b} 个，现在一共有 {r} 个苹果。'
    if op == 'sub':
        return f'{name}有 {a} 本书，借出 {b} 本，还剩 {r} 本书。'
    if op == 'mul':
        return f'每排有 {a} 个座位，共 {b} 排，一共有 {r} 个座位。'
    return f'{a * b} 颗糖平均分给 {b} 个小朋友，每人分得 {r} 颗糖。'   # 句首 a*b 固定——仅结果位变化


def tpl_C(op, x_true, k, r):
    """组 C：R 用真值 x 固定（错误只改结果位 r——R 不随 r 变——单 token 差异断言可过）——
    错误文本"x 加 18 等于 22，x 等于 6"——需计算 22−18≠6 才知错——无表面 cue"""
    if op == 'add_eq':
        return f'解方程：x 加 {k} 等于 {x_true + k}。解得 x 等于 {r}。'
    return f'解方程：{k} 乘 x 等于 {x_true * k}。解得 x 等于 {r}。'


# 每对的结果槽定位（前缀, 后缀）——断言用
SLOT = {'A': ('等于 数字 ', ''), 'B_add': ('一共有 ', ' 个苹果。'), 'B_sub': ('还剩 ', ' 本书。'),
        'B_mul': ('一共有 ', ' 个座位。'), 'B_div': ('分得 ', ' 颗糖。'),
        'C_add': ('解得 x 等于 ', '。'), 'C_mul': ('解得 x 等于 ', '。'),
        'C_add_eq': ('解得 x 等于 ', '。'), 'C_mul_eq': ('解得 x 等于 ', '。')}


def strip_result(t, prefix, suffix):
    pat = re.escape(prefix) + r'\d+' + re.escape(suffix)
    return re.sub(pat, prefix + '⟨R⟩' + suffix, t)


def assert_single_token_diff(tc, te, prefix, suffix):
    """硬断言：两文本仅结果槽不同（评审点 ⑦——禁 str.replace——槽位定位）"""
    assert strip_result(tc, prefix, suffix) == strip_result(te, prefix, suffix), \
        f'对内差异超过结果槽: {tc!r} vs {te!r}'
    rc = re.search(re.escape(prefix) + r'(\d+)' + re.escape(suffix), tc).group(1)
    re_ = re.search(re.escape(prefix) + r'(\d+)' + re.escape(suffix), te).group(1)
    assert rc != re_, f'结果槽未变: {tc!r} vs {te!r}'


def build_corpus(rng):
    """96 对 = 192 文本——labels 96 对 × 2（正确 0/错误 1）——断言全过——落盘冻结"""
    groups = []  # (gname, row, tpl_fn, slot_key, op_key)
    A = gen_structured(rng)
    for op in OP_ORDER:
        for a, b, c in A[op]:
            groups.append(('A', (op, a, b, c), lambda r=None, op=op, a=a, b=b: tpl_A(op, a, b, r), 'A', op))
    B, names = gen_applications_pairs(rng)
    for op in OP_ORDER:
        for a, b, r in B[op]:
            nm = str(rng.choice(names))
            groups.append(('B', (op, a, b, r), lambda r=None, op=op, nm=nm, a=a, b=b: tpl_B(op, nm, a, b, r), f'B_{op}', op))
    C = gen_equations_pairs(rng)
    for op in ['add_eq', 'mul_eq']:
        for x, k, r in C[op]:
            groups.append(('C', (op, x, k, r), lambda r=None, op=op, x=x, k=k: tpl_C(op, x, k, r), f'C_{op}', op))
    assert len(groups) == 96, f'总对数 {len(groups)} != 96'
    assert sum(1 for g in groups if g[0] == 'A') == 32 and sum(1 for g in groups if g[0] == 'B') == 32 \
        and sum(1 for g in groups if g[0] == 'C') == 32, '3 组 × 32 对断言失败'

    texts, labels, pair_ids, meta = [], [], [], []
    up_cnt = {'A': 0, 'B': 0, 'C': 0}
    for gi, (gname, row, tpl, slot_key, op) in enumerate(groups):
        r_true = row[3]
        E, up, down = error_set(r_true)
        pool = up if up_cnt[gname] < 16 else down
        if not pool:
            pool = down if up_cnt[gname] < 16 else up
        e = int(rng.choice(pool))
        up_cnt[gname] += 1 if e > r_true else 0
        tc = tpl(r_true)
        te = tpl(e)
        prefix, suffix = SLOT[slot_key]
        assert_single_token_diff(tc, te, prefix, suffix)
        texts += [tc, te]
        labels += [0, 1]
        pair_ids += [gi, gi]
        meta.append({'group': gname, 'op': op, 'a': row[1], 'b': row[2],
                     'r_true': r_true, 'r_err': e, 'direction': 'up' if e > r_true else 'down',
                     'slot': slot_key})
    labels = np.array(labels)
    pair_ids = np.array(pair_ids)
    # 方向平衡断言（组内 16/16——差 ≤2 预注册）
    for gname in ('A', 'B', 'C'):
        us = sum(1 for m in meta if m['group'] == gname and m['direction'] == 'up')
        assert abs(us - 16) <= 2, f'{gname} 方向平衡失败: {us}'
    # 奇偶恒定/同档断言（对级）
    for m in meta:
        assert (m['r_true'] % 2) == (m['r_err'] % 2), f'奇偶恒定失败: {m}'
        assert bin_of(m['r_true']) == bin_of(m['r_err']), f'同档失败: {m}'
    # 无重复文本
    assert len(set(texts)) == len(texts), '存在重复文本'
    # 值域 overlap（KS——报告不判据）
    from scipy import stats as sc
    c_vals = np.array([m['r_true'] for m in meta])
    e_vals = np.array([m['r_err'] for m in meta])
    ks = sc.ks_2samp(c_vals, e_vals)
    overlap = np.intersect1d(c_vals, e_vals).size / min(len(np.unique(c_vals)), len(np.unique(e_vals)) + 1)
    return {'texts': texts, 'labels': labels, 'pair_ids': pair_ids, 'meta': meta,
            'groups': groups, 'c_vals': c_vals, 'e_vals': e_vals,
            'ks_p': float(ks.pvalue), 'overlap_frac': float(overlap)}


def struct_features(texts, ops):
    """F_struct 11 维（评审点 ⑧红线：禁结果-操作数关系特征）——op one-hot(4) + 结果位数 +
    结果奇偶 + 大小档 one-hot(3) + 数字总数 + 文本长度——op 从生成 meta 给（表面属性）"""
    n = len(texts)
    F = np.zeros((n, 11))
    op_idx = {'add': 0, 'sub': 1, 'mul': 2, 'div': 3,
              'add_eq': 0, 'mul_eq': 2}        # 组 C 方程映射到加减乘除键
    for i, (t, op) in enumerate(zip(texts, ops)):
        digs = re.findall(r'\d+', t)
        r = int(digs[-1])                      # 结果槽 = 最后一个数字
        F[i, op_idx[op]] = 1
        F[i, 4] = len(str(r))
        F[i, 5] = r % 2
        F[i, 6 + bin_of(r)] = 1
        F[i, 9] = len(digs)
        F[i, 10] = len(t)
    return F


def extract_features(texts, enc, disc):
    """F_bge(512) + F_fp(64) + F_struct(11) + F_all(587)"""
    from para_dimensions import fingerprint
    import torch
    sv = enc.encode(texts, normalize_embeddings=True, batch_size=16,
                    show_progress_bar=False, device='cpu').astype(np.float32)
    with torch.no_grad():
        F_fp = fingerprint(torch.from_numpy(sv), disc).detach().cpu().numpy()
    F_bge = sv
    return {'bge': F_bge, 'fp': F_fp}


def lr_fit_predict(Xtr, ytr, Xte, C=1.0):
    """train 折内标准化 + LR——返回正类概率"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=C, max_iter=1000).fit(sc.transform(Xtr), ytr)
    return clf.predict_proba(sc.transform(Xte))[:, 1]


def lr_nested(Xtr, ytr, Xte, Cs=(0.01, 0.1, 1.0, 10.0)):
    """评审点 ③：主 acc 嵌套选 C（train 折内 3-fold——网格）——测试折不参与"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import KFold
    best_c, best_a = Cs[0], -1.0
    for C in Cs:
        accs = []
        for tr2, va2 in KFold(3, shuffle=True, random_state=SEED_LOO).split(Xtr):
            sc = StandardScaler().fit(Xtr[tr2])
            clf = LogisticRegression(C=C, max_iter=1000).fit(sc.transform(Xtr[tr2]), ytr[tr2])
            accs.append(clf.score(sc.transform(Xtr[va2]), ytr[va2]))
        a = float(np.mean(accs))
        if a > best_a:
            best_a, best_c = a, C
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=best_c, max_iter=1000).fit(sc.transform(Xtr), ytr)
    return clf.predict_proba(sc.transform(Xte))[:, 1]


def pair_loo_cv(X, y, pair_ids, fit_predict, verbose=False):
    """按对 LOO（96 折——test 1 对 2 条——train 95 对）——s_i=1[p(x_c)>p(x_e)]，tie→0.5
    返回 acc/per_fold/per_group/fold_type_counts"""
    n_pairs = len(np.unique(pair_ids))
    per_fold = np.zeros(n_pairs)
    for pi in range(n_pairs):
        te = np.where(pair_ids == pi)[0]
        tr = np.setdiff1d(np.arange(len(y)), te)
        p = fit_predict(X[tr], y[tr], X[te])
        s = 1.0 if p[0] > p[1] else (0.5 if p[0] == p[1] else 0.0)
        per_fold[pi] = s
    acc = float(np.mean(per_fold))
    types = {'1.0': int(np.sum(per_fold == 1.0)), '0.5': int(np.sum(per_fold == 0.5)),
             '0.0': int(np.sum(per_fold == 0.0))}
    return {'acc': acc, 'per_fold': per_fold, 'fold_type_counts': types}


def group_acc(per_fold, meta, n_pairs=96):
    """按 A/B/C 组聚合 acc（报告）"""
    out = {}
    for gname in ('A', 'B', 'C'):
        idx = [i for i in range(n_pairs) if meta[i]['group'] == gname]
        out[gname] = round(float(np.mean(per_fold[idx])), 4)
    return out


def _perm_chunk(X, y, pair_ids, seed_base, n):
    """单 worker 块：n 次全局标签置换（固定 C=1.0——同一 LOO 机械）——joblib 分块并行"""
    rng = np.random.default_rng(seed_base)
    out = []
    for _ in range(n):
        yp = rng.permutation(y)
        out.append(pair_loo_cv(X, yp, pair_ids,
                               lambda a, b, c: lr_fit_predict(a, b, c, C=1.0))['acc'])
    return out


def perm_global_labels(X, y, pair_ids, n_perm=N_PERM_MAIN, seed=SEED_PERM, C=1.0, n_jobs=None):
    """全局标签置换（保持 48/48——破坏对内配对——同一 LOO 机械——固定 C=1.0）
    p = (1 + #{acc_null ≥ acc_obs}) / (1 + n_perm)——joblib 多进程分块并行（资源利用率）"""
    from joblib import Parallel, delayed
    if n_jobs is None:
        n_jobs = min(12, os.cpu_count() - 1)
    rng = np.random.default_rng(seed)
    obs = pair_loo_cv(X, y, pair_ids, lambda a, b, c: lr_fit_predict(a, b, c, C=C))['acc']
    n_chunk = max(1, n_perm // n_jobs)
    chunks = Parallel(n_jobs=n_jobs, backend='loky', verbose=1)(
        delayed(_perm_chunk)(X, y, pair_ids, seed + 1000 + i, n_chunk)
        for i in range(n_jobs))
    nulls = np.array([a for c in chunks for a in c])
    p = (1 + int(np.sum(nulls >= obs))) / (1 + n_perm)
    return {'obs_acc': obs, 'p': float(p), 'p95': float(np.percentile(nulls, 95)),
            'mean_null': float(np.mean(nulls))}


def m3_value_pool_reassign(corpus, rng):
    """M-M3 决定性控制（评审点 ⑥）：错误值从同组正确值池 {c_j} 无放回抽取——
    约束 e'_i ∉ {c_i, e_i}、同位数/同奇偶/同档（与 c_i 比较）——冲突退化有放回（计数≤2）
    值边缘模型（"c-token 好"）崩溃到 50%——关系性模型维持高 acc"""
    meta = corpus['meta']
    n_pairs = len(meta)
    new_texts = list(corpus['texts'])
    conflicts = 0
    rng2 = np.random.default_rng(SEED_M3)
    for gname in ('A', 'B', 'C'):
        idx = [i for i, m in enumerate(meta) if m['group'] == gname]
        pool = [meta[i]['r_true'] for i in idx]
        order = rng2.permutation(idx)
        used = set()
        for i in order:
            m = meta[i]
            cand = [c for c in pool if c != m['r_true'] and c != m['r_err']
                    and len(str(c)) == len(str(m['r_true']))
                    and c % 2 == m['r_true'] % 2 and bin_of(c) == bin_of(m['r_true'])
                    and c not in used]
            if not cand:
                conflicts += 1
                cand = [c for c in pool if c != m['r_true'] and c != m['r_err']
                        and len(str(c)) == len(str(m['r_true']))
                        and c % 2 == m['r_true'] % 2 and bin_of(c) == bin_of(m['r_true'])]
                if not cand:
                    continue
            e2 = int(rng2.choice(cand))
            used.add(e2)
            # 槽位替换（禁 str.replace 改值——用 strip/正则按槽位重建）
            t = corpus['texts'][2 * i + 1]
            prefix, suffix = SLOT[m['slot']]
            pat = re.escape(prefix) + r'\d+' + re.escape(suffix)
            new_texts[2 * i + 1] = re.sub(pat, f'{prefix}{e2}{suffix}', t)
    assert conflicts <= 2, f'M-M3 冲突数 {conflicts} > 2——停'
    return new_texts, conflicts


def m4_direction(X, pair_ids):
    """M-M4：对内差向量 v_i = n(x_c) − n(x_e)——κ = mean((v_i·ĵ)²)——集中度报告"""
    def l2norm(X):
        return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    n_pairs = len(np.unique(pair_ids))
    V = np.zeros((n_pairs, X.shape[1]))
    for pi in range(n_pairs):
        te = np.where(pair_ids == pi)[0]
        V[pi] = l2norm(X[te[0]:te[0] + 1])[0] - l2norm(X[te[1]:te[1] + 1])[0]
    V = l2norm(V)
    jhat = l2norm(V.mean(axis=0, keepdims=True))[0]
    kappa = float(np.mean((V @ jhat) ** 2))
    return {'kappa': kappa, 'V': V, 'jhat': jhat}


def mlp_probe_cv5(X, y, pair_ids, dim, seed=42):
    """稳健臂：dim→64→1、LayerNorm+ReLU、BCEWithLogits、AdamW 3e-4/1e-4、15ep、组分层 5 折"""
    import torch
    import torch.nn as nn
    from sklearn.model_selection import StratifiedKFold
    torch.manual_seed(seed)
    device = 'cpu'
    Xt = torch.tensor(X, dtype=torch.float32).to(device)
    yt = torch.tensor(y, dtype=torch.float32).to(device)
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    accs = []
    for tr, va in skf.split(X, y):
        model = nn.Sequential(nn.Linear(dim, 64), nn.LayerNorm(64), nn.ReLU(),
                              nn.Linear(64, 1)).to(device)
        lossf = nn.BCEWithLogitsLoss()
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        for ep in range(15):
            model.train()
            p = torch.randperm(len(tr), device=device)
            for i in range(0, len(tr), 64):
                bi = tr[p[i:i + 64]]
                loss = lossf(model(Xt[bi]).squeeze(-1), yt[bi])
                opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            prob = torch.sigmoid(model(Xt[va]).squeeze(-1)).numpy()
        # 按对聚合（同一折内——对可能跨折被切——用 pair 平均近似）
        # 简化：按条 acc（稳健臂——报告角色）
        accs.append(((prob > 0.5) == y[va]).mean())
    return {'acc5_mean': round(float(np.mean(accs)), 4), 'acc5_std': round(float(np.std(accs)), 4)}


def knn1_pair_loo_full(X, y, pair_ids):
    """KNN-1 报告臂（评审点 ④ 冻结）：测试对两样本分别对训练集做 cosine 1-NN——
    硬标签——配对得分 s_i = 1 if (p_c, p_e) == (0, 1) else 0"""
    def l2norm(X):
        return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    Xn = l2norm(X)
    n_pairs = len(np.unique(pair_ids))
    n_ok = 0
    for pi in range(n_pairs):
        te = np.where(pair_ids == pi)[0]
        tr = np.setdiff1d(np.arange(len(y)), te)
        sims = Xn[te] @ Xn[tr].T
        nn_idx = sims.argmax(axis=1)
        p_c = y[tr][nn_idx[0]]            # 正确文本最近邻标签
        p_e = y[tr][nn_idx[1]]            # 错误文本最近邻标签
        if p_c == 0 and p_e == 1:
            n_ok += 1
    return round(n_ok / n_pairs, 4)


def load_or_build_corpus(rng):
    """P0 缓存恢复：corpus.json 存在则加载（冻结）——否则生成+落盘"""
    f = OUT / 'math_correctness_corpus.json'
    if f.exists():
        d = json.loads(f.read_text(encoding='utf-8'))
        print(f'  缓存命中 math_correctness_corpus.json（{len(d["meta"])} 对——冻结）')
        return {'texts': d['texts'], 'labels': np.array(d['labels']),
                'pair_ids': np.array(d['pair_ids']), 'meta': d['meta'],
                'ks_p': d['ks_p'], 'overlap_frac': d['overlap_frac']}
    print('  生成 96 对语料:')
    corpus = build_corpus(rng)
    print(f'  文本 {len(corpus["texts"])} 条——对 {len(corpus["meta"])} 对——'
          f'标签 正确 {sum(corpus["labels"] == 0)} / 错误 {sum(corpus["labels"] == 1)}')
    print(f'  KS(正确值, 错误值) p={corpus["ks_p"]:.3f} overlap={corpus["overlap_frac"]:.3f}')
    f.write_text(json.dumps(
        {'texts': corpus['texts'], 'labels': corpus['labels'].tolist(),
         'pair_ids': corpus['pair_ids'].tolist(), 'meta': corpus['meta'],
         'ks_p': corpus['ks_p'], 'overlap_frac': corpus['overlap_frac'],
         'error_rules': 'e≠c, |e−c|∈[1,max(2,c//10)], 同位数, 同奇偶, 同大小档, e≥0',
         'n_pairs': 96, 'seed': SEED}, ensure_ascii=False, indent=1), encoding='utf-8')
    print('  落盘（冻结）✓')
    return corpus


def load_or_build_features(texts, enc, disc, ops):
    """P1 缓存恢复：features.npz 存在则加载——否则编码+落盘"""
    f = OUT / 'math_correctness_features.npz'
    if f.exists():
        z = np.load(f)
        print('  缓存命中 math_correctness_features.npz')
        F = {'bge': z['bge'], 'fp': z['fp'], 'struct': z['struct'], 'all_': z['all_']}
        fp_deg = float(np.mean(F['fp'].std(axis=0) < 0.05))
        return F, fp_deg, float(F['bge'].std(axis=0).min())
    print('  编码 192 条:')
    t0 = time.time()
    F = extract_features(texts, enc, disc)
    F_struct = struct_features(texts, ops)
    F_all = np.hstack([F['bge'], F['fp'], F_struct])
    F['struct'] = F_struct
    F['all_'] = F_all
    print(f'  F_bge {F["bge"].shape} F_fp {F["fp"].shape} F_struct {F_struct.shape} '
          f'F_all {F_all.shape}（{time.time() - t0:.0f}s）')
    fp_deg = float(np.mean(F['fp'].std(axis=0) < 0.05))
    bge_std_min = float(F['bge'].std(axis=0).min())
    assert bge_std_min > 1e-6, 'F_bge 退化（全零嵌入）'
    np.savez(f, bge=F['bge'], fp=F['fp'], struct=F_struct, all_=F_all)
    print('  落盘 ✓')
    return F, fp_deg, bge_std_min


def run_perm_batch(X, y, pair_ids, n_run, n_jobs, seed_base):
    """joblib 多进程分块置换（电力充足——N_JOBS 进程）——余数块补齐（1000 = 11×90+10——不丢次）"""
    from joblib import Parallel, delayed
    n_chunk = max(1, n_run // n_jobs)
    n_chunks = (n_run + n_chunk - 1) // n_chunk
    sizes = [n_chunk] * (n_chunks - 1) + [n_run - n_chunk * (n_chunks - 1)]
    chunks = Parallel(n_jobs=n_jobs, backend='loky', verbose=1)(
        delayed(_perm_chunk)(X, y, pair_ids, seed_base + 1000 + i, s)
        for i, s in enumerate(sizes))
    return np.array([a for c in chunks for a in c])


def perm_cached(X, y, pair_ids, perm_f, target, seed_base, n_jobs, label):
    """置换批处理缓存：已有 + 补跑至 target——每批落盘（断点恢复）——满 target 算 p"""
    nulls = np.array([])
    if perm_f.exists():
        nulls = np.load(perm_f)['nulls']
    if len(nulls) < target:
        n_run = target - len(nulls)
        print(f'  {label}: 缓存 {len(nulls)}/{target}——补跑 {n_run} 次（{n_jobs} 进程）', flush=True)
        new = run_perm_batch(X, y, pair_ids, n_run, n_jobs, seed_base + len(nulls))
        nulls = np.concatenate([nulls, new])
        np.savez(perm_f, nulls=nulls)
        print(f'  {label}: {len(nulls)}/{target}——落盘 ✓', flush=True)
    obs = pair_loo_cv(X, y, pair_ids, lambda a, b, c: lr_fit_predict(a, b, c, C=1.0))['acc']
    p = (1 + int(np.sum(nulls >= obs))) / (1 + target)
    return nulls, obs, float(p)


def main():
    print('===== v0.89 数学推理对错判别力探测（判据预注册——分块断点恢复） =====')
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    from para_dimensions import load_models
    enc, disc = load_models('cpu')
    rng = np.random.default_rng(SEED)

    # ===== P0 语料生成（冻结——缓存恢复） =====
    print('\nP0 语料生成（96 对——seed 20260817）:')
    corpus = load_or_build_corpus(rng)

    # ===== P1 特征提取（缓存恢复） =====
    print('\nP1 特征提取:')
    ops = []
    for m in corpus['meta']:
        ops += [m['op'], m['op']]
    F, fp_deg, bge_std_min = load_or_build_features(corpus['texts'], enc, disc, ops)
    F_struct = F['struct']
    F_all = F['all_']
    print(f'  F_fp 逐维 std<0.05 占比: {fp_deg:.3f}——F_bge std min: {bge_std_min:.4f}')

    y = corpus['labels']
    pair_ids = corpus['pair_ids']
    meta = corpus['meta']
    n_pairs = len(meta)

    # ===== P2 主判 =====
    print('\nP2 主判（按对 LOO 96 折——LR 嵌套选 C）:')
    t1 = time.time()
    res_m1 = pair_loo_cv(F_all, y, pair_ids, lr_nested)
    gacc = group_acc(res_m1['per_fold'], meta)
    print(f'  M-M1: acc={res_m1["acc"]:.4f}（>0.60 门） per_fold 类型 {res_m1["fold_type_counts"]} '
          f'分组 {gacc}（{time.time() - t1:.0f}s）')

    # 固定 C=1.0 对照（评审点 ③——参数敏感性）
    t1b = time.time()
    res_fixed = pair_loo_cv(F_all, y, pair_ids, lambda a, b, c: lr_fit_predict(a, b, c, C=1.0))
    c_sens = abs(res_fixed['acc'] - res_m1['acc'])
    print(f'  固定 C=1.0 对照: acc={res_fixed["acc"]:.4f}——Δ={c_sens:+.4f}'
          f'（{"参数敏感——结论存疑" if c_sens > 0.03 else "参数不敏感"}）')

    print('  置换检验（全局标签打乱 1000——固定 C=1.0——批处理缓存）:')
    nulls_main, obs_main, p_main = perm_cached(
        F_all, y, pair_ids, OUT / 'math_correctness_perm_main.npz',
        N_PERM_MAIN, SEED_PERM, N_JOBS, '主置换 F_all')
    perm = {'obs_acc': obs_main, 'p': p_main, 'p95': float(np.percentile(nulls_main, 95)),
            'mean_null': float(np.mean(nulls_main)), 'n_null': int(len(nulls_main))}
    print(f'  perm: obs_acc={perm["obs_acc"]:.4f} p={perm["p"]:.4f} null mean={perm["mean_null"]:.4f} '
          f'p95={perm["p95"]:.4f}')
    m_m1_pass = res_m1['acc'] > 0.60 and perm['p'] < 0.05
    print(f'  M-M1: {"PASS" if m_m1_pass else "FAIL"}（acc={res_m1["acc"]:.4f}>0.60: '
          f'{"✓" if res_m1["acc"] > 0.60 else "✗"} p={perm["p"]:.4f}<0.05: '
          f'{"✓" if perm["p"] < 0.05 else "✗"}）')

    # ===== M-M2 特征分解（报告——四组 LOO + 300 置换） =====
    print('\nM-M2 特征分解（四组——报告）:')
    m2 = {}
    for name, Xf in (('bge', F['bge']), ('fp', F['fp']), ('struct', F_struct), ('all', F_all)):
        r = pair_loo_cv(Xf, y, pair_ids, lr_nested)
        nulls_r, obs_r, p_r = perm_cached(
            Xf, y, pair_ids, OUT / f'math_correctness_perm_{name}.npz',
            N_PERM_REPORT, SEED_PERM + 7, N_JOBS, f'M-M2 {name}')
        m2[name] = {'acc': round(r['acc'], 4), 'perm_p': round(p_r, 4),
                    'group_acc': group_acc(r['per_fold'], meta)}
        print(f'  {name}: acc={r["acc"]:.4f} p={p_r:.4f} 分组 {m2[name]["group_acc"]}')
    m2['delta_fusion'] = round(m2['all']['acc'] - max(m2['bge']['acc'], m2['fp']['acc'],
                                                      m2['struct']['acc']), 4)
    print(f'  Δf（融合−最好单组）= {m2["delta_fusion"]:+.4f}')

    # ===== M-M3 决定性控制（预注册：仅 M-M1 PASS 时执行——FAIL 时跳过并记录） =====
    if m_m1_pass:
        print('\nM-M3 正确值池重分配控制（M-M1 PASS——条件门触发）:')
        new_texts, conflicts = m3_value_pool_reassign(corpus, rng)
        print(f'  冲突数 {conflicts}（≤2 断言）')
        F3 = extract_features(new_texts, enc, disc)
        F3_struct = struct_features(new_texts, ops)
        F3_all = np.hstack([F3['bge'], F3['fp'], F3_struct])
        res_m3 = pair_loo_cv(F3_all, y, pair_ids, lr_nested)
        m3_verdict = ('关系性' if res_m3['acc'] >= res_m1['acc'] - 0.05
                      else ('值边缘泄漏' if res_m3['acc'] <= 0.55 else '混合'))
        print(f'  M-M3: acc={res_m3["acc"]:.4f}（对照 M-M1 {res_m1["acc"]:.4f}）→ {m3_verdict}')
    else:
        res_m3 = {'acc': None}
        m3_verdict = 'skipped（M-M1 FAIL——预注册条件门未触发）'
        print(f'  M-M3: {m3_verdict}')

    # ===== M-M4 几何 =====
    m4_bge = m4_direction(F['bge'], pair_ids)
    m4_fp = m4_direction(F['fp'], pair_ids)
    print(f'  M-M4 κ: bge={m4_bge["kappa"]:.4f} fp={m4_fp["kappa"]:.4f}')

    # ===== bootstrap CI（折结果重采样——按对） =====
    rngb = np.random.default_rng(SEED_BOOT)
    accs = []
    for _ in range(2000):
        idx = rngb.integers(0, n_pairs, n_pairs)
        accs.append(float(np.mean(res_m1['per_fold'][idx])))
    boot_ci = [round(float(np.percentile(accs, 2.5)), 4), round(float(np.percentile(accs, 97.5)), 4)]
    print(f'  bootstrap 95% CI: {boot_ci}')

    # ===== 裁定 =====
    if not m_m1_pass:
        overall = 'REJECTED'
        interp = ('按对分割下，bge 语义嵌入/风格指纹/手写表面结构对数学推理对错无可读判别'
                  '（acc ≤ 0.60，置换不显著——功率范围 θ ≤ 0.65）——文本表面表征不承载'
                  '结果-操作数运算关系——数学逻辑判定需专用结构表征或符号计算——'
                  '架构分离主张获反向支持（与 v0.88 Δg 显著为负一致）')
    elif m3_verdict == '关系性':
        overall = 'PASS'
        interp = ('存在非值边缘驱动的可读统计规律（可能 bge 事实性记忆）——'
                  '进入路线②：CodeBERT 第二编码器 + 专用对错判别器')
    elif m3_verdict == '值边缘泄漏':
        overall = '泄漏否定'
        interp = '数字值域统计驱动（M-M3 正确值池重分配后 acc 崩溃至 0.55 以下）——归入否定'
    else:
        overall = 'PASS-弱'
        interp = 'M-M3 混合——存在部分关系信号但非决定性——声明后按 PASS-弱处理'
    print(f'\n  总判定: {overall}')
    print(f'  解释: {interp}')

    # ===== P3 稳健臂 =====
    print('\nP3 稳健臂:')
    mlp = {}
    for name, Xf in (('bge', F['bge']), ('fp', F['fp']), ('struct', F_struct), ('all', F_all)):
        mlp[name] = mlp_probe_cv5(Xf, y, pair_ids, Xf.shape[1])
        print(f'  MLP {name}: {mlp[name]}')
    knn_bge = knn1_pair_loo_full(F['bge'], y, pair_ids)
    knn_all = knn1_pair_loo_full(F_all, y, pair_ids)
    print(f'  KNN-1: bge={knn_bge} all={knn_all}')

    # ===== 图 =====
    colors = ['#1f6fb2', '#e67e22', '#27ae60', '#8e44ad']
    names_cn = ['bge 语义', '指纹 64', '结构 11', '融合 587']
    fig, ax = plt.subplots(figsize=(9, 5.5))
    accs4 = [m2[n]['acc'] for n in ('bge', 'fp', 'struct', 'all')]
    x = np.arange(4)
    ax.bar(x, accs4, 0.5, color=colors)
    for xi, (a, n_) in enumerate(zip(accs4, ('bge', 'fp', 'struct', 'all'))):
        ax.text(xi, a + 0.008, f'{a:.3f}\np={m2[n_]["perm_p"]:.3f}', ha='center', fontsize=8)
    ax.axhline(0.60, color='r', ls='--', lw=1.2)
    ax.text(3.35, 0.605, '0.60 门', color='r', fontsize=8)
    ax.axhline(0.5, color='gray', ls=':', lw=1)
    # M-M3 对照（仅 M-M1 PASS 时有值——否则占位标注 skipped）
    if m_m1_pass:
        ax.bar(4.2, res_m3['acc'], 0.5, color='#c0392b', alpha=0.8)
        ax.text(4.2, res_m3['acc'] + 0.008, f'M-M3\n{res_m3["acc"]:.3f}', ha='center', fontsize=8)
    else:
        ax.bar(4.2, 0.5, 0.5, color='#7f8c8d', alpha=0.4)
        ax.text(4.2, 0.52, 'M-M3\nskipped', ha='center', fontsize=8)
    ax.set_xticks(np.arange(5))
    ax.set_xticklabels(names_cn + ['M-M3 控制'])
    ax.set_ylabel('按对 LOO acc')
    ax.set_title(f'数学对错判别力探测——LR 按对 LOO（总判定: {overall}）——CI {boot_ci}')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(PAPER / 'fig_correctness_probe.png', dpi=150)
    plt.close()

    # 几何图
    fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5.2))
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2).fit(m4_bge['V'])
    Z = pca.transform(m4_bge['V'])
    for gi, gname in enumerate(('A', 'B', 'C')):
        idx = [i for i, m in enumerate(meta) if m['group'] == gname]
        axes2[0].scatter(Z[idx, 0], Z[idx, 1], s=36, color=colors[gi], label=f'组 {gname}', alpha=0.8)
    j2 = pca.transform(m4_bge['jhat'].reshape(1, -1))
    axes2[0].annotate('', xy=(j2[0, 0] * 4, j2[0, 1] * 4), xytext=(0, 0),
                      arrowprops=dict(color='k', lw=1.8, ls='--'))
    axes2[0].legend(fontsize=9)
    axes2[0].set_title('对内差向量 v_i（bge——PCA 2D——虚线=共同方向 jhat）')
    axes2[0].grid(alpha=0.3)
    # 对内 vs 跨对距离
    def l2n(X):
        return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    Xb = l2n(F['bge'])
    within = [np.linalg.norm(Xb[2 * i] - Xb[2 * i + 1]) for i in range(n_pairs)]
    cross = []
    for i in range(min(60, n_pairs)):
        for j in range(i + 1, min(60, n_pairs)):
            cross.append(np.linalg.norm(Xb[2 * i] - Xb[2 * j]))
    axes2[1].hist(within, bins=20, alpha=0.6, label='对内（正确 vs 错误）', color='#1f6fb2')
    axes2[1].hist(cross, bins=40, alpha=0.4, label='跨对', color='#7f8c8d')
    axes2[1].axvline(np.mean(within), color='#1f6fb2', ls='--')
    axes2[1].legend(fontsize=8)
    axes2[1].set_title(f'嵌入距离分布（bge）——kappa={m4_bge["kappa"]:.3f}')
    axes2[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PAPER / 'fig_correctness_geometry.png', dpi=150)
    plt.close()

    # 分组 × 特征热图 + F_struct 权重
    fig3, axes3 = plt.subplots(1, 2, figsize=(13, 5.2))
    gmat = np.array([[m2[n]['group_acc'][g] for n in ('bge', 'fp', 'struct', 'all')] for g in ('A', 'B', 'C')])
    im3 = axes3[0].imshow(gmat, cmap='Reds', vmin=0.4, vmax=0.7)
    axes3[0].set_yticks(range(3)); axes3[0].set_yticklabels(['组 A 算式', '组 B 应用题', '组 C 方程'])
    axes3[0].set_xticks(range(4)); axes3[0].set_xticklabels(names_cn)
    for i in range(3):
        for j in range(4):
            axes3[0].text(j, i, f'{gmat[i, j]:.2f}', ha='center', va='center', fontsize=9)
    axes3[0].set_title('3 组 × 4 特征 acc 热图')
    fig3.colorbar(im3, ax=axes3[0], fraction=0.046)
    # F_struct 权重（LR 在 train 全部上拟合——报告性）
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(F_struct)
    clf = LogisticRegression(C=1.0, max_iter=1000).fit(sc.transform(F_struct), y)
    wnames = ['加', '减', '乘', '除', '结果位数', '结果奇偶', '档小', '档中', '档大', '数字总数', '文本长度']
    coef = clf.coef_[0]
    maxc = max(abs(c) for c in coef)
    axes3[1].barh(np.arange(11), coef, color='#1f6fb2')
    for i, c in enumerate(coef):
        axes3[1].text(c + (0.003 if c >= 0 else -0.003), i, f'{c:.3f}',
                      va='center', fontsize=7, ha='left' if c >= 0 else 'right')
    axes3[1].set_yticks(np.arange(11)); axes3[1].set_yticklabels(wnames)
    axes3[1].set_xlim(-maxc * 1.8 - 0.02, maxc * 1.8 + 0.02)
    axes3[1].set_title(f'F_struct LR 系数（11 维——表面特征——max|w|={maxc:.4f}≈无判别力）')
    axes3[1].grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(PAPER / 'fig_correctness_groups.png', dpi=150)
    plt.close()

    # ===== 落盘 =====
    out = {
        'meta': {'seed': SEED, 'n_pairs': n_pairs, 'groups': 'A 算式/4类×8 + B 应用题/4类×8 + C 方程/2类×16',
                 'error_rules': 'e≠c, |e−c|∈[1,max(2,c//10)], 同位数, 同奇偶, 同大小档, e≥0',
                 'design_note': '按对 LOO——LR 主估计器嵌套选 C——置换 null 固定 C——M-M3 正确值池重分配决定性控制'},
        'p0': {'n_texts': len(corpus['texts']), 'n_pairs': n_pairs,
               'ks_p': round(corpus['ks_p'], 4), 'overlap_frac': round(corpus['overlap_frac'], 4),
               'direction_balance': {g: sum(1 for m in meta if m['group'] == g and m['direction'] == 'up')
                                     for g in ('A', 'B', 'C')},
               'asserts': '单token差异/奇偶恒定/同档/无重复/方向平衡——全过'},
        'features': {'fp_std_lt005_frac': round(fp_deg, 4), 'bge_std_min': round(bge_std_min, 6)},
        'm_m1': {'acc': round(res_m1['acc'], 4), 'perm_p': round(perm['p'], 4),
                 'perm_p95': round(perm['p95'], 4), 'perm_mean_null': round(perm['mean_null'], 4),
                 'boot_ci': boot_ci, 'group_acc': gacc,
                 'fold_type_counts': res_m1['fold_type_counts'],
                 'acc_fixed_C_control': round(res_fixed['acc'], 4),
                 'c_sensitivity': round(c_sens, 4), 'pass': bool(m_m1_pass)},
        'm_m2': {k: v for k, v in m2.items()},
        'm_m3': ({'acc': round(res_m3['acc'], 4), 'm_m1_acc': round(res_m1['acc'], 4),
                  'verdict': m3_verdict, 'conflicts': conflicts,
                  'note': '错误值从同组正确值池抽取——值边缘模型崩溃到 50%——关系性模型维持'}
                 if m_m1_pass else {'verdict': m3_verdict,
                                    'note': '预注册条件门：M-M1 FAIL 不执行'}),
        'm_m4': {'kappa_bge': round(m4_bge['kappa'], 4), 'kappa_fp': round(m4_fp['kappa'], 4)},
        'mlp_robust': mlp,
        'knn1': {'bge': knn_bge, 'all': knn_all},
        'verdict': {'overall': overall, 'interpretation': interp},
    }
    (OUT / 'math_correctness.json').write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                               encoding='utf-8')
    print('\n落盘 math_correctness.json + fig_correctness_* × 3 ✓')


if __name__ == '__main__':
    main()
