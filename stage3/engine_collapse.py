# -*- coding: utf-8 -*-
"""v0.88 数学运算跨层级坍缩向量预实验（纯离线——零 GPU——评审 5 条修订版）

用户假说：数学运算的意图不藏在某一层，而藏在"从语言到符号"的位移方向（坍缩向量）里。
C = F_L4 − F_L1——望远镜恒等：(F_L2−F_L1)+(F_L3−F_L2)+(F_L4−F_L3) = F_L4−F_L1——
"4 层阶梯"对主判数学上等价于双层设计——中层级仅逐级诊断贡献（预注册声明）。

【预注册判据表（先写后跑——评审 5 条吸收）】

| # | 判据 | 判定线 |
|---|------|--------|
| M-C0 | 有效性（非判定） | median ‖D‖（真词臂 48 阶梯——未归一化端点位移 D = n(F_L4)−n(F_L1)，非单位向量 C）< 0.15 → INVALID（位移退化为噪声——不称否定） |
| M-C1 | 跨实例稳定性 | 残差空间 within ≥ 0.5 且 cross-op shuffle（L4 端 op 标签置换 1000 次 seed+1）p < 0.05 |
| M-C2 | 运算特异性（主判） | g_res ≥ 0.15 且 Δg = g_res,real − g_res,ph ≥ 0.10 且 op 标签置换 1000（seed+2）p < 0.05 |
| M-C3 | 对齐（仅报告） | abs(cos) > 0.25 且 bootstrap CI 下限 > 0.10 标记；极性对照 HUMAN_ORG/AI_FEAT |
| 512 臂 | 判别器贡献（仅报告） | 只报置换 p 与 Δg——无绝对阈值（512 维随机基线 0.035 尺度不同） |

评审吸收：
①占位臂必须用同一真词臂 Ĉ 残差化（Δg 同基准差量——占位臂禁用自有 Ĉ_ph）；
②条件消融（仅 PARTIAL 执行）预注册判据写死——零额外编码（共享池——现成指纹拼接）；
③PARTIAL 精确措辞："运算特异方向存在于语言实体层，但未观察到从语言到符号的抽象化
  过程中产生额外的运算特异几何结构"；
④Ĉ = L2norm( (1/48) Σ C_i )（全部 48 个真词臂 C 算术平均——非逐 op 平均再平均）；
⑤M-C0 检查未归一化位移 ‖D‖（C 是单位向量范数恒为 1——不能用于 M-C0）；
⑥共享内容池（同 (A,B,物体,变量组) 跨 4 运算——A 值域分布逐点一致——C 值域望远镜抵消）；
⑦P0 改为 L4-only/L2-only 同层占位 token sil/acc 预检（原坍缩空间口径必触发换词）；
⑧cross-op shuffle 为主 null（same-op 降级诊断——与假说方向相反）；
⑨坍缩定义唯一化：C = D/‖D‖（逐级归一化差之和 ≠ 端点定义——变体仅诊断）。

总判定：M-C1 ∧ M-C2 → STRONG（主张覆盖"模板+token 联合承载"——token 单独由 Δg 承担）；
g_res ≥ 0.15 ∧ 置换 p<0.05 ∧ Δg < 0.10 ∧ M-C1 → PARTIAL（精确措辞见③）；
g_res < 0.15 或 M-C1 FAIL → REJECTED；M-C0 FAIL → INVALID。
PARTIAL 时执行条件消融（C_mix1 = n(F_L4,mul) − n(F_L1,add) + 对称 C_mix2——同一 Ĉ 残差化——
接近 C_add（Δcos ≥ 0.15 且 CI 不含混）→ "L1 模板主导"；接近 C_mul → "L4 符号主导"；Δcos < 0.15 → "混合"）。
"""
import sys, json, os
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

SEED = 20260816
OPS = ['add', 'sub', 'mul', 'div']
OP_CN = {'add': '加', 'sub': '减', 'mul': '乘', 'div': '除以'}
OP_SYM = {'add': '+', 'sub': '-', 'mul': '×', 'div': '÷'}
PH = ['甲', '乙', '丙', '丁']
PH_RANDOM = ['wu', 'ji', 'pan', 'kuo']
N_LADDER = 12
OBJ_POOL = ['苹果', '石子', '书本', '水杯', '铅笔', '糖果',
            '积木', '树叶', '椅子', '饼干', '信封', '棋子']
VAR_POOL = [('a', 'b', 'c'), ('x', 'y', 'z'), ('m', 'n', 'p'), ('u', 'v', 'w'),
            ('e', 'f', 'g'), ('r', 's', 't'), ('h', 'i', 'j'), ('k', 'l', 'q'),
            ('c', 'd', 'f'), ('o', 'p', 'q'), ('g', 't', 'u'), ('v', 'w', 'x')]
L1_TPL = {'add': '{A}个{obj}和{B}个{obj}放在一起',
          'sub': '从{A}个{obj}中拿走{B}个{obj}',
          'mul': '{A}个袋子里各有{B}个{obj}',
          'div': '把{A}个{obj}平均分成{B}份'}
L2_TPL = '{A}{opw}{B}等于{C}'
L3_TPL = '{A} {sym} {B} = {C}'
L4_TPL = '{x} {sym} {y} = {z}'
TARGET11 = [5, 10, 11, 22, 26, 34, 43, 46, 48, 52, 59]
HUMAN_ORG = [10, 11, 34, 46, 48, 59]
AI_FEAT = [22, 26, 43, 52, 5]
RANDOM_NULL_64 = 0.10


def num2cn(n):
    """0-999 规范中文数字（十/二十/一百零五——评审：mul C 最大 180）"""
    d0 = '零一二三四五六七八九'
    if n < 10:
        return d0[n]
    if n < 20:
        return '十' + (d0[n % 10] if n % 10 else '')
    h, rem = divmod(n, 100)
    if h:
        t, o = divmod(rem, 10)
        s = d0[h] + '百'
        if t == 0 and o:
            s += '零' + d0[o]
        elif t:
            s += ('' if t == 1 else d0[t]) + '十' + (d0[o] if o else '')
        return s
    t, o = divmod(rem, 10)
    return d0[t] + '十' + (d0[o] if o else '')


def gen_pool(rng):
    """共享内容池：12 个整除对——A∈[2,20] B∈[2,9] B|A A//B∈[2,9]（A≥B 自动——sub C≥2 无零值）"""
    valid = [(a, b) for a in range(2, 21)
             for b in range(2, 10) if a % b == 0 and 2 <= a // b <= 9]
    assert len(valid) >= N_LADDER, f'共享池候选不足: {len(valid)}'
    idx = rng.choice(len(valid), N_LADDER, replace=False)
    return [valid[i] for i in idx]


def build_ladders(pool, rng):
    """4 op × 12 阶梯——共享内容池——每 op 内物体/变量组无放回置换（频率平衡——断言）"""
    ladders = {}
    for op in OPS:
        objs = rng.permutation(OBJ_POOL).tolist()
        vars_ = [tuple(v) for v in rng.permutation(VAR_POOL).tolist()]
        assert sorted(objs) == sorted(OBJ_POOL) and len(set(vars_)) == N_LADDER
        rows = []
        for i, (a, b) in enumerate(pool):
            c = {'add': a + b, 'sub': a - b, 'mul': a * b, 'div': a // b}[op]
            rows.append({'op': op, 'A': a, 'B': b, 'C': int(c),
                         'obj': objs[i], 'vars': vars_[i]})
        ladders[op] = rows
    return ladders


def make_texts(ladders, word_map, sym_map):
    """展平 4 op × 12 阶梯 × 4 层 = 192 条——返回 (texts, op_labels(48), ladder_idx(48), levels(48,4)顺序)"""
    texts, meta = [], []
    for oi, op in enumerate(OPS):
        for li, row in enumerate(ladders[op]):
            x, y, z = row['vars']
            l1 = L1_TPL[op].format(A=num2cn(row['A']), B=num2cn(row['B']), obj=row['obj'])
            l2 = L2_TPL.format(A=num2cn(row['A']), opw=word_map[op], B=num2cn(row['B']),
                               C=num2cn(row['C']))
            l3 = L3_TPL.format(A=row['A'], sym=sym_map[op], B=row['B'], C=row['C'])
            l4 = L4_TPL.format(x=x, sym=sym_map[op], y=y, z=z)
            texts += [l1, l2, l3, l4]
            meta.append({'op': oi, 'ladder': oi * N_LADDER + li})
    op_labels = np.array([m['op'] for m in meta])          # (48,)
    ladder_idx = np.array([m['ladder'] for m in meta])     # (48,)
    return texts, op_labels, ladder_idx


def encode(texts, enc, disc):
    """(n,64) 原始指纹 + (n,512) 归一化 bge 嵌入（batch 16 CPU——支持臂零额外成本）"""
    import torch
    from para_dimensions import fingerprint
    sv = enc.encode(texts, normalize_embeddings=True, batch_size=16,
                    show_progress_bar=False, device='cpu').astype(np.float32)
    SV = torch.from_numpy(sv)
    with torch.no_grad():
        F = fingerprint(SV, disc).detach().cpu().numpy()
    return F, sv


def l2norm(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


def arr4_of(F):
    """(192,64) → (48,4,64) 阶梯×层"""
    return F.reshape(-1, 4, F.shape[1])


def collapse_vectors(A4):
    """D = n(F_L4) − n(F_L1)（未归一化——M-C0 用）；C = D/‖D‖（单位方向——主判用）"""
    L1 = l2norm(A4[:, 0, :])
    L4 = l2norm(A4[:, 3, :])
    D = L4 - L1
    C = l2norm(D)
    return D, C


def common_direction(C_real):
    """Ĉ = L2norm( (1/48) Σ C_i )（全部真词臂 C 算术平均——非逐 op 平均——评审点 4）"""
    return l2norm(C_real.mean(axis=0, keepdims=True))[0]


def residualize(C, Chat):
    """Gram-Schmidt：C⊥ = L2norm(C − (C·Ĉ)Ĉ)——真词臂与占位臂用同一 Ĉ（评审点 1）"""
    proj = np.outer(C @ Chat, Chat)
    return l2norm(C - proj)


def g_stat(Cres, op_labels, unit_ids=None):
    """within = (1/4)Σ_op mean_{i<j} cos（组大小自适应——bootstrap 重采样后 op 组可变）；
    unit_ids 用于 bootstrap：排除同一单元重复出现的对（重复对 cos=1 只进 within 不进 between——
    使 bootstrap g 偏置上移——按单元身份排除后 CI 无偏）"""
    if unit_ids is None:
        unit_ids = np.arange(Cres.shape[0])
    per_op = []
    for oi in range(4):
        mask = op_labels == oi
        C = Cres[mask]
        ids = unit_ids[mask]
        sims = []
        for i in range(C.shape[0]):
            for j in range(i + 1, C.shape[0]):
                if ids[i] == ids[j]:
                    continue
                sims.append(C[i] @ C[j])
        per_op.append(float(np.mean(sims)) if sims else 0.0)
    within = float(np.mean(per_op))
    between_all = []
    for oi in range(4):
        for oj in range(oi + 1, 4):
            Ci = Cres[op_labels == oi]
            Cj = Cres[op_labels == oj]
            between_all.append(float(np.mean(Ci @ Cj.T)))
    between = float(np.mean(between_all))
    return {'within': within, 'between': between, 'g': within - between,
            'per_op_within': {OPS[oi]: round(v, 4) for oi, v in enumerate(per_op)}}


def perm_label(Cres, op_labels, n_perm=1000, seed=SEED + 2):
    """op 标签置换 1000 次——p = mean(g_null ≥ g_obs)"""
    rng = np.random.default_rng(seed)
    obs = g_stat(Cres, op_labels)['g']
    nulls = []
    for _ in range(n_perm):
        perm = rng.permutation(op_labels)
        nulls.append(g_stat(Cres, perm)['g'])
    nulls = np.array(nulls)
    return {'obs_g': obs, 'p': float(np.mean(nulls >= obs)), 'p95': float(np.percentile(nulls, 95)),
            'mean_null': float(np.mean(nulls))}


def perm_shuffle_cross(A4_real, op_labels, n_perm=1000, seed=SEED + 1):
    """cross-op shuffle（主 null——评审点 5）：σ 作用于 L1 端 op 标签——同时打破模板与 token
    注意：shuffled 向量同样用同一 Ĉ 残差化（与 M-C1 口径一致）"""
    rng = np.random.default_rng(seed)
    Chat = Chat_from(A4_real)
    _, C_aligned = collapse_vectors(A4_real)
    obs_within = g_stat(residualize(C_aligned, Chat), op_labels)['within']
    L4n = l2norm(A4_real[:, 3, :])
    L1n = l2norm(A4_real[:, 0, :])
    nulls = []
    for _ in range(n_perm):
        sigma = rng.permutation(4)
        L1s = np.zeros_like(L4n)
        for oi in range(4):
            sel = op_labels == oi
            L1s[sel] = L1n[sigma[oi] * N_LADDER:(sigma[oi] + 1) * N_LADDER]
        Cshuf = residualize(l2norm(L4n - L1s), Chat)
        nulls.append(g_stat(Cshuf, op_labels)['within'])
    nulls = np.array(nulls)
    return {'obs_within': obs_within, 'p': float(np.mean(nulls >= obs_within)),
            'mean_null': float(np.mean(nulls))}


def Chat_from(A4):
    _, C = collapse_vectors(A4)
    return common_direction(C)


def perm_shuffle_sameop(A4_real, op_labels, n_perm=1000, seed=SEED + 5):
    """same-op shuffle（仅诊断——评审点 5 降级）：同 op 内 L1 阶梯置换——测试数字配对依赖"""
    rng = np.random.default_rng(seed)
    A4 = l2norm(A4_real)
    obs_within = g_stat(residualize(collapse_vectors(A4_real)[1], Chat_from(A4_real)), op_labels)['within']
    nulls = []
    for _ in range(n_perm):
        Cshuf = np.zeros_like(A4[:, 0, :])
        for oi in range(4):
            sel = op_labels == oi
            idx = np.nonzero(sel)[0]
            sigma = rng.permutation(N_LADDER)
            L1 = A4[idx[sigma], 0, :]
            L4 = A4[idx, 3, :]
            Cshuf[idx] = l2norm(L4 - L1)
        nulls.append(g_stat(residualize(Cshuf, Chat_from(A4_real)), op_labels)['within'])
    nulls = np.array(nulls)
    return {'obs_within': obs_within, 'p': float(np.mean(nulls >= obs_within)),
            'mean_null': float(np.mean(nulls))}


def boot_g(Cres_real, Cres_ph, op_labels, n_boot=2000, seed=SEED + 3):
    """按 48 阶梯单元重采样（配对语义——两臂同索引）——g_real/g_ph/Δg 95% CI
    注意：标签必须随 idx 重排（op_labels[idx]）——否则分组错位（v0.88 自查修复）"""
    rng = np.random.default_rng(seed)
    n = Cres_real.shape[0]
    gs_r, gs_p, gs_d = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        gs_r.append(g_stat(Cres_real[idx], op_labels[idx], unit_ids=idx)['g'])
        gs_p.append(g_stat(Cres_ph[idx], op_labels[idx], unit_ids=idx)['g'])
        gs_d.append(gs_r[-1] - gs_p[-1])
    ci = lambda a: [round(float(np.percentile(a, 2.5)), 4), round(float(np.percentile(a, 97.5)), 4)]
    return {'g_real_ci': ci(gs_r), 'g_ph_ci': ci(gs_p), 'delta_g_ci': ci(gs_d)}


def level_diag(A4_real, A4_ph, op_labels):
    """逐级 g（d12/d23/d34 归一化差——真词/占位双报 + Δg——评审点 8）+ 4 层位置 sil（含 C 值域标注）"""
    from engine_math_geometry import sil_acc
    res = {'steps': {}, 'level_pos_sil': {}}
    for s0, s1, name in [(0, 1, 'd12'), (1, 2, 'd23'), (2, 3, 'd34')]:
        dr = l2norm(l2norm(A4_real[:, s1, :]) - l2norm(A4_real[:, s0, :]))
        dp = l2norm(l2norm(A4_ph[:, s1, :]) - l2norm(A4_ph[:, s0, :]))
        gr = g_stat(residualize(dr, Chat_from(A4_real)), op_labels)
        gp = g_stat(residualize(dp, Chat_from(A4_real)), op_labels)
        res['steps'][name] = {'g_real': round(gr['g'], 4), 'g_ph': round(gp['g'], 4),
                              'delta_g': round(gr['g'] - gp['g'], 4)}
    for li, name in enumerate(['L1', 'L2', 'L3', 'L4']):
        sr = sil_acc(A4_real[:, li, :], op_labels)
        sp = sil_acc(A4_ph[:, li, :], op_labels)
        res['level_pos_sil'][name] = {'sil_real': sr['sil'], 'sil_ph': sp['sil'],
                                      'acc_lda_real': sr['acc_lda']}
    res['level_pos_sil']['note'] = 'L2/L3 含 C 值域表面特征——仅上下文不判据'
    return res


def align_report(C_real, op_labels, D_shared, n_boot=2000, seed=SEED + 4):
    """M-C3（仅报告不判据）：μ_op 与 D_shared 及 11 未解释维 |cos|——标记 0.25 且 CI 下限 > 0.10"""
    rng = np.random.default_rng(seed)
    n_per = N_LADDER
    d_u = l2norm(D_shared.reshape(1, -1))[0]
    out = {'D_shared': {}, 'target11': {'matrix': [], 'human_org': [], 'ai_feat': []},
           'random_null': RANDOM_NULL_64, 'flag_line': 0.25, 'flagged': []}
    for oi, op in enumerate(OPS):
        C = C_real[op_labels == oi]
        mu = l2norm(C.mean(axis=0, keepdims=True))[0]
        cos_d = float(abs(mu @ d_u))
        # bootstrap CI（op 内 12 阶梯重采样）
        coses = []
        for _ in range(n_boot):
            idx = rng.integers(0, n_per, n_per)
            m = l2norm(C[idx].mean(axis=0, keepdims=True))[0]
            coses.append(abs(m @ d_u))
        ci = [round(float(np.percentile(coses, 2.5)), 3), round(float(np.percentile(coses, 97.5)), 3)]
        out['D_shared'][op] = {'abs_cos': round(cos_d, 3), 'boot_ci': ci,
                               'flag': bool(cos_d > 0.25 and ci[0] > 0.10)}
        row = [round(float(abs(mu[k])), 3) for k in TARGET11]
        out['target11']['matrix'].append(row)
        out['target11']['human_org'].append([round(float(mu[k]), 3) for k in HUMAN_ORG])
        out['target11']['ai_feat'].append([round(float(mu[k]), 3) for k in AI_FEAT])
        for di, k in enumerate(TARGET11):
            if abs(mu[k]) > 0.25:
                out['flagged'].append({'op': op, 'dim': k, 'abs_cos': round(float(abs(mu[k])), 3)})
    return out


def support_512(sv4_real, sv4_ph, op_labels):
    """512 维支持臂（仅报告不判据）：同一机制——只报置换 p 与 Δg——无绝对阈值"""
    E_r = l2norm(sv4_real[:, 3, :]) - l2norm(sv4_real[:, 0, :])
    E_p = l2norm(sv4_ph[:, 3, :]) - l2norm(sv4_ph[:, 0, :])
    C_r = l2norm(E_r)
    C_p = l2norm(E_p)
    Chat = common_direction(C_r)
    Cr = residualize(C_r, Chat)
    Cp = residualize(C_p, Chat)
    gr = g_stat(Cr, op_labels)
    gp = g_stat(Cp, op_labels)
    perm = perm_label(Cr, op_labels, n_perm=1000, seed=SEED + 6)
    return {'g_real': round(gr['g'], 4), 'g_ph': round(gp['g'], 4),
            'delta_g': round(gr['g'] - gp['g'], 4), 'label_perm_p': round(perm['p'], 4),
            'note': '置换意义——无绝对阈值（512 维随机基线 0.035）'}


def ablate_mix(A4_real, op_labels):
    """条件消融（仅 PARTIAL——评审点 2 预注册判据）：C_mix1 = n(F_L4,mul) − n(F_L1,add)
    零额外编码（共享池——现成指纹拼接）——裁定 L1 模板 vs L4 符号主导"""
    n_per = N_LADDER
    A4 = l2norm(A4_real)
    Chat = Chat_from(A4_real)
    _, C = collapse_vectors(A4_real)
    mu = {}
    for oi, op in enumerate(OPS):
        mu[op] = l2norm(residualize(C[op_labels == oi], Chat).mean(axis=0, keepdims=True))[0]
    mix1 = l2norm(A4[2 * n_per:(2 + 1) * n_per, 3, :] - A4[0 * n_per:(0 + 1) * n_per, 0, :])
    mix2 = l2norm(A4[0 * n_per:(0 + 1) * n_per, 3, :] - A4[2 * n_per:(2 + 1) * n_per, 0, :])
    m1 = l2norm(residualize(mix1, Chat).mean(axis=0, keepdims=True))[0]
    m2 = l2norm(residualize(mix2, Chat).mean(axis=0, keepdims=True))[0]

    def judge(mix, label):
        c_add = float(mix @ mu['add'])
        c_mul = float(mix @ mu['mul'])
        dcos = c_add - c_mul
        verdict = 'L1 模板主导' if dcos >= 0.15 else ('L4 符号主导' if dcos <= -0.15 else '混合')
        return {'name': label, 'cos_add': round(c_add, 4), 'cos_mul': round(c_mul, 4),
                'delta_cos': round(dcos, 4), 'verdict': verdict}
    j1 = judge(m1, 'C_mix1(L1加×L4乘)')
    j2 = judge(m2, 'C_mix2(L1乘×L4加)')
    consistent = j1['verdict'] == j2['verdict']
    return {'mix1': j1, 'mix2': j2, 'consistent': bool(consistent),
            'overall': j1['verdict'] if consistent else '混合（两方向不一致）'}


def main():
    print('===== v0.88 数学运算跨层级坍缩向量预实验（判据预注册——见文件头） =====')
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    from para_dimensions import load_models
    enc, disc = load_models('cpu')

    rng = np.random.default_rng(SEED)
    pool = gen_pool(rng)
    ladders = build_ladders(pool, rng)
    print(f'共享内容池 12 对: {pool}')
    print(f'C 分布: 加[{min(r["C"] for r in ladders["add"])},{max(r["C"] for r in ladders["add"])}] '
          f'减[{min(r["C"] for r in ladders["sub"])},{max(r["C"] for r in ladders["sub"])}] '
          f'乘[{min(r["C"] for r in ladders["mul"])},{max(r["C"] for r in ladders["mul"])}] '
          f'除[{min(r["C"] for r in ladders["div"])},{max(r["C"] for r in ladders["div"])}]')

    word_ph = dict(zip(OPS, PH))
    sym_ph = dict(zip(OPS, PH))

    # ===== P0 占位词预检（L4-only/L2-only 同层——评审点 7）=====
    print('\nP0 占位词预检（L4-only/L2-only 同层 sil/acc——甲乙丙丁顺序语义检查）:')
    from engine_math_geometry import sil_acc
    ph_texts, op_labels, _ = make_texts(ladders, word_ph, sym_ph)
    F_ph, sv_ph = encode(ph_texts, enc, disc)
    A4_ph = arr4_of(F_ph)
    st_l4 = sil_acc(A4_ph[:, 3, :], op_labels)
    st_l2 = sil_acc(A4_ph[:, 1, :], op_labels)
    print(f'  L4-only 甲乙丙丁: sil={st_l4["sil"]:.3f} acc_lda={st_l4["acc_lda"]:.3f}')
    print(f'  L2-only 甲乙丙丁: sil={st_l2["sil"]:.3f} acc_lda={st_l2["acc_lda"]:.3f}')
    switched = False
    if st_l4['sil'] > 0.25 or st_l4['acc_lda'] > 0.6 or st_l2['sil'] > 0.25 or st_l2['acc_lda'] > 0.6:
        print('  → 甲乙丙丁有顺序语义——更换随机拼音 wu/ji/pan/kuo')
        word_ph = dict(zip(OPS, PH_RANDOM))
        sym_ph = dict(zip(OPS, PH_RANDOM))
        ph_texts, _, _ = make_texts(ladders, word_ph, sym_ph)
        F_ph, sv_ph = encode(ph_texts, enc, disc)
        A4_ph = arr4_of(F_ph)
        st_l4 = sil_acc(A4_ph[:, 3, :], op_labels)
        st_l2 = sil_acc(A4_ph[:, 1, :], op_labels)
        print(f'  L4-only 随机拼音: sil={st_l4["sil"]:.3f} acc_lda={st_l4["acc_lda"]:.3f}')
        print(f'  L2-only 随机拼音: sil={st_l2["sil"]:.3f} acc_lda={st_l2["acc_lda"]:.3f}')
        switched = True
    ph_ok = not (st_l4['sil'] > 0.25 or st_l4['acc_lda'] > 0.6 or st_l2['sil'] > 0.25 or st_l2['acc_lda'] > 0.6)
    print(f'  占位词基线合格: {ph_ok}')
    if not ph_ok:
        print('  ⚠ P0 未过——记录并继续（基线偏高的解读在报告中声明）')

    # ===== P1 真词臂编码 =====
    print('\nP1 真词臂编码（192 条）:')
    real_texts, _, _ = make_texts(ladders, OP_CN, OP_SYM)
    F_real, sv_real = encode(real_texts, enc, disc)
    A4_real = arr4_of(F_real)
    sv4_real = arr4_of(sv_real)
    sv4_ph = arr4_of(sv_ph)  # 占位臂 sv512 已在 P0 捕获（两分支均赋值）
    print(f'  真词臂 fp64: {F_real.shape} sv512: {sv_real.shape}')

    # ===== P2 统计 =====
    print('\nP2 统计:')
    D_real, C_real = collapse_vectors(A4_real)
    D_ph, C_ph = collapse_vectors(A4_ph)
    m_c0 = float(np.median(np.linalg.norm(D_real, axis=1)))
    per_op_D = {op: round(float(np.mean(np.linalg.norm(D_real[op_labels == oi], axis=1))), 3)
                for oi, op in enumerate(OPS)}
    print(f'  M-C0 有效性: median ‖D‖={m_c0:.3f}（<0.15 → INVALID） per-op: {per_op_D}')

    Chat = common_direction(C_real)
    kappa = float(((C_real @ Chat) ** 2).mean())
    print(f'  共同分量占比 κ = mean((C·Ĉ)²) = {kappa:.3f}')
    Cres_real = residualize(C_real, Chat)
    Cres_ph = residualize(C_ph, Chat)

    gr = g_stat(Cres_real, op_labels)
    gp = g_stat(Cres_ph, op_labels)
    delta_g = gr['g'] - gp['g']
    print(f'  真词臂（残差空间）: within={gr["within"]:.3f} between={gr["between"]:.3f} '
          f'g={gr["g"]:.3f} per_op={gr["per_op_within"]}')
    print(f'  占位臂（同一 Ĉ）: within={gp["within"]:.3f} between={gp["between"]:.3f} g={gp["g"]:.3f}')
    print(f'  Δg = {delta_g:+.3f}')

    pl = perm_label(Cres_real, op_labels, n_perm=1000)
    print(f'  标签置换（真词臂）: p={pl["p"]:.4f}（null p95={pl["p95"]:.3f} mean={pl["mean_null"]:.3f}）')

    cs = perm_shuffle_cross(A4_real, op_labels, n_perm=1000)
    print(f'  cross-op shuffle（M-C1 null）: obs_within={cs["obs_within"]:.3f} p={cs["p"]:.4f}')
    so = perm_shuffle_sameop(A4_real, op_labels, n_perm=1000)
    print(f'  same-op shuffle（仅诊断）: obs_within={so["obs_within"]:.3f} p={so["p"]:.4f}')

    bci = boot_g(Cres_real, Cres_ph, op_labels)
    print(f'  bootstrap（48 阶梯单元 2000）: g_real CI={bci["g_real_ci"]} g_ph CI={bci["g_ph_ci"]} '
          f'Δg CI={bci["delta_g_ci"]}')

    ld = level_diag(A4_real, A4_ph, op_labels)
    print(f'  逐级诊断: {ld["steps"]}')
    print(f'  层级位置 sil: {ld["level_pos_sil"]}')

    from engine_ratio_validate import get_D_shared
    D_shared = get_D_shared()
    ar = align_report(C_real, op_labels, D_shared)
    print(f'  M-C3 D_shared: {ar["D_shared"]}')
    print(f'  M-C3 target11 flagged: {ar["flagged"]}')

    sup = support_512(sv4_real, sv4_ph, op_labels)
    print(f'  512 维支持臂: {sup}')

    # ===== 裁定 =====
    c1 = gr['within'] >= 0.5 and cs['p'] < 0.05
    c2_g = gr['g'] >= 0.15
    c2_d = delta_g >= 0.10
    c2_p = pl['p'] < 0.05
    m_c2 = c2_g and c2_d and c2_p
    m_c0_ok = m_c0 >= 0.15
    if not m_c0_ok:
        overall = 'INVALID'
        interp = 'M-C0 失效：L4 与 L1 指纹近重合——位移退化为噪声——不称否定'
    elif c1 and m_c2:
        overall = 'STRONG'
        interp = ('坍缩方向在残差空间稳定且显著区分运算（超出占位词基线 Δg≥0.10）——'
                  '主张覆盖"模板+token 联合承载"，token 单独增量由 Δg 承担——'
                  '最低门槛：不声称捕捉真实数学推理轨迹或意图本体')
    elif c2_g and c2_p and (not c2_d) and c1:
        overall = 'PARTIAL'
        interp = ('运算特异方向存在于语言实体层，但未观察到从语言到符号的抽象化过程中'
                  '产生额外的运算特异几何结构（Δg<0.10）')
    else:
        overall = 'REJECTED'
        interp = 'g_res < 0.15 或 M-C1 未过——坍缩方向未被确证为运算特异'
    print(f'\n  M-C1: {"PASS" if c1 else "FAIL"}'
          f'（within={gr["within"]:.3f}≥0.5: {"✓" if gr["within"] >= 0.5 else "✗"} '
          f'cross-shuffle p={cs["p"]:.4f}<0.05: {"✓" if cs["p"] < 0.05 else "✗"}）')
    print(f'  M-C2: {"PASS" if m_c2 else "FAIL"}'
          f'（g={gr["g"]:.3f}≥0.15: {"✓" if c2_g else "✗"} '
          f'Δg={delta_g:+.3f}≥0.10: {"✓" if c2_d else "✗"} '
          f'perm p={pl["p"]:.4f}<0.05: {"✓" if c2_p else "✗"}）')
    print(f'  总判定: {overall}')
    print(f'  解释: {interp}')

    ab = None
    if overall == 'PARTIAL':
        print('\n  → 执行条件消融（预注册判据——零额外编码）:')
        ab = ablate_mix(A4_real, op_labels)
        print(f'  {json.dumps(ab, ensure_ascii=False, indent=1)}')

    # ===== P3 图 =====
    from sklearn.decomposition import PCA
    colors = ['#1f6fb2', '#e67e22', '#27ae60', '#8e44ad']
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    for ax, Z, title, draw_d in (
            (axes[0], C_real, '真词臂 坍缩方向（原始空间 PCA）', True),
            (axes[1], Cres_real, '真词臂 坍缩方向（残差空间——主判所在）', False),
            (axes[2], C_ph, '占位臂 坍缩方向（原始空间 PCA）', False)):
        pca = PCA(n_components=2).fit(Z)
        Zp = pca.transform(Z)
        for k in range(4):
            m = op_labels == k
            ax.scatter(Zp[m, 0], Zp[m, 1], s=36, color=colors[k],
                       label=OP_CN[OPS[k]] if draw_d else None, alpha=0.8)
            mu2 = pca.transform(l2norm(Z[m].mean(axis=0, keepdims=True)))
            ax.annotate('', xy=(mu2[0, 0], mu2[0, 1]),
                        xytext=(0, 0), arrowprops=dict(color=colors[k], alpha=0.6, lw=1.6))
        if draw_d:
            d2 = pca.components_ @ l2norm(D_shared.reshape(1, -1))[0]
            ax.plot([0, d2[0] * 6], [0, d2[1] * 6], 'k--', lw=1.4, label='D_shared(×6)')
            ax.legend(fontsize=8)
        ax.set_title(title)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PAPER / 'fig_collapse_pca.png', dpi=150)
    plt.close()

    # fig_collapse_cos.png：均值方向 cos 热图 + g/Δg 条带 + 逐级诊断
    fig2, axes2 = plt.subplots(1, 3, figsize=(17, 5.2))
    mu_mat = np.zeros((4, 4))
    for oi in range(4):
        for oj in range(4):
            m1 = l2norm(Cres_real[op_labels == oi].mean(axis=0, keepdims=True))[0]
            m2 = l2norm(Cres_real[op_labels == oj].mean(axis=0, keepdims=True))[0]
            mu_mat[oi, oj] = m1 @ m2
    im = axes2[0].imshow(mu_mat, cmap='RdBu_r', vmin=-1, vmax=1)
    axes2[0].set_xticks(range(4)); axes2[0].set_xticklabels([OP_CN[o] for o in OPS])
    axes2[0].set_yticks(range(4)); axes2[0].set_yticklabels([OP_CN[o] for o in OPS])
    for i in range(4):
        for j in range(4):
            axes2[0].text(j, i, f'{mu_mat[i, j]:.2f}', ha='center', va='center', fontsize=9)
    axes2[0].set_title('残差空间 4×4 均值方向 cos（真词臂）')
    fig2.colorbar(im, ax=axes2[0], fraction=0.046)
    bars = ['g_real', 'g_ph', 'Δg']
    vals = [gr['g'], gp['g'], delta_g]
    cis = [(bci['g_real_ci'][0], bci['g_real_ci'][1]), (bci['g_ph_ci'][0], bci['g_ph_ci'][1]),
           (bci['delta_g_ci'][0], bci['delta_g_ci'][1])]
    x = np.arange(3)
    axes2[1].bar(x, vals, 0.5, color=['#1f6fb2', '#7f8c8d', '#27ae60'])
    for xi, (v, ci) in enumerate(zip(vals, cis)):
        axes2[1].plot([xi, xi], [ci[0], ci[1]], 'k-', lw=2)
        axes2[1].plot(xi, ci[0], 'k_', ms=8)
        axes2[1].plot(xi, ci[1], 'k_', ms=8)
        axes2[1].text(xi, v + 0.02, f'{v:+.3f}', ha='center', fontsize=9)
    axes2[1].axhline(0.15, color='r', ls='--', lw=1)
    axes2[1].axhline(0.10, color='orange', ls='--', lw=1)
    axes2[1].text(2.4, 0.155, '0.15', color='r', fontsize=8)
    axes2[1].text(2.4, 0.105, '0.10', color='orange', fontsize=8)
    axes2[1].set_xticks(x); axes2[1].set_xticklabels(bars)
    axes2[1].set_title('g / Δg（残差空间——bootstrap 95% CI 误差棒）')
    axes2[1].grid(axis='y', alpha=0.3)
    steps = ld['steps']
    sx = np.arange(3)
    gv = [steps[s]['g_real'] for s in ['d12', 'd23', 'd34']]
    pv = [steps[s]['g_ph'] for s in ['d12', 'd23', 'd34']]
    axes2[2].bar(sx - 0.18, gv, 0.36, label='真词', color='#1f6fb2')
    axes2[2].bar(sx + 0.18, pv, 0.36, label='占位', color='#7f8c8d')
    for si, (g, p) in enumerate(zip(gv, pv)):
        axes2[2].text(si, max(g, p) + 0.01, f'Δ={g - p:+.2f}', ha='center', fontsize=8)
    axes2[2].set_xticks(sx); axes2[2].set_xticklabels(['L1→L2', 'L2→L3', 'L3→L4'])
    axes2[2].set_title('逐级诊断 g（Δ=该级 token 贡献）')
    axes2[2].legend(fontsize=8)
    axes2[2].grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(PAPER / 'fig_collapse_cos.png', dpi=150)
    plt.close()

    # fig_collapse_align.png：D_shared 对齐条带 + 11 维热图
    fig3, axes3 = plt.subplots(1, 2, figsize=(13, 5.2))
    dnames = [ar['D_shared'][o] for o in OPS]
    x4 = np.arange(4)
    axes3[0].bar(x4, [d['abs_cos'] for d in dnames], 0.5, color='#1f6fb2')
    for xi, d in enumerate(dnames):
        axes3[0].plot([xi, xi], [d['boot_ci'][0], d['boot_ci'][1]], 'k-', lw=2)
        axes3[0].plot(xi, d['boot_ci'][0], 'k_', ms=8)
        axes3[0].plot(xi, d['boot_ci'][1], 'k_', ms=8)
    axes3[0].axhline(0.10, color='gray', ls='--', lw=1)
    axes3[0].axhline(0.25, color='r', ls='--', lw=1)
    axes3[0].set_xticks(x4); axes3[0].set_xticklabels([OP_CN[o] for o in OPS])
    axes3[0].set_title('|μ_op·D_shared|（随机基线 0.10——标记线 0.25）')
    axes3[0].grid(axis='y', alpha=0.3)
    mat = np.array(ar['target11']['matrix'])
    ax3 = axes3[1]
    im3 = ax3.imshow(mat, cmap='Reds', vmin=0, vmax=0.5)
    ax3.set_yticks(range(4)); ax3.set_yticklabels([OP_CN[o] for o in OPS])
    ax3.set_xticks(range(11)); ax3.set_xticklabels(TARGET11, rotation=45, fontsize=7)
    for gi, gname in [(0, 'HUMAN_ORG'), (1, 'AI_FEAT')]:
        dims = HUMAN_ORG if gname == 'HUMAN_ORG' else AI_FEAT
        for d in dims:
            c = TARGET11.index(d)
            ax3.add_patch(plt.Rectangle((c - 0.5, -0.5), 1, 4, fill=False,
                                        edgecolor='#e67e22' if gname == 'HUMAN_ORG' else '#1f6fb2', lw=1.2))
    for i in range(4):
        for j in range(11):
            if mat[i, j] > 0.25:
                ax3.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor='k', lw=1.6))
                ax3.text(j, i, f'{mat[i, j]:.2f}', ha='center', va='center', fontsize=6, color='white')
    ax3.set_title('4 op × 11 未解释维 |cos|（HUMAN_ORG 橙框/AI_FEAT 蓝框——>0.25 黑框标注）')
    fig3.colorbar(im3, ax=ax3, fraction=0.046)
    plt.tight_layout()
    plt.savefig(PAPER / 'fig_collapse_align.png', dpi=150)
    plt.close()

    # ===== 落盘 =====
    out = {
        'meta': {'seed': SEED, 'n_ladders_per_op': N_LADDER, 'ops': 4,
                 'n_texts_real': len(real_texts), 'n_texts_ph': len(ph_texts),
                 'shared_pool': True,
                 'design_note': 'C=端点差（望远镜恒等：中层仅逐级诊断贡献）——共享内容池——'
                                '同一真词臂 Ĉ 残差化两臂（Δg 同基准）'},
        'pool': {'pairs': [[a, b] for a, b in pool],
                 'objects': OBJ_POOL, 'var_groups': [list(v) for v in VAR_POOL]},
        'p0': {'words': list(word_ph.values()), 'L4_sil': round(st_l4['sil'], 4),
               'L4_acc': round(st_l4['acc_lda'], 4), 'L2_sil': round(st_l2['sil'], 4),
               'L2_acc': round(st_l2['acc_lda'], 4), 'switched': bool(switched),
               'ok': bool(ph_ok)},
        'validity': {'median_D_norm': round(m_c0, 4), 'per_op_mean_D': per_op_D,
                     'common_frac_kappa': round(kappa, 4)},
        'g_real': {**gr, 'label_perm_p': round(pl['p'], 4), 'boot_ci': bci['g_real_ci']},
        'g_ph': {**gp, 'boot_ci': bci['g_ph_ci']},
        'delta_g': {'value': round(float(delta_g), 4), 'boot_ci': bci['delta_g_ci'],
                    'paired': True},
        'm_c1': {'within_res_cos': round(gr['within'], 4), 'cross_shuffle_p': round(cs['p'], 4),
                 'same_op_shuffle_p_diag': round(so['p'], 4),
                 'cross_mean_null': round(cs['mean_null'], 4), 'pass': bool(c1)},
        'm_c2': {'g_ge_015': bool(c2_g), 'delta_g_ge_010': bool(c2_d),
                 'perm_p_lt_005': bool(c2_p), 'pass': bool(m_c2)},
        'm_c3': ar,
        'support_512': sup,
        'level_diag': ld,
        'ablation': ab,
        'verdict': {'m_c0_valid': bool(m_c0_ok), 'm_c1': bool(c1), 'm_c2': bool(m_c2),
                    'overall': overall, 'interpretation': interp},
    }
    (OUT / 'math_collapse.json').write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                            encoding='utf-8')
    print('\n落盘 math_collapse.json + fig_collapse_* × 3 ✓')


if __name__ == '__main__':
    main()
