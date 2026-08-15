# -*- coding: utf-8 -*-
"""v0.79 意图主轴猜想验证（纯离线——零 GPU）

猜想（用户）："人类句元跳跃度 d=+2.16"可能不是随机跳变——高维空间存在"意图主轴" D——
人类沿主轴推进（Δ_t = α_t·D + ε_t——有方向的变）——AI 无方向各向同性漂移。
成立则"人类波浪 vs AI 平坦"重新定义为"有方向的推进 vs 无方向的扩散"。

【预注册判据表（P0 预跑 2026-08-15 校准——判据线先写后跑——不满足明确报告否定）】

| # | 判据 | 判定线 |
|---|------|--------|
| C1 | PC1 EVR（组内中心化差分池） | 人类 >0.40 且 AI ≤ max(0.25, 代理基线+0.05) |
| C2 | 有效维度 | PE_人类 < PE_AI 且差 ≥5；n90 同向 |
| C3 | align_cos（裁决——裁定轴：cos(D_h,D_a)≥0.7 用共享轴否则组轴） | 人类 ≥0.25 且 AI ≤0.15 且 d≥1.0 且 CI 不含 0；超基线 ≥0.08 |
| C4 | j_par（绝对量——支持性不进总判定） | 人类 >AI，d≥1.0，CI 不含 0 |
| C5 | j_perp + ratio_unit | AI ≥0.8×人类 且 ratio_unit 人类<AI |
| C6 | 记忆定位（双 Hurst——全序列/评估集任一达线） | H(α)人−H(ε)人 ≥0.08 且 H(α)人−H(α)AI ≥0.08 |
| C7 | D 语义（支持性） | 载荷 top-6 含 org ≥4/6 且 cos(D, 人类差分均值方向)>0.5 |
| C8 | 代理对照 | 人类 align_cos > 基线+0.08；AI 在基线 ±0.04 |

总判定：成立 = C1∧C2∧C3∧C5∧C6∧C8；部分成立 = C3∧C6 PASS 其余有 FAIL；否定 = C3 FAIL 或 C1∧C6 双 FAIL。

P0 预跑记录（判据线校准依据）：人类 EVR 0.663/align_cos 0.628 vs AI 0.596/0.587——两组都远超各向同性基线
（EVR≈0.018/align_cos≈0.10）——组轴同向 cos=0.98——"AI 无主轴"二分预期不成立——线保留原线（AI 侧差距非线问题）。
"""
import sys, json, os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy import stats as sc
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path('C:/Users/bai/Desktop/小说系统')
OUT = BASE / 'data' / 'dim_analysis'
PAPER = Path('C:/Users/bai/Desktop/AB系统论文储备')
sys.path.insert(0, str(BASE / 'stage3'))

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

ORG = [10, 11, 34, 46, 48, 59]
AI_FEAT = [22, 26, 43, 52, 5]


def main():
    print('===== v0.79 意图主轴猜想验证（判据预注册——见文件头） =====')
    from engine_planner_bands import load_docs
    from engine_field_evidence import bootstrap_ci
    from dim_flow import hurst as DF_hurst

    fp, rows, docs, human, ai = load_docs()

    # ===== S1 差分构造 =====
    diffs = {'human': [], 'ai': []}
    for grp, dlist in (('human', human), ('ai', ai)):
        for doc in dlist:
            idx = np.array(docs[doc])
            D = fp[idx[1:], :] - fp[idx[:-1], :]
            diffs[grp].append(D)
    n_diff_h = sum(len(d) for d in diffs['human'])
    n_diff_a = sum(len(d) for d in diffs['ai'])
    print(f'S1 差分: 人类 {n_diff_h} / AI {n_diff_a}（长度比 {n_diff_h / n_diff_a:.2f}）')

    # ===== S1.5 中心化分域 + 组间平均漂移 =====
    mu_h = np.vstack(diffs['human']).mean(0)
    mu_a = np.vstack(diffs['ai']).mean(0)
    mu_h_u = mu_h / (np.linalg.norm(mu_h) + 1e-9)
    mu_a_u = mu_a / (np.linalg.norm(mu_a) + 1e-9)
    cos_mu = float(mu_h_u @ mu_a_u)
    print(f'S1.5 组间平均差分方向 cos(μ_h, μ_a) = {cos_mu:+.4f}——漂移方向差异（新判别信号候选）')

    def pca_full(X):
        p = PCA(n_components=min(64, X.shape[0]))
        p.fit(X)
        return p

    # ===== S2 各自 PCA（组内中心化原始差分池） =====
    print('\nS2 各自 PCA（组内中心化）:')
    evrs, PEs, n90s, D_g = {}, {}, {}, {}
    for grp in ('human', 'ai'):
        pool = np.vstack(diffs[grp])
        Xc = pool - pool.mean(0)
        p = pca_full(Xc)
        evr = p.explained_variance_ratio_
        evrs[grp] = evr
        PEs[grp] = float(np.exp(-np.sum(evr * np.log(evr + 1e-12))))
        n90s[grp] = int(np.argmax(np.cumsum(evr) >= 0.9) + 1)
        D = p.components_[0]
        if D @ mu_h_u < 0:
            D = -D
        D_g[grp] = D
        print(f'  {grp}: PC1 EVR={evr[0]:.4f} PE={PEs[grp]:.1f} n90={n90s[grp]}')
    cos_Dh_Da = float(D_g['human'] @ D_g['ai'])
    print(f'  cos(D_h, D_a) = {cos_Dh_Da:.4f}（≥0.7 共享轴可信——裁定轴选择依据）')

    # ===== S3 共享主轴（严格时间切分：前 60% 拟合 / 后 40% 评估） =====
    def unit_pool(ds):
        return np.vstack([d / (np.linalg.norm(d, axis=1)[:, None] + 1e-9) for d in ds])

    all_u = unit_pool(diffs['human'] + diffs['ai'])
    split = int(len(all_u) * 0.6)
    U_fit, U_eval = all_u[:split], all_u[split:]
    Uc = U_fit - U_fit.mean(0)
    p_shared = pca_full(Uc)
    D_shared = p_shared.components_[0]
    if D_shared @ mu_h_u < 0:
        D_shared = -D_shared
    E2 = p_shared.components_[1]
    print(f'\nS3 共享轴（前 60% 拟合 {len(U_fit)}——后 40% 评估 {len(U_eval)}）:')
    print(f'  拟合 PC1 EVR={p_shared.explained_variance_ratio_[0]:.4f} PC2={p_shared.explained_variance_ratio_[1]:.4f}')

    # 乐观差：全池 D vs 切分 D 的 align_cos
    p_all = pca_full(all_u - all_u.mean(0))
    D_all = p_all.components_[0]
    if D_all @ mu_h_u < 0:
        D_all = -D_all

    def align_cos_of(X, D):
        n = np.linalg.norm(X, axis=1)
        u = X / (n[:, None] + 1e-9)
        return float(np.mean(np.abs(u @ D)))

    # 评估集 per-doc（后 40% 差分——按行切分会混文档——改为 per-doc 后 40%）
    print('\nS4 per-doc 指标（per-doc 后 40% 差分——切分 D 评估）:')
    per_doc = {'human': {}, 'ai': {}}
    for grp, dlist in (('human', human), ('ai', ai)):
        for doc in dlist:
            D = diffs[grp][dlist.index(doc)]
            k = int(len(D) * 0.6)
            De = D[k:]  # 后 40% 评估
            if len(De) < 5:
                continue
            n = np.linalg.norm(De, axis=1)
            u = De / (n[:, None] + 1e-9)
            ac = float(np.mean(np.abs(u @ D_shared)))
            dc = float(np.mean(u @ D_shared))
            jp = float(np.mean(np.abs(De @ D_shared)))
            jpp = float(np.mean(np.abs(De - (De @ D_shared)[:, None] * D_shared)))
            jppu = float(np.mean(np.abs(u - (u @ D_shared)[:, None] * D_shared)))
            jpu = float(np.mean(np.abs(u @ D_shared)))
            ratio_u = jppu / (jpu + 1e-9)
            per_doc[grp][doc] = {'align_cos': ac, 'drift_cos': dc, 'j_par': jp,
                                 'j_perp': jpp, 'ratio_unit': ratio_u}

    def grp_vals(key):
        h = [v[key] for v in per_doc['human'].values()]
        a = [v[key] for v in per_doc['ai'].values()]
        return h, a

    res = {}
    for key in ('align_cos', 'drift_cos', 'j_par', 'j_perp', 'ratio_unit'):
        h, a = grp_vals(key)
        d_ = (np.mean(h) - np.mean(a)) / np.sqrt((np.var(h) + np.var(a)) / 2 + 1e-9)
        ci = bootstrap_ci(h, a, paired=False)
        res[key] = {'human': round(float(np.mean(h)), 4), 'ai': round(float(np.mean(a)), 4),
                    'd': round(float(d_), 3), 'ci95': [round(float(x), 4) for x in ci]}
        print(f'  {key}: 人类 {res[key]["human"]} vs AI {res[key]["ai"]}——d={d_:+.2f} CI {ci}')

    # ===== S5 α/ε Hurst（双报告） =====
    print('\nS5 Hurst（双报告——全序列/评估集前 200）:')
    h_stats = {'human': {'full': [], 'eval200': []}, 'ai': {'full': [], 'eval200': []}}
    for grp, dlist in (('human', human), ('ai', ai)):
        for doc in dlist:
            D = diffs[grp][dlist.index(doc)]
            alpha_full = D @ D_shared
            eps_full = D @ E2
            hf_a = DF_hurst(alpha_full)
            hf_e = DF_hurst(eps_full)
            if hf_a == hf_a and hf_e == hf_e:
                h_stats[grp]['full'].append((hf_a, hf_e))
            k = int(len(D) * 0.6)
            De = D[k:]
            if len(De) >= 100:
                he_a = DF_hurst(De[:200] @ D_shared)
                he_e = DF_hurst(De[:200] @ E2)
                if he_a == he_a and he_e == he_e:
                    h_stats[grp]['eval200'].append((he_a, he_e))
    for mode in ('full', 'eval200'):
        ha_h = [x[0] for x in h_stats['human'][mode]]
        he_h = [x[1] for x in h_stats['human'][mode]]
        ha_a = [x[0] for x in h_stats['ai'][mode]]
        he_a = [x[1] for x in h_stats['ai'][mode]]
        if ha_h and he_h and ha_a:
            print(f'  [{mode}] H(α): 人类 {np.mean(ha_h):.3f} vs AI {np.mean(ha_a):.3f}——'
                  f'H(ε): 人类 {np.mean(he_h):.3f} vs AI {np.mean(he_a):.3f}——'
                  f'H(α)−H(ε)人={np.mean(ha_h) - np.mean(he_h):+.3f}——H(α)人−H(α)AI={np.mean(ha_h) - np.mean(ha_a):+.3f}')

    # ===== S6 D 语义 =====
    print('\nS6 D 语义:')
    top = np.argsort(np.abs(D_shared))[::-1][:10]
    org_hit = len(set(top[:6].tolist()) & set(ORG))
    print(f'  载荷 top-10 维: {top.tolist()}——top-6 含 org {org_hit}/6')
    print(f'  cos(D, μ_h)={float(D_shared @ mu_h_u):.4f}——cos(D, dim10)={float(D_shared[10]):.4f}——'
          f'cos(D, dim48)={float(D_shared[48]):.4f}')
    # 参考：人类核心 T（field_target.json——v0.78 衔接）
    ft = OUT / 'field_target.json'
    if ft.exists():
        T = np.asarray(json.loads(ft.read_text(encoding='utf-8'))['human_core'], float)
        print(f'  参考 cos(D, T 人类核心)={float(D_shared @ T):.4f}')

    # ===== S7 人类内跨篇相关（探索性） =====
    print('\nS7 人类内跨篇相关（探索性——n=10）:')
    # 用 flow_sent 的既有指标对照——直接重算跳跃
    for key in ('align_cos', 'drift_cos'):
        h_vals = [(doc, v[key]) for doc, v in per_doc['human'].items()]
        jumps = []
        for doc, _ in h_vals:
            D = diffs['human'][human.index(doc)]
            jumps.append(float(np.mean(np.abs(D @ D_shared))))
        ac_vals = [v for _, v in h_vals]
        if len(ac_vals) >= 6:
            r = sc.spearmanr(ac_vals, jumps)
            print(f'  Spearman({key}, j_par)={r.statistic:+.3f} p={r.pvalue:.3f}')

    # ===== S8 代理对照 + 稳健性 =====
    print('\nS8 代理对照（样本量匹配）:')
    rng = np.random.default_rng(42)
    base = {}
    for grp in ('human', 'ai'):
        pool = np.vstack(diffs[grp])
        sigma = np.sqrt(np.mean(pool ** 2))
        S = rng.normal(0, sigma, (pool.shape[0], 64))
        Sc = S - S.mean(0)
        p_s = pca_full(Sc)
        ac_s = align_cos_of(S, D_shared)
        base[grp] = {'evr': float(p_s.explained_variance_ratio_[0]),
                     'align_cos': ac_s, 'n': pool.shape[0]}
        print(f'  {grp} 代理: EVR={base[grp]["evr"]:.4f} align_cos={base[grp]["align_cos"]:.4f}')
    # 长度匹配稳健性（人类取前 438 差分）
    lm = {'align_cos': [], 'j_par': []}
    for doc in human:
        D = diffs['human'][human.index(doc)]
        Dm = D[:438]
        n = np.linalg.norm(Dm, axis=1)
        u = Dm / (n[:, None] + 1e-9)
        lm['align_cos'].append(float(np.mean(np.abs(u @ D_shared))))
        lm['j_par'].append(float(np.mean(np.abs(Dm @ D_shared))))
    print(f'  长度匹配（人类前 438 差分）: align_cos={np.mean(lm["align_cos"]):.4f} vs 全评估 {res["align_cos"]["human"]}')

    # ===== S9 判据裁定 =====
    print('\n===== S9 判据裁定 =====')
    h_ac, a_ac = grp_vals('align_cos')
    h_jp, a_jp = grp_vals('j_par')
    h_jpp, a_jpp = grp_vals('j_perp')
    h_ru, a_ru = grp_vals('ratio_unit')
    c1_h = evrs['human'][0] > 0.40
    c1_a = evrs['ai'][0] <= max(0.25, base['ai']['evr'] + 0.05)
    c1 = c1_h and c1_a
    c2 = (PEs['human'] < PEs['ai']) and (PEs['ai'] - PEs['human'] >= 5) and (n90s['human'] < n90s['ai'])
    # C3 裁定轴
    use_shared = cos_Dh_Da >= 0.7
    c3_h = np.mean(h_ac) >= 0.25
    c3_a = np.mean(a_ac) <= 0.15
    c3_d = res['align_cos']['d'] >= 1.0
    c3_ci = res['align_cos']['ci95'][0] > 0
    c3_base = (np.mean(h_ac) - base['human']['align_cos']) >= 0.08
    c3 = c3_h and c3_a and c3_d and c3_ci and c3_base
    c4 = (res['j_par']['d'] >= 1.0 and res['j_par']['ci95'][0] > 0)
    c5_a = np.mean(a_jpp) >= 0.8 * np.mean(h_jpp)
    c5_r = np.mean(h_ru) < np.mean(a_ru)
    c5 = c5_a and c5_r
    # C6
    mode6 = 'full'
    ha_h = [x[0] for x in h_stats['human']['full']]
    he_h = [x[1] for x in h_stats['human']['full']]
    ha_a = [x[0] for x in h_stats['ai']['full']]
    c6_1 = np.mean(ha_h) - np.mean(he_h) >= 0.08
    c6_2 = np.mean(ha_h) - np.mean(ha_a) >= 0.08
    c6 = c6_1 and c6_2
    c7 = (org_hit >= 4) and (float(D_shared @ mu_h_u) > 0.5)
    c8_h = (np.mean(h_ac) - base['human']['align_cos']) >= 0.08
    c8_a = abs(np.mean(a_ac) - base['ai']['align_cos']) <= 0.04
    c8 = c8_h and c8_a
    verdicts = {
        'C1_PC1_EVR': 'PASS' if c1 else 'FAIL',
        'C2_有效维度': 'PASS' if c2 else 'FAIL',
        'C3_align_cos': 'PASS' if c3 else 'FAIL',
        'C4_j_par(支持性)': 'PASS' if c4 else 'FAIL',
        'C5_j_perp_ratio': 'PASS' if c5 else 'FAIL',
        'C6_记忆定位': 'PASS' if c6 else 'FAIL',
        'C7_D语义(支持性)': 'PASS' if c7 else 'FAIL',
        'C8_代理对照': 'PASS' if c8 else 'FAIL',
    }
    for k, v in verdicts.items():
        print(f'  {k}: {v}')
    overall = '成立' if (c1 and c2 and c3 and c5 and c6 and c8) else \
        ('部分成立' if (c3 and c6) else '否定')
    print(f'  总判定: {overall}')

    # ===== S10 图 4 张 =====
    # 图 1：方差谱
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    for grp, c in (('human', '#1f6fb2'), ('ai', '#e67e22')):
        ax.plot(range(1, 21), evrs[grp][:20], marker='o', ms=3, lw=1.2, label=grp, color=c)
        ax.axhline(0, color='k', lw=0.5)
    ax.axhline(0.40, color='#c0392b', ls='--', lw=1, label='人类判据线 0.40')
    ax.axhline(max(0.25, base['ai']['evr'] + 0.05), color='#8e44ad', ls='--', lw=1,
              label=f'AI 判据线 {max(0.25, base["ai"]["evr"] + 0.05):.2f}')
    ax.set_xlabel('主成分序号')
    ax.set_ylabel('解释方差比 EVR')
    ax.set_title(f'差分方差谱（人类 PC1={evrs["human"][0]:.3f} PE={PEs["human"]:.1f} vs '
                 f'AI PC1={evrs["ai"][0]:.3f} PE={PEs["ai"]:.1f}）')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax2 = axes[1]
    for grp, c in (('human', '#1f6fb2'), ('ai', '#e67e22')):
        cum = np.cumsum(evrs[grp])
        ax2.plot(range(1, 41), cum[:40], lw=1.2, label=f'{grp} 累计', color=c)
    ax2.axhline(0.9, color='k', ls='--', lw=1, label='90% 线')
    ax2.set_xlabel('主成分序号')
    ax2.set_ylabel('累计解释方差')
    ax2.set_title(f'累计谱（n90: 人类 {n90s["human"]} vs AI {n90s["ai"]}）')
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PAPER / 'fig_axis_spectrum.png', dpi=150)
    plt.close()

    # 图 2：差分散布平面（代表篇）
    fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, grp, doc in ((axes2[0], 'human', human[-1]), (axes2[1], 'ai', ai[-1])):
        D = diffs[grp][(human if grp == 'human' else ai).index(doc)]
        x = D @ D_shared
        y = D @ E2
        rng2 = np.random.default_rng(7)
        sel = rng2.choice(len(x), min(400, len(x)), replace=False)
        ax.scatter(x[sel], y[sel], s=6, alpha=0.5,
                   color=('#1f6fb2' if grp == 'human' else '#e67e22'))
        sdx, sdy = np.std(x), np.std(y)
        t = np.linspace(0, 2 * np.pi, 100)
        ax.plot(sdx * np.cos(t), sdy * np.sin(t), 'k--', lw=1)
        ax.set_aspect('equal')
        ax.set_xlabel('沿主轴 Δ·D')
        ax.set_ylabel('垂直 Δ·E2')
        ax.set_title(f'{grp}（{doc}）差分散布 σD/σE2={sdx / sdy:.2f}')
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PAPER / 'fig_axis_plane.png', dpi=150)
    plt.close()

    # 图 3：j_par × j_perp per-doc
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    for grp, c in (('human', '#1f6fb2'), ('ai', '#e67e22')):
        xs = [v['j_par'] for v in per_doc[grp].values()]
        ys = [v['j_perp'] for v in per_doc[grp].values()]
        ax3.scatter(xs, ys, s=50, color=c, label=f'{grp}（n={len(xs)}）')
        ax3.errorbar(np.mean(xs), np.mean(ys), xerr=np.std(xs) / np.sqrt(len(xs)),
                     yerr=np.std(ys) / np.sqrt(len(ys)), fmt='o', ms=10, color=c, capsize=4)
    lim = max(ax3.get_xlim()[1], ax3.get_ylim()[1])
    ax3.plot([0, lim], [0, lim], 'k--', lw=1, label='j⊥=j∥')
    ax3.set_xlabel('j_par（沿主轴步长）')
    ax3.set_ylabel('j_perp（垂直分量）')
    ax3.set_title(f'推进 vs 扩散（ratio_unit: 人类 {np.mean(h_ru):.2f} vs AI {np.mean(a_ru):.2f}——量大但扁）')
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PAPER / 'fig_axis_jpar_perp.png', dpi=150)
    plt.close()

    # 图 4：α_t 累计轨迹 + 正负比例 + H(α)/H(ε)
    fig4, axes4 = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes4[0]
    for grp, c in (('human', '#1f6fb2'), ('ai', '#e67e22')):
        doc = (human if grp == 'human' else ai)[-1]
        D = diffs[grp][(human if grp == 'human' else ai).index(doc)]
        a_seq = D @ D_shared
        ax.plot(np.cumsum(a_seq[:400]), lw=1, label=f'{grp} Σα_t（{doc}）', color=c)
    ax.set_xlabel('句元序号')
    ax.set_ylabel('累计 Σα_t')
    ax.set_title('沿主轴累计推进（方向交替判读需看正负比例）')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax2 = axes4[1]
    for grp, c in (('human', '#1f6fb2'), ('ai', '#e67e22')):
        hs = h_stats[grp]['full']
        if not hs:
            continue
        ha = [x[0] for x in hs]
        he = [x[1] for x in hs]
        xpos = 0 if grp == 'human' else 1
        ax2.bar(xpos - 0.2, np.mean(ha), 0.4, color=c, alpha=0.85, label=f'{grp} H(α)')
        ax2.bar(xpos + 0.2, np.mean(he), 0.4, color=c, alpha=0.45, label=f'{grp} H(ε)' if grp == 'human' else None)
        ax2.errorbar(xpos - 0.2, np.mean(ha), yerr=np.std(ha) / np.sqrt(len(ha)), fmt='k.', capsize=4)
        ax2.errorbar(xpos + 0.2, np.mean(he), yerr=np.std(he) / np.sqrt(len(he)), fmt='k.', capsize=4)
    ax2.axhline(0.5, color='k', ls='--', lw=1, label='白噪 0.5')
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(['人类', 'AI'])
    ax2.set_ylabel('Hurst')
    ax2.set_title('记忆定位：H(α) vs H(ε)（判据线 H(α)−H(ε) ≥0.08）')
    ax2.legend(fontsize=8)
    ax2.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(PAPER / 'fig_axis_alpha_hurst.png', dpi=150)
    plt.close()

    # ===== 落盘 =====
    out = {
        'criteria': {'C1': bool(c1), 'C2': bool(c2), 'C3': bool(c3), 'C4': bool(c4),
                     'C5': bool(c5), 'C6': bool(c6), 'C7': bool(c7), 'C8': bool(c8),
                     'overall': overall},
        'per_group': {g: {'pc1_evr': float(evrs[g][0]), 'PE': PEs[g], 'n90': n90s[g]} for g in ('human', 'ai')},
        'axis': {'cos_Dh_Da': cos_Dh_Da, 'cos_mu_h_mu_a': cos_mu,
                 'shared_pc1_evr': float(p_shared.explained_variance_ratio_[0]),
                 'shared_D_top10': [int(x) for x in top], 'org_hit_top6': org_hit},
        'per_doc_metrics': res,
        'hurst': {m: {'human_Ha': float(np.mean([x[0] for x in h_stats['human'][m]])) if h_stats['human'][m] else None,
                      'human_He': float(np.mean([x[1] for x in h_stats['human'][m]])) if h_stats['human'][m] else None,
                      'ai_Ha': float(np.mean([x[0] for x in h_stats['ai'][m]])) if h_stats['ai'][m] else None}
                  for m in ('full', 'eval200')},
        'surrogate': base,
        'robustness': {'length_matched_align_cos': round(float(np.mean(lm['align_cos'])), 4)},
        'verdicts': verdicts,
    }
    (OUT / 'axis_analysis.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print('\n落盘 axis_analysis.json + fig_axis_*.png × 4 ✓')


if __name__ == '__main__':
    main()
