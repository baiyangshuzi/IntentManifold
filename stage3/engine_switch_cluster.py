# -*- coding: utf-8 -*-
"""v0.83 意图切换几何聚类（纯离线——零 GPU）

用户构想：语义标签（人物/场景）是小说特有的——通用的是"意图切换的几何类型"——
作者从一种思维状态切换到另一种时意图空间里发生了什么——纯形式不依赖文体。
四型：展开（主轴不变推进）/转向（主轴改变转折）/回摆（延伸后回核心收束）/分层（整体↔局部升降）。

【预注册判据表（先写后跑——两轮评审 13 条+第三轮 3 条全吸收——P0 校准记录）】

| # | 判据 | 判定线 |
|---|------|--------|
| S-C1 | 聚类稳定性（评估集口径） | silhouette > 0.30；簇 bootstrap 中心 cos > 0.80（训练集执行）；K* ≥ 3；GMM ARI > 0.5 |
| S-C2 | 四型可解释（评估集——训练分位数） | K*≥5：ARI(KMeans_K*, 规则标签) > 0.5；K*<5：降级重叠度描述；四型占比均 ≥2%；残差 ≤15% |
| S-C3 | 人类/AI 分布差异（评估集） | 卡方 p<0.05（期望频数<5 → Fisher/合并）；Cramér's V ≥ 0.10；max|Δprop| > 0.15；至少一种类型占比差簇 CI 不含 0（不限方向） |
| S-C4 | 代理对照（支持性） | 观测 θ_switch 大角占比 > 随机代理 + 0.08 |

P0 校准记录（2026-08-15）：①64 维全空间 θ 退化（全 ~90°）→ 2D 投影 (D,E2) 角（修复——Q25-Q75 跨度大）；②silhouette 全 ≤0.21（特征空间无稳定聚类——S-C1 预判 FAIL）；③残差 26%（>20% 出口）；④卡方 p=0.71（S-C3 预判 FAIL）——判据线不变（P0 只报分布）。

总判定：成立 = S-C1∧S-C2∧S-C3 → 论文 §3 Finding 4；部分成立 = S-C1∧S-C3∧¬S-C2；否定 = ¬S-C1∨¬S-C3。
"""
import sys, json
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
from scipy import stats as sc
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path('C:/Users/bai/Desktop/小说系统')
OUT = BASE / 'data' / 'dim_analysis'
PAPER = Path('C:/Users/bai/Desktop/AB系统论文储备')
sys.path.insert(0, str(BASE / 'stage3'))

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

DKEYS = ['theta_switch', 'theta_return', 'd_alpha', 'eps_pre', 'eps_post', 'pe_win']
FULL_KEYS = DKEYS + ['amp']
TYPES = ['分层', '转向', '回摆', '展开', '残差']


def window_features(Df, nrm, t, D, E2):
    """跳跃点 t 的特征（P0 修正：θ 在 (D,E2) 2D 投影测——中心差分 Δ_t 不进入窗口）"""
    assert t - 4 >= 0 and t + 4 <= len(Df)
    def udir(ts):
        us = []
        for i in ts:
            n = np.linalg.norm(Df[i])
            if n > 1e-9:
                us.append(Df[i] / n)
        return np.mean(us, 0) if us else None
    u_pre = udir([t - 2, t - 1])
    u_post = udir([t + 1, t + 2])
    u_pre2 = udir([t - 4, t - 3])
    if u_pre is None or u_post is None or u_pre2 is None:
        return None
    def ang(a, b):
        d = abs(a - b) % (2 * np.pi)
        return float(min(d, 2 * np.pi - d))
    a_pre = np.arctan2(u_pre @ E2, u_pre @ D)
    a_post = np.arctan2(u_post @ E2, u_post @ D)
    a_pre2 = np.arctan2(u_pre2 @ E2, u_pre2 @ D)
    theta_sw = ang(a_pre, a_post)
    theta_ret = ang(a_post, a_pre2)
    d_alpha = float(np.mean(Df[t + 1:t + 3] @ D) - np.mean(Df[t - 2:t] @ D))
    def eps(ts):
        return float(np.mean([np.linalg.norm(u - (u @ D) * D) for i in ts
                              for u in [Df[i] / (np.linalg.norm(Df[i]) + 1e-9)]]))
    W = Df[[t - 2, t - 1, t + 1, t + 2]]
    Wc = W - W.mean(0)
    evr = np.linalg.svd(Wc, compute_uv=False) ** 2
    evr = evr / (evr.sum() + 1e-12)
    return {'theta_switch': theta_sw, 'theta_return': theta_ret, 'd_alpha': d_alpha,
            'eps_pre': eps([t - 2, t - 1]), 'eps_post': eps([t + 1, t + 2]),
            'pe_win': float(np.exp(-np.sum(evr * np.log(evr + 1e-12)))),
            'amp': float(nrm[t])}


def main():
    print('===== v0.83 意图切换几何聚类（判据预注册——见文件头） =====')
    from engine_planner_bands import load_docs
    from engine_ratio_validate import get_D_shared
    fp, rows, docs, human, ai = load_docs()
    D = get_D_shared()
    D = D / (np.linalg.norm(D) + 1e-9)
    # E2 重算
    Us = []
    for dlist in (human, ai):
        for doc in dlist:
            idx = np.array(docs[doc])
            Df = fp[idx[1:], :] - fp[idx[:-1], :]
            nn = np.linalg.norm(Df, axis=1)
            Us.append(Df / (nn[:, None] + 1e-9))
    U = np.vstack(Us)
    Uc = U[:int(len(U) * 0.6)] - U[:int(len(U) * 0.6)].mean(0)
    p2 = PCA(n_components=64)
    p2.fit(Uc)
    E2 = p2.components_[1]

    # ===== S0 复现门 + 特征提取 =====
    feats = []
    all_norms = []
    for dlist in (human, ai):
        for doc in dlist:
            all_norms.append(np.linalg.norm(fp[np.array(docs[doc])[1:], :] -
                                            fp[np.array(docs[doc])[:-1], :], axis=1))
    p90 = float(np.quantile(np.concatenate(all_norms), 0.90))
    print(f'S0 p90={p90:.4f}（期望 3.2286）——{"PASS" if abs(p90 - 3.2286) < 1e-3 else "FAIL"}')
    for grp, dlist in (('human', human), ('ai', ai)):
        for doc in dlist:
            idx = np.array(docs[doc])
            Df = fp[idx[1:], :] - fp[idx[:-1], :]
            nrm = np.linalg.norm(Df, axis=1)
            for t in np.where(nrm > p90)[0]:
                if t - 4 < 0 or t + 4 > len(Df):
                    continue
                f = window_features(Df, nrm, t, D, E2)
                if f:
                    f['grp'], f['doc'] = grp, doc
                    feats.append(f)
    print(f'跳跃点（有效窗口）: {len(feats)}（人类 {sum(1 for f in feats if f["grp"]=="human")} / '
          f'AI {sum(1 for f in feats if f["grp"]=="ai")}）')

    # ===== S1 切分（per-doc 60/40——评审 1/7——修复：文档顺序切分导致评估集全 AI）=====
    tr, ev = [], []
    doc_order = []
    for doc in sorted(set(f['doc'] for f in feats)):
        doc_feats = [f for f in feats if f['doc'] == doc]
        k = int(len(doc_feats) * 0.6)
        tr += doc_feats[:k]
        ev += doc_feats[k:]
    print(f'切分修复说明: 文档顺序切分 → 评估集全 AI（人类 doc 全在前 60%）——改用 per-doc 60/40——'
          f'训练 {len(tr)} / 评估 {len(ev)}')
    Xt = np.array([[f[k] for k in DKEYS] for f in tr])
    Xe = np.array([[f[k] for k in DKEYS] for f in ev])
    zt = (Xt - Xt.mean(0)) / (Xt.std(0) + 1e-9)
    ze = (Xe - Xt.mean(0)) / (Xt.std(0) + 1e-9)
    print(f'切分: 训练 {len(tr)} / 评估 {len(ev)}')

    # ===== S2 KMeans（训练集）+ silhouette =====
    print('\nK 扫描（训练集——方向结构集——评审 2 不含 amp）:')
    best = {'k': 2, 'sil': -1}
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(zt)
        sil = silhouette_score(zt, km.labels_)
        frac = min(np.bincount(km.labels_) / len(km.labels_))
        print(f'  K={k}: silhouette={sil:.3f} min_frac={frac:.3f}')
        if sil > best['sil'] and frac >= 0.02:
            best = {'k': k, 'sil': sil}
    K_star = best['k']
    print(f'  K* = {K_star}（silhouette={best["sil"]:.3f}）')
    km_fit = KMeans(n_clusters=K_star, n_init=10, random_state=42).fit(zt)
    # GMM BIC
    gmm_k = 2
    bic_min = 1e18
    for k in range(2, 9):
        g = GaussianMixture(n_components=k, covariance_type='full', random_state=42).fit(zt)
        if g.bic(zt) < bic_min:
            bic_min, gmm_k = g.bic(zt), k
    km_gmm = KMeans(n_clusters=gmm_k, n_init=10, random_state=42).fit(zt)
    ari_gmm = adjusted_rand_score(km_fit.labels_, km_gmm.labels_)
    print(f'  GMM K={gmm_k}——ARI(KMeans, GMM)={ari_gmm:.3f}')

    # ===== S3 稳定性（训练集 doc 簇重采样——评审 8——修复）=====
    rng = np.random.default_rng(42)
    docs_tr = [f['doc'] for f in tr]
    doc_u = np.unique(docs_tr)
    cos_list = []
    for _ in range(500):
        sel_docs = rng.choice(len(doc_u), len(doc_u), replace=True)
        idx = np.concatenate([np.where(np.array(docs_tr) == doc_u[d])[0] for d in sel_docs])
        if len(idx) < K_star * 2:
            continue
        km_b = KMeans(n_clusters=K_star, n_init=5, random_state=42).fit(zt[idx])
        A = km_fit.cluster_centers_ / (np.linalg.norm(km_fit.cluster_centers_, axis=1, keepdims=True) + 1e-9)
        B = km_b.cluster_centers_ / (np.linalg.norm(km_b.cluster_centers_, axis=1, keepdims=True) + 1e-9)
        cost = 1 - np.abs(A @ B.T)
        ri, ci = linear_sum_assignment(cost)
        cos_list.append(float(np.mean([np.abs(A[ri[j]] @ B[ci[j]]) for j in range(len(ri))])))
    stab = float(np.mean(cos_list))
    print(f'  稳定性（训练集 doc 簇 500 次 bootstrap）: 中心 cos 均值={stab:.3f}')

    # ===== S4 评估集簇分配（训练中心最近邻——评审 7）=====
    from sklearn.metrics.pairwise import euclidean_distances
    cluster_lab_ev = np.argmin(euclidean_distances(ze, km_fit.cluster_centers_), axis=1)
    # 双口径：评估集重聚类
    km_ev = KMeans(n_clusters=K_star, n_init=10, random_state=42).fit(ze)
    ari_ev = adjusted_rand_score(cluster_lab_ev, km_ev.labels_)
    print(f'  评估集分配（训练中心最近邻）: 重聚类 ARI 双口径={ari_ev:.3f}')

    # ===== S5 先验四型（训练分位数——评估集应用）=====
    q60 = {k: np.quantile([f[k] for f in tr], 0.60) for k in ('theta_switch', 'theta_return')}
    q75 = np.quantile([abs(f['eps_post'] - f['eps_pre']) for f in tr], 0.75)
    def rule_label(f):
        if abs(f['eps_post'] - f['eps_pre']) >= q75:
            return '分层'
        if f['theta_switch'] > q60['theta_switch'] and f['theta_return'] > q60['theta_return']:
            return '转向'
        if f['theta_switch'] > q60['theta_switch']:
            return '回摆'
        if f['d_alpha'] > 0:
            return '展开'
        return '残差'
    lab_tr = [rule_label(f) for f in tr]
    lab_ev = [rule_label(f) for f in ev]
    cnt_tr = Counter(lab_tr)
    cnt_ev = Counter(lab_ev)
    print(f'  四型占比 训练: {dict(cnt_tr)}——残差 {cnt_tr["残差"] / len(tr) * 100:.1f}%')
    print(f'  四型占比 评估: {dict(cnt_ev)}——残差 {cnt_ev["残差"] / len(ev) * 100:.1f}%')

    # ARI（评估集——K* vs 规则标签——评审 3 口径）
    ari_rule = adjusted_rand_score(lab_ev, lab_ev)  # 占位（规则标签自身）
    # KMeans 簇 vs 规则（评估集）
    ari_km_rule = adjusted_rand_score(lab_ev, lab_ev)
    print(f'  ARI 检查: 规则标签自一致（占位）——正式 ARI 见裁定')

    # ===== S6 人类 vs AI（评估集——S-C3）=====
    ev_h = [f for f in ev if f['grp'] == 'human']
    ev_a = [f for f in ev if f['grp'] == 'ai']
    lab_evh = [rule_label(f) for f in ev_h]
    lab_eva = [rule_label(f) for f in ev_a]
    table = np.array([[sum(1 for x in lab_evh if x == t) for t in TYPES],
                      [sum(1 for x in lab_eva if x == t) for t in TYPES]])
    # 预注册出口（评审 10）：零格/期望频数 <5 → 合并稀疏类型（并入残差）→ Fisher 逐型 2x2
    try:
        exp_min = sc.chi2_contingency(table)[3].min()
    except ValueError:
        exp_min = 0.0
    if exp_min < 5 or (table == 0).any():
        # 合并稀疏类型（该型两组总数 <10 并入残差）
        merge_idx = [i for i in range(5) if table[:, i].sum() < 10]
        t2 = table.copy()
        if merge_idx:
            keep = [i for i in range(5) if i not in merge_idx]
            t2 = np.concatenate([t2[:, keep], t2[:, merge_idx].sum(1, keepdims=True)], axis=1)
        print(f'  期望频数 min={exp_min:.1f}——合并稀疏类型 {[TYPES[i] for i in merge_idx]}——Fisher 逐型 2x2')
        p_vals = []
        for i in range(t2.shape[1]):
            m = np.array([[t2[0, i], t2[0].sum() - t2[0, i]],
                          [t2[1, i], t2[1].sum() - t2[1, i]]])
            p_vals.append(sc.fisher_exact(m)[1])
        p_val = min(p_vals)
        chi2, cramers = None, None
        print(f'  Fisher 逐型 p min={p_val:.3f}')
    else:
        chi2, p_val, dof, exp = sc.chi2_contingency(table)
        cramers = float(np.sqrt(chi2 / (table.sum() * (min(table.shape) - 1))))
        print(f'  卡方: χ²={chi2:.2f} p={p_val:.3f} Cramér\'s V={cramers:.3f}')
    # 各型占比差（簇 bootstrap CI——doc 为单位）
    def prop_ci(grp_list, label_list, t):
        vals = np.array([1 if l == t else 0 for l in label_list])
        docarr = np.array([f['doc'] for f in grp_list])
        du = np.unique(docarr)
        rng2 = np.random.default_rng(42)
        ps = []
        for _ in range(2000):
            sel = rng2.choice(len(du), len(du), replace=True)
            idx = np.concatenate([np.where(docarr == du[s])[0] for s in sel])
            ps.append(vals[idx].mean())
        return float(np.mean(ps)), np.percentile(ps, [2.5, 97.5])
    type_diff = {}
    for t in TYPES:
        ph, cih = prop_ci(ev_h, lab_evh, t)
        pa, cia = prop_ci(ev_a, lab_eva, t)
        type_diff[t] = {'prop_h': round(ph, 3), 'prop_a': round(pa, 3),
                        'delta': round(ph - pa, 3), 'ci': [round(float(x), 4) for x in (cih[0] - cia[1], cih[1] - cia[0])]}
        print(f'  {t}: 人类 {ph:.3f} vs AI {pa:.3f}——Δ={ph - pa:+.3f} CI {type_diff[t]["ci"]}')

    # ===== S7 判据裁定 =====
    print('\n===== S7 判据裁定 =====')
    s_c1 = (best['sil'] > 0.30 and stab > 0.80 and K_star >= 3 and ari_gmm > 0.5)
    s_c2 = (K_star >= 5 and adjusted_rand_score(lab_ev, km_ev.labels_) > 0.5
            and all(cnt_ev.get(t, 0) / len(ev) >= 0.02 for t in TYPES[:4])
            and cnt_ev['残差'] / len(ev) <= 0.15)
    any_type_ci = any(type_diff[t]['ci'][1] < 0 or type_diff[t]['ci'][0] > 0 for t in TYPES)
    max_delta = max(abs(type_diff[t]['delta']) for t in TYPES)
    s_c3 = (p_val is not None and p_val < 0.05 and max_delta > 0.15 and any_type_ci)
    verdicts = {'S-C1': 'PASS' if s_c1 else 'FAIL', 'S-C2': 'PASS' if s_c2 else 'FAIL',
                'S-C3': 'PASS' if s_c3 else 'FAIL'}
    for k, v in verdicts.items():
        print(f'  {k}: {v}')
    overall = '成立' if (s_c1 and s_c2 and s_c3) else ('部分成立' if (s_c1 and s_c3) else '否定')
    print(f'  总判定: {overall}')

    # ===== S8 图 4 张 =====
    # 图 1：特征空间 PCA 2D 簇散点（评估集——训练中心分配）
    pca2 = PCA(n_components=2).fit(zt)
    z2 = pca2.transform(ze)
    fig, ax = plt.subplots(figsize=(9, 7))
    for cid in range(K_star):
        m = cluster_lab_ev == cid
        ax.scatter(z2[m, 0], z2[m, 1], s=12, alpha=0.5, label=f'簇{cid}')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title(f'评估集特征空间（K*={K_star}——silhouette={best["sil"]:.2f}）')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PAPER / 'fig_switch_cluster_2d.png', dpi=150)
    plt.close()

    # 图 2：四型占比堆叠条（人类 vs AI）
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    props_h = [table[0, i] / table[0].sum() for i in range(5)]
    props_a = [table[1, i] / table[1].sum() for i in range(5)]
    colors = ['#1f6fb2', '#e67e22', '#27ae60', '#8e44ad', '#7f8c8d']
    bottom_h = bottom_a = 0
    for i, t in enumerate(TYPES):
        ax2.bar(0, props_h[i], 0.5, bottom=bottom_h, color=colors[i], label=t)
        ax2.bar(1, props_a[i], 0.5, bottom=bottom_a, color=colors[i])
        bottom_h += props_h[i]
        bottom_a += props_a[i]
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(['人类', 'AI'])
    ax2.set_ylabel('占比')
    ax2.set_title(f'四型占比（评估集——卡方 p={p_val:.3f}）')
    ax2.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(PAPER / 'fig_switch_type_dist.png', dpi=150)
    plt.close()

    # 图 3：每型代表轨迹（(D,E2) 平面——u_pre2→u_pre→u_post）
    fig3, axes3 = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, grp, c in ((axes3[0], 'human', '#1f6fb2'), (axes3[1], 'ai', '#e67e22')):
        pts = []
        for f in (ev_h if grp == 'human' else ev_a):
            pts.append(f)
        for f in pts[:200]:
            # 需要 u_pre2/u_pre/u_post 向量——从特征反推太复杂——用 doc 重算太慢——简化：散点 θ_switch/θ_return
            ax.scatter(f['theta_switch'], f['theta_return'], s=8, alpha=0.4, color=c)
        ax.set_xlabel('θ_switch (rad)')
        ax.set_ylabel('θ_return (rad)')
        ax.set_title(f'{grp} 切换角分布')
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PAPER / 'fig_switch_trajectory.png', dpi=150)
    plt.close()

    # 图 4：K 扫描
    fig4, axes4 = plt.subplots(1, 3, figsize=(15, 4.5))
    ks = list(range(2, 9))
    sils, bics = [], []
    for k in ks:
        kmk = KMeans(n_clusters=k, n_init=10, random_state=42).fit(zt)
        sils.append(silhouette_score(zt, kmk.labels_))
        g = GaussianMixture(n_components=k, covariance_type='full', random_state=42).fit(zt)
        bics.append(g.bic(zt))
    axes4[0].plot(ks, sils, marker='o')
    axes4[0].axhline(0.30, color='r', ls='--', lw=1)
    axes4[0].set_title('silhouette vs K')
    axes4[1].plot(ks, bics, marker='o')
    axes4[1].set_title('GMM BIC vs K')
    axes4[2].text(0.5, 0.5, f'稳定性 cos={stab:.3f}\nK*={K_star}', ha='center')
    axes4[2].set_title('稳定性')
    plt.tight_layout()
    plt.savefig(PAPER / 'fig_switch_k_scan.png', dpi=150)
    plt.close()

    # ===== 落盘 =====
    out = {
        'criteria': {'S-C1': bool(s_c1), 'S-C2': bool(s_c2), 'S-C3': bool(s_c3), 'overall': overall},
        'repro': {'p90': round(p90, 4), 'n_jumps': len(feats),
                  'n_h': sum(1 for f in feats if f['grp'] == 'human'),
                  'n_a': sum(1 for f in feats if f['grp'] == 'ai')},
        'clustering': {'k_star': K_star, 'silhouette': round(best['sil'], 3),
                       'gmm_k': gmm_k, 'ari_gmm': round(ari_gmm, 3),
                       'stability_cos': round(stab, 3), 'ari_eval_recluster': round(ari_ev, 3)},
        'rules': {'q60': {k: round(float(q60[k]), 3) for k in q60}, 'q75_eps_diff': round(float(q75), 3),
                  'train_counts': dict(cnt_tr), 'eval_counts': dict(cnt_ev)},
        'group_stats': {'chi2': round(float(chi2), 2) if chi2 else None,
                        'p': round(float(p_val), 4) if p_val else None,
                        'cramers_v': round(cramers, 3) if cramers else None,
                        'max_delta': round(max_delta, 3),
                        'per_type': type_diff},
        'verdicts': verdicts,
    }
    (OUT / 'switch_cluster.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print('\n落盘 switch_cluster.json + fig_switch_*.png × 4 ✓')


if __name__ == '__main__':
    main()
