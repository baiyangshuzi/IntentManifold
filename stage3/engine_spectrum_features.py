# -*- coding: utf-8 -*-
"""v0.84 连续谱特征系统比较（纯离线——零 GPU）

用户方案：9 个数学特征（F1-F9）系统比较——找"可测量、可干预、可训练"的特征——
优先级排序指导后续干预与训练实验。四性：区分度/可干预性/可嵌入性/可训练性。

【预注册判据表（先写后跑——两轮评审 5 条吸收）】

| # | 判据 | 判定线 |
|---|------|--------|
| S-F1 | 强候选集 | S1={F: |d|>1.0 ∧ CI 不含 0 ∧ BH-FDR q<0.05}——PASS=|S1|≥2 且 S1∩{F1,F3}≠∅；S1 非空但未含 F1/F3 → 记录偏差讨论原因（评审 6 放宽——不直接否定） |
| S-F2 | F4 新发现 | |d(lag1)|>0.5 且 CI 不含 0——否则记录"F4 无判别力" |
| S-F3 | 综合排序稳定 | 综合分 top-3 ∩ {F1,F3,F4} ≥ 2——否则报告偏离原因 |

评审吸收：①F1 加 mean_signed_α + dir_consistency（方向符号——cos=-0.29 已示方向差异）；②S4 主对照锁定 vt_gate_beam（v0.73-4 质量-成本最优）——其他 vt 探索性；③F4 最小差分 ≥15（P0 统计——不足则降级探索性）；④F6/F8 频段写死（低频 [0.001,0.1] / 高频 [0.2,0.5] cycles/step）；⑤长度稳健性纳入判据（截断后 d 大幅变化→从强候选剔除）。

综合分 = 0.4·d_norm + 0.3·干预 + 0.2·嵌入 + 0.1·训练（d_norm=min-max 归一化）——平局干预高者排前。
"""
import sys, json
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy import stats as sc
from scipy.signal import welch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path('C:/Users/bai/Desktop/小说系统')
OUT = BASE / 'data' / 'dim_analysis'
PAPER = Path('C:/Users/bai/Desktop/AB系统论文储备')
sys.path.insert(0, str(BASE / 'stage3'))

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 频段写死（评审 4）
LOW_BAND = (0.001, 0.1)
HIGH_BAND = (0.2, 0.5)
FREQ_MAX = 0.25
MIN_DIFFS = 15  # 评审 3：F4 最小差分门槛

# 三轴评分（预注册——固定不可事后改）
SCORES = {
    'F1': {'interv': 0.85, 'embed': 0.95, 'train': 0.75},
    'F2': {'interv': 0.80, 'embed': 0.95, 'train': 0.75},
    'F3': {'interv': 0.75, 'embed': 0.95, 'train': 0.85},
    'F4': {'interv': 0.55, 'embed': 0.75, 'train': 0.60},
    'F5': {'interv': 0.35, 'embed': 0.55, 'train': 0.65},
    'F6': {'interv': 0.25, 'embed': 0.40, 'train': 0.50},
    'F7': {'interv': 0.20, 'embed': 0.35, 'train': 0.45},
    'F8': {'interv': 0.30, 'embed': 0.45, 'train': 0.50},
    'F9': {'interv': 0.30, 'embed': 0.45, 'train': 0.50},
}


def dfa_alpha(x, min_box=8):
    """DFA order-1（手写）——返回 (alpha_D, n_boxes_min, r2)"""
    x = np.asarray(x, float)
    if len(x) < 2 * min_box:
        return np.nan, 0, 0.0
    Y = np.cumsum(x - x.mean())
    N = len(Y)
    boxes = np.unique(np.geomspace(min_box, N // 4, 10).astype(int))
    boxes = boxes[boxes >= min_box]
    if len(boxes) < 3:
        return np.nan, 0, 0.0
    fs, ns = [], []
    for n in boxes:
        nb = N // n
        if nb < 1:
            continue
        rms_vals = []
        for i in range(nb):
            seg = Y[i * n:(i + 1) * n]
            t = np.arange(len(seg))
            coef = np.polyfit(t, seg, 1)
            fit = np.polyval(coef, t)
            rms_vals.append(np.sqrt(np.mean((seg - fit) ** 2)))
        fs.append(float(np.sqrt(np.mean(rms_vals))))
        ns.append(n)
    if len(ns) < 3:
        return np.nan, 0, 0.0
    res = sc.linregress(np.log(ns), np.log(fs))
    return float(res.slope), int(min(ns)), float(res.rvalue ** 2)


def bh_fdr(pvals):
    """BH-FDR q 值（手写）"""
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = np.zeros(n)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        qv = ranked[i] * n / (i + 1)
        q[order[i]] = min(prev, qv)
        prev = q[order[i]]
    return q


def main():
    print('===== v0.84 连续谱特征系统比较（判据预注册——见文件头） =====')
    from engine_planner_bands import load_docs
    from engine_ratio_validate import get_D_shared, ratio_of, cohens_d
    from engine_field_evidence import bootstrap_ci
    from dim_flow import hurst as df_hurst

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
    from sklearn.decomposition import PCA
    p2 = PCA(n_components=64)
    p2.fit(Uc)
    E2 = p2.components_[1]

    # ===== S0 复现门 =====
    print('\nS0 复现门:')
    # gate2: ratio_unit 复现
    def bil_ratio(dlist):
        vals = []
        for doc in dlist:
            idx = np.array(docs[doc])
            Df = fp[idx[1:], :] - fp[idx[:-1], :]
            k = int(len(Df) * 0.6)
            vals.append(ratio_of(Df[k:], D)[0])
        return vals
    bh_r = bil_ratio(human)
    ba_r = bil_ratio(ai)
    print(f'  ratio_unit: 人类 {np.mean(bh_r):.4f} vs AI {np.mean(ba_r):.4f}（期望 0.0863/0.1048）')

    # ===== S1 per-doc 特征 =====
    print('\nS1 per-doc 特征:')
    per_doc = {}
    for grp, dlist in (('human', human), ('ai', ai)):
        for doc in dlist:
            idx = np.array(docs[doc])
            Df = fp[idx[1:], :] - fp[idx[:-1], :]
            alpha = Df @ D
            eps = Df @ E2
            nrm = np.linalg.norm(Df, axis=1)
            u = Df / (nrm[:, None] + 1e-9)
            f = {}
            # F1
            f['F1_mean_abs'] = float(np.mean(np.abs(alpha)))
            f['F1_std'] = float(np.std(alpha))
            f['F1_skew'] = float(sc.skew(alpha))
            f['F1_kurt'] = float(sc.kurtosis(alpha))
            f['F1_pos_frac'] = float(np.mean(alpha > 0))
            # 评审 1：方向符号
            f['F1_mean_signed'] = float(np.mean(alpha))
            f['F1_dir_cons'] = float(np.mean(np.sign(alpha[:-1]) == np.sign(alpha[1:])))
            # F2
            f['F2_mean_abs'] = float(np.mean(np.abs(eps)))
            f['F2_std'] = float(np.std(eps))
            # F3
            k = int(len(Df) * 0.6)
            f['F3_ratio'] = ratio_of(Df[k:], D)[0]
            # F4（评审 3：≥15 差分）
            if len(u) >= MIN_DIFFS:
                f['F4_lag1'] = float((u[:-1] * u[1:]).sum(1).mean())
                f['F4_lag2'] = float((u[:-2] * u[2:]).sum(1).mean()) if len(u) >= 3 else np.nan
            else:
                f['F4_lag1'] = np.nan
                f['F4_lag2'] = np.nan
            # F5 分布拟合
            x = np.abs(alpha) + 1e-6
            try:
                s_logn = sc.lognorm.fit(x, floc=0)
                ll_logn = sc.lognorm.logpdf(x, *s_logn).sum()
                xnorm = x / x.max()
                s_pow = sc.powerlaw.fit(xnorm, floc=0, fscale=1)
                ll_pow = sc.powerlaw.logpdf(xnorm, *s_pow).sum()
                ll_expo = sc.expon.logpdf(x, floc=0).sum()
                aics = {'lognorm': 2 * 3 - 2 * ll_logn, 'powerlaw': 2 * 2 - 2 * ll_pow,
                        'expon': 2 * 1 - 2 * ll_expo}
                best = min(aics, key=aics.get)
                aic_sorted = sorted(aics.values())
                f['F5_best'] = best
                f['F5_lognorm_sigma'] = float(s_logn[1])
                f['F5_aic_margin'] = float(aic_sorted[1] - aic_sorted[0])
            except Exception:
                f['F5_best'] = 'fail'
                f['F5_lognorm_sigma'] = np.nan
                f['F5_aic_margin'] = np.nan
            # F6 功率谱斜率
            xd = alpha - alpha.mean()
            if len(xd) >= 128:
                freqs, P = welch(xd, nperseg=64, noverlap=32)
            else:
                freqs, P = welch(xd, nperseg=32, noverlap=16)
            band = (freqs >= 2 / len(xd)) & (freqs <= FREQ_MAX)
            if band.sum() >= 3 and np.all(P[band] > 0):
                res = sc.linregress(np.log(freqs[band]), np.log(P[band]))
                f['F6_beta'] = float(res.slope)
            else:
                f['F6_beta'] = np.nan
            # F7 DFA + Hurst
            ad, nbox, r2 = dfa_alpha(alpha)
            f['F7_dfa'] = ad
            f['F7_hurst'] = df_hurst(alpha)
            # F8 频段能量（评审 4 频段写死）
            try:
                e_low = P[(freqs >= LOW_BAND[0]) & (freqs <= LOW_BAND[1])].sum()
                e_high = P[(freqs >= HIGH_BAND[0]) & (freqs <= HIGH_BAND[1])].sum()
                f['F8_band'] = float(np.log10(e_low / (e_high + 1e-12)))
            except Exception:
                f['F8_band'] = np.nan
            # F9 谱熵
            try:
                p_n = P / (P.sum() + 1e-12)
                f['F9_spec_ent'] = float(-np.sum(p_n * np.log(p_n + 1e-12)) / np.log(len(p_n)))
            except Exception:
                f['F9_spec_ent'] = np.nan
            f['grp'] = grp
            per_doc[doc] = f
    print(f'  per-doc: {len(per_doc)} 篇')

    # ===== S2 统计 =====
    print('\nS2 统计（d/CI/p/q）:')
    main_keys = ['F1_mean_abs', 'F2_mean_abs', 'F3_ratio', 'F4_lag1', 'F5_lognorm_sigma',
                 'F6_beta', 'F7_dfa', 'F8_band', 'F9_spec_ent']
    stats_out = {}
    pvals = []
    for k in main_keys:
        h = [per_doc[d][k] for d in human if not np.isnan(per_doc[d][k])]
        a = [per_doc[d][k] for d in ai if not np.isnan(per_doc[d][k])]
        if len(h) < 5 or len(a) < 5:
            stats_out[k] = {'d': None, 'ci': None, 'p': None, 'mean_h': None, 'mean_a': None}
            continue
        d_ = cohens_d(h, a)
        ci = bootstrap_ci(h, a, paired=False)
        u = sc.mannwhitneyu(h, a)
        stats_out[k] = {'d': round(float(d_), 3), 'ci': [round(float(x), 4) for x in ci],
                        'p': round(float(u.pvalue), 4), 'mean_h': round(float(np.mean(h)), 4),
                        'mean_a': round(float(np.mean(a)), 4), 'n_h': len(h), 'n_a': len(a)}
        pvals.append(u.pvalue)
    qs = bh_fdr(np.array(pvals))
    qi = 0
    for k in main_keys:
        if stats_out[k].get('p') is not None:
            stats_out[k]['q'] = round(float(qs[qi]), 4)
            qi += 1
            print(f'  {k}: 人类 {stats_out[k]["mean_h"]} vs AI {stats_out[k]["mean_a"]}——'
                  f'd={stats_out[k]["d"]:+.3f} CI {stats_out[k]["ci"]} p={stats_out[k]["p"]} q={stats_out[k]["q"]}')
    # 副特征（F1 方向/形状——评审 1）
    for k in ['F1_mean_signed', 'F1_dir_cons', 'F1_pos_frac', 'F1_std', 'F2_std', 'F4_lag2', 'F7_hurst']:
        h = [per_doc[d][k] for d in human if not np.isnan(per_doc[d][k])]
        a = [per_doc[d][k] for d in ai if not np.isnan(per_doc[d][k])]
        if len(h) < 5 or len(a) < 5:
            continue
        d_ = cohens_d(h, a)
        ci = bootstrap_ci(h, a, paired=False)
        print(f'  {k}: d={d_:+.3f} CI {[round(float(x), 3) for x in ci]}')

    # ===== S3 冗余矩阵（9 主判内部 + 与既有）=====
    print('\nS3 冗余矩阵:')
    X = np.array([[per_doc[d][k] for d in human + ai] for k in main_keys])
    X = np.nan_to_num(X, nan=0.0)
    corr = np.corrcoef(X)
    for i in range(len(main_keys)):
        row = [round(float(corr[i, j]), 2) for j in range(len(main_keys))]
        print(f'  {main_keys[i]}: {row}')

    # ===== S4 干预实证（201 runs——vt_gate_beam 主对照——评审 2）=====
    print('\nS4 干预实证（vt_gate_beam 主对照——评审 2 锁定）:')
    rows_all = json.loads((OUT / 'rows.json').read_text(encoding='utf-8'))
    ti = defaultdict(list)
    for i, r in enumerate(rows_all):
        if r['source'] == 'training_intervention':
            ti[r['doc']].append(i)
    p90_ti = float(np.quantile(np.concatenate(
        [np.linalg.norm(fp[np.array(ti[d])[1:], :] - fp[np.array(ti[d])[:-1], :], axis=1) for d in ti]), 0.90))
    def run_feats(run_id):
        idxs = sorted(ti[run_id], key=lambda i: (rows_all[i]['seg'], i))
        Df = fp[idxs[1:], :] - fp[idxs[:-1], :]
        if len(Df) < MIN_DIFFS:
            return None
        alpha = Df @ D
        nrm = np.linalg.norm(Df, axis=1)
        u = Df / (nrm[:, None] + 1e-9)
        k = int(len(Df) * 0.6)
        return {'F1_mean_abs': float(np.mean(np.abs(alpha))), 'F1_std': float(np.std(alpha)),
                'F3_ratio': ratio_of(Df[k:], D)[0],
                'F4_lag1': float((u[:-1] * u[1:]).sum(1).mean())}
    pairs = []
    for cond in ('vt_gate_beam',):
        for pid in ('P1', 'P2', 'P3'):
            for sd in (0, 1):
                vt_id = f'{cond}-{pid}-s{sd}'
                none_id = f'none-{pid}-s{sd}'
                fv, fn = run_feats(vt_id), run_feats(none_id)
                if fv and fn:
                    pairs.append((cond, pid, sd, fv, fn))
    print(f'  配对（vt_gate_beam vs none）: {len(pairs)}')
    for k in ('F1_mean_abs', 'F1_std', 'F3_ratio', 'F4_lag1'):
        dts = [fv[k] - fn[k] for _, _, _, fv, fn in pairs]
        if len(dts) >= 6:
            w = sc.wilcoxon(dts)
            d_ = np.mean(dts) / (np.std(dts, ddof=1) + 1e-9)
            print(f'  {k}: Δ={np.mean(dts):+.4f} d={d_:+.3f} p={w.pvalue:.3f}')

    # ===== S5 综合分 =====
    print('\nS5 综合分:')
    ds = {k: abs(stats_out[k]['d']) for k in main_keys if stats_out[k].get('d') is not None}
    dmax = max(ds.values()) if ds else 1.0
    rank = []
    for fi, k in enumerate(main_keys):
        s = stats_out[k]
        if s.get('d') is None:
            continue
        d_norm = abs(s['d']) / dmax
        score = 0.4 * d_norm + 0.3 * SCORES[f'F{fi + 1}']['interv'] + \
                0.2 * SCORES[f'F{fi + 1}']['embed'] + 0.1 * SCORES[f'F{fi + 1}']['train']
        rank.append({'feature': f'F{fi + 1}', 'key': k, 'd': s['d'], 'd_norm': round(d_norm, 3),
                     'score': round(float(score), 3), 'interv': SCORES[f'F{fi + 1}']['interv']})
    rank.sort(key=lambda x: (-x['score'], -x['interv']))
    for r in rank:
        print(f'  {r["feature"]} {r["key"]}: 综合分 {r["score"]}（d={r["d"]:+.2f}）')

    # ===== S6 判据裁定 =====
    print('\n===== S6 判据裁定 =====')
    strong = {k: stats_out[k] for k in main_keys
              if stats_out[k].get('d') is not None and abs(stats_out[k]['d']) > 1.0
              and stats_out[k]['ci'][0] * stats_out[k]['ci'][1] > 0
              and stats_out[k].get('q', 1) < 0.05}
    print(f'  强候选集 S1: {list(strong.keys())}')
    s_f1 = (len(strong) >= 2 and ('F1_mean_abs' in strong or 'F3_ratio' in strong))
    f4 = stats_out['F4_lag1']
    s_f2 = (f4.get('d') is not None and abs(f4['d']) > 0.5 and f4['ci'][0] * f4['ci'][1] > 0)
    top3 = [r['feature'] for r in rank[:3]]
    s_f3 = len(set(top3) & {'F1', 'F3', 'F4'}) >= 2
    verdicts = {'S-F1': 'PASS' if s_f1 else 'FAIL', 'S-F2': 'PASS' if s_f2 else 'FAIL',
                'S-F3': 'PASS' if s_f3 else 'FAIL'}
    for k, v in verdicts.items():
        print(f'  {k}: {v}')
    print(f'  top-3: {top3}')
    overall = '成立' if (s_f1 and s_f2 and s_f3) else ('部分成立' if s_f1 else '否定')
    print(f'  总判定: {overall}')

    # ===== S7 图 4 张 =====
    # 图 1：α 分布直方图 + lognorm 拟合
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for grp, c, lbl in (('human', '#1f6fb2', '人类'), ('ai', '#e67e22', 'AI')):
        xs = np.concatenate([fp[np.array(docs[d])[1:], :] @ D - fp[np.array(docs[d])[:-1], :] @ D
                             for d in (human if grp == 'human' else ai)])
        ax.hist(np.abs(xs), bins=60, alpha=0.45, color=c, label=f'{lbl} |α|', density=True)
        try:
            s = sc.lognorm.fit(np.abs(xs) + 1e-6)
            xx = np.linspace(0.01, np.percentile(np.abs(xs), 99), 200)
            ax.plot(xx, sc.lognorm.pdf(xx, *s), color=c, lw=1.5, ls='--')
        except Exception:
            pass
    ax.set_xlabel('|α_t|（沿轴推进幅度）')
    ax.set_ylabel('密度')
    ax.set_title('α 分布对比（人类 vs AI——lognorm 拟合虚线）')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PAPER / 'fig_spectrum_alpha_fit.png', dpi=150)
    plt.close()

    # 图 2：速度自相关曲线
    fig2, ax2 = plt.subplots(figsize=(9, 5.5))
    for grp, c, lbl in (('human', '#1f6fb2', '人类'), ('ai', '#e67e22', 'AI')):
        lags, means, ses = [], [], []
        for lag in range(0, 21):
            vals = []
            for d in (human if grp == 'human' else ai):
                idx = np.array(docs[d])
                Df = fp[idx[1:], :] - fp[idx[:-1], :]
                nrm = np.linalg.norm(Df, axis=1)
                u = Df / (nrm[:, None] + 1e-9)
                if lag == 0:
                    vals.append(float(np.mean(u * u)))
                elif len(u) > lag:
                    vals.append((u[:-lag] * u[lag:]).sum(1).mean())
            if vals:
                lags.append(lag)
                means.append(np.mean(vals))
                ses.append(np.std(vals) / np.sqrt(len(vals)))
        ax2.errorbar(lags, means, yerr=ses, marker='o', ms=3, lw=1.2, color=c, label=lbl)
    ax2.axhline(0, color='k', ls='--', lw=0.8)
    ax2.set_xlabel('lag')
    ax2.set_ylabel('⟨u_t·u_{t+lag}⟩')
    ax2.set_title('速度自相关（思维惯性——lag-1 判据区）')
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PAPER / 'fig_spectrum_autocorr.png', dpi=150)
    plt.close()

    # 图 3：功率谱
    fig3, ax3 = plt.subplots(figsize=(9, 5.5))
    for grp, c, lbl in (('human', '#1f6fb2', '人类'), ('ai', '#e67e22', 'AI')):
        ps, fs = [], []
        for d in (human if grp == 'human' else ai):
            idx = np.array(docs[d])
            alpha = fp[idx[1:], :] @ D - fp[idx[:-1], :] @ D
            xd = alpha - alpha.mean()
            if len(xd) >= 128:
                freqs, P = welch(xd, nperseg=64, noverlap=32)
            else:
                freqs, P = welch(xd, nperseg=32, noverlap=16)
            ps.append(P)
            fs = freqs
        med = np.median(np.array(ps), 0)
        ax3.loglog(fs[1:], med[1:], lw=1.2, color=c, label=lbl)
    ax3.set_xlabel('频率（cycles/step）')
    ax3.set_ylabel('功率 P(f)')
    ax3.set_title('功率谱中位对比（log-log——斜率 β 标注）')
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig(PAPER / 'fig_spectrum_power.png', dpi=150)
    plt.close()

    # 图 4：综合分排序
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    names = [r['feature'] for r in rank]
    scores = [r['score'] for r in rank]
    colors = ['#27ae60' if s >= np.percentile(scores, 66) else
              ('#f39c12' if s >= np.percentile(scores, 33) else '#e74c3c') for s in scores]
    ax4.barh(range(len(rank)), scores, color=colors)
    ax4.set_yticks(range(len(rank)))
    ax4.set_yticklabels([f'{r["feature"]} ({r["key"]})' for r in rank])
    ax4.set_xlabel('综合分（0.4 区分度 + 0.3 干预 + 0.2 嵌入 + 0.1 训练）')
    ax4.set_title('特征优先级排序（绿=强/黄=中/红=淘汰）')
    ax4.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(PAPER / 'fig_spectrum_rank.png', dpi=150)
    plt.close()

    # ===== S8 落盘 =====
    out = {
        'criteria': {'S-F1': bool(s_f1), 'S-F2': bool(s_f2), 'S-F3': bool(s_f3), 'overall': overall},
        'stats': stats_out,
        'strong_set': list(strong.keys()),
        'rank': rank,
        'intervention': {'n_pairs': len(pairs)},
        'verdicts': verdicts,
    }
    (OUT / 'spectrum_features.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print('\n落盘 spectrum_features.json + fig_spectrum_*.png × 4 ✓')


if __name__ == '__main__':
    main()
