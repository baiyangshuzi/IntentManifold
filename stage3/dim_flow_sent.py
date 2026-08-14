# -*- coding: utf-8 -*-
"""v0.74-3 句元级意图流转 + 转移熵（细粒度度量——用户修正"流形/断裂点无差"）

句元级轨迹：每句元 dim10/dim48 激活（人类 ~772 点/AI ~509 点）——跳跃度/Hurst/转折点类型
转移熵：TE(dim10→dim48) 与 TE(dim48→dim10)——方向性信息流（符号化 3-bin）
产出图放论文储备路径（fig_flow_sent_*.png）
"""
import sys, json, os, re
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats as sc

BASE = Path('C:/Users/bai/Desktop/小说系统')
OUT = BASE / 'data' / 'dim_analysis'
PAPER = Path('C:/Users/bai/Desktop/AB系统论文储备')
sys.path.insert(0, str(BASE / 'stage3'))
import dim_flow as DF  # 复用 hurst/paras_of

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
TARGET = [5, 10, 11, 22, 26, 34, 43, 46, 48, 52, 59]


def transfer_entropy(x, y, bins=3, lag=1):
    """TE(X→Y)：Y 的未来在给定 Y 过去与 X 过去的条件下的信息增益
    符号化（分位数 3-bin）——TE = H(Yt|Ypast) - H(Yt|Ypast, Xpast)"""
    def qbin(z):
        qs = np.quantile(z, [1 / 3, 2 / 3])
        return np.digitize(z, qs)
    xs, ys = qbin(x), qbin(y)
    n = len(xs) - lag
    # 联合计数（3×3×3）
    cnt_yy = np.zeros((bins, bins))
    cnt_yyx = np.zeros((bins, bins, bins))
    for t in range(lag, len(xs)):
        cnt_yy[ys[t - lag], ys[t]] += 1
        cnt_yyx[xs[t - lag], ys[t - lag], ys[t]] += 1
    p_yy = cnt_yy / cnt_yy.sum()
    p_yyx = cnt_yyx / cnt_yyx.sum()
    p_y = p_yy.sum(0)
    p_cond_y = p_yy / (p_yy.sum(1, keepdims=True) + 1e-12)
    p_cond_yx = p_yyx / (p_yyx.sum(2, keepdims=True) + 1e-12)
    te = 0.0
    for xb in range(bins):
        for yb in range(bins):
            for yf in range(bins):
                if p_yyx[xb, yb, yf] > 0 and p_cond_y[yb, yf] > 0:
                    te += p_yyx[xb, yb, yf] * np.log2(
                        p_cond_yx[xb, yb, yf] / (p_cond_y[yb, yf] + 1e-12))
    return max(te, 0.0)


def turn_points(z):
    """转折点类型：峰（局部极大）/谷（局部极小）——陡度/深度"""
    z = np.asarray(z, float)
    d = np.diff(z)
    peaks = np.where((d[:-1] > 0) & (d[1:] <= 0))[0] + 1
    valleys = np.where((d[:-1] < 0) & (d[1:] >= 0))[0] + 1
    # 峰陡度：峰两侧下降幅度均值
    peak_steep = [abs(z[p] - min(z[p - 1], z[p + 1])) for p in peaks if 0 < p < len(z) - 1]
    val_depth = [abs(max(z[p - 1], z[p + 1]) - z[p]) for p in valleys if 0 < p < len(z) - 1]
    return {'n_peaks': len(peaks), 'n_valleys': len(valleys),
            'peak_steep_mean': float(np.mean(peak_steep)) if peak_steep else 0.0,
            'valley_depth_mean': float(np.mean(val_depth)) if val_depth else 0.0}


def main():
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    d = np.load(OUT / 'fp_matrix.npz')
    fp = d['fp']
    rows = json.loads((OUT / 'rows.json').read_text(encoding='utf-8'))

    # 句元级序列（bilingual zh——按 doc 按段内顺序拼接——rows 顺序即句元顺序）
    from collections import defaultdict
    doc_clauses = defaultdict(list)  # doc -> [(para, row_idx)]
    for i, r in enumerate(rows):
        if r['source'] == 'bilingual_zh':
            doc_clauses[r['doc']].append((r['para'], i))
    docs = {}
    for doc, items in doc_clauses.items():
        # 按段排序（段内保持 rows 顺序）
        items.sort(key=lambda x: (x[0], x[1]))
        docs[doc] = [i for _, i in items]

    human_docs = sorted([x for x in docs if x.startswith('ZH-H')], key=lambda x: len(docs[x]))
    ai_docs = sorted([x for x in docs if x.startswith('ZH-A')], key=lambda x: len(docs[x]))
    print(f'句元级: 人类 {len(human_docs)} 篇（{len(docs[human_docs[-1]])} 句元/最长）——AI {len(ai_docs)} 篇')

    # ===== 句元级指标 =====
    sent_stats = {'human': {}, 'ai': {}}
    for grp, dlist in (('human', human_docs), ('ai', ai_docs)):
        for doc in dlist:
            idx = np.array(docs[doc])
            t10 = fp[idx, 10]
            t48 = fp[idx, 48]
            sent_stats[grp][doc] = {
                'n': len(t10),
                'dim10_jump': float(np.mean(np.abs(np.diff(t10)))),
                'dim48_jump': float(np.mean(np.abs(np.diff(t48)))),
                'dim10_hurst': DF.hurst(t10),
                'dim48_hurst': DF.hurst(t48),
                'te_10_to_48': transfer_entropy(t10, t48),
                'te_48_to_10': transfer_entropy(t48, t10),
                **{f'turn10_{k}': v for k, v in turn_points(t10).items()},
                **{f'turn48_{k}': v for k, v in turn_points(t48).items()},
            }
    print('=== 句元级指标（人类 vs AI——中位数）===')
    keys = ['dim10_jump', 'dim48_jump', 'dim10_hurst', 'dim48_hurst',
            'te_10_to_48', 'te_48_to_10',
            'turn10_peak_steep_mean', 'turn10_valley_depth_mean',
            'turn48_peak_steep_mean', 'turn48_valley_depth_mean']
    res = {}
    for k in keys:
        h = [sent_stats['human'][x][k] for x in human_docs]
        a = [sent_stats['ai'][x][k] for x in ai_docs]
        u = sc.mannwhitneyu(h, a)
        d_ = (np.mean(h) - np.mean(a)) / np.sqrt((np.var(h) + np.var(a)) / 2 + 1e-9)
        res[k] = {'human': round(float(np.nanmedian(h)), 4), 'ai': round(float(np.nanmedian(a)), 4),
                  'd': round(float(d_), 3), 'p': round(float(u.pvalue), 4)}
        print(f'  {k}: 人类 {res[k]["human"]} vs AI {res[k]["ai"]}——d={d_:+.2f} p={u.pvalue:.3f}')

    # ===== 句元级轨迹图（代表篇——人类 vs AI——dim10 前 300 句元）=====
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    rep_h = human_docs[-1]
    rep_a = ai_docs[-1]
    for ax, doc, grp in ((axes[0], rep_h, 'human'), (axes[1], rep_a, 'ai')):
        idx = np.array(docs[doc])
        t10 = fp[idx, 10][:300]
        t48 = fp[idx, 48][:300]
        ax.plot(t10, label='dim10 主题', color='#1f6fb2', lw=0.8)
        ax.plot(t48, label='dim48 指代', color='#e67e22', lw=0.8)
        ax.set_title(f'{grp} 句元级轨迹（{doc}——前 300 句元）')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PAPER / 'fig_flow_sent_trajectory.png', dpi=150)
    plt.close()

    # ===== 转移熵图 =====
    fig2, ax = plt.subplots(figsize=(7, 5))
    groups = []
    for grp in ('human', 'ai'):
        te_10_48 = [sent_stats[grp][x]['te_10_to_48'] for x in (human_docs if grp == 'human' else ai_docs)]
        te_48_10 = [sent_stats[grp][x]['te_48_to_10'] for x in (human_docs if grp == 'human' else ai_docs)]
        groups.append((grp, te_10_48, te_48_10))
    xpos = np.arange(2)
    w = 0.35
    for i, (grp, te_a, te_b) in enumerate(groups):
        ax.bar(xpos + (i - 0.5) * w, [np.mean(te_a), np.mean(te_b)], w,
               label=grp, color=('#1f6fb2' if grp == 'human' else '#e67e22'))
    ax.set_xticks(xpos)
    ax.set_xticklabels(['TE(dim10→dim48)\n主题→指代', 'TE(dim48→dim10)\n指代→主题'])
    ax.set_ylabel('转移熵（bits——符号化 3-bin）')
    ax.set_title('转移熵：方向性信息流（人类 vs AI——句元级）')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(PAPER / 'fig_flow_te.png', dpi=150)
    plt.close()

    out = {'sent_stats': sent_stats, 'metrics': res}
    (OUT / 'flow_sent_analysis.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print('落盘 flow_sent_analysis.json + 论文储备 figs ✓')


if __name__ == '__main__':
    main()
