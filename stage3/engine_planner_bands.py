# -*- coding: utf-8 -*-
"""v0.77-1 意图动力学引擎——轨迹规划器第一模块：波动模式学习（目标分布）

从人类 vs AI 的实测指纹数据刻画"目标状态分布"——轨迹规划器的输入：
1. **五指标目标分布**：跳跃度/转折陡度/Hurst/转移熵/耦合——每篇一个值——人类 vs AI 分布带（mean±std/p25-p75）
   ——规划器生成的轨迹应落入人类分布带（"正向宽/负向窄波动带"的可执行形式）
2. **64 维波动带**：每篇每维段轨迹 std（波动宽度）+ 均值——人类 vs AI 逐维对比
   ——验证"正向宽/负向窄"假设：人类组织组（dim10/11/34/46/48/59）人类波动宽度 > AI；
   AI 特征组（dim22/26/43/52/5）人类波动宽度 < AI（人类稳定不摆动）

纯离线（fp_matrix.npz 已落盘）——输出 planner_targets.json + 图 2 张 + REPORT 节
"""
import sys, json, os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats as sc

BASE = Path('C:/Users/bai/Desktop/小说系统')
OUT = BASE / 'data' / 'dim_analysis'
PAPER = Path('C:/Users/bai/Desktop/AB系统论文储备')
sys.path.insert(0, str(BASE / 'stage3'))
import dim_flow_sent as DFS  # 复用 transfer_entropy/turn_points

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

TARGET = [5, 10, 11, 22, 26, 34, 43, 46, 48, 52, 59]          # 11 维目标集
HUMAN_ORG = [10, 11, 34, 46, 48, 59]                            # 人类组织组（正极性）
AI_FEAT = [22, 26, 43, 52, 5]                                   # AI 特征组（负极性）


def load_docs():
    d = np.load(OUT / 'fp_matrix.npz')
    fp = d['fp']
    rows = json.loads((OUT / 'rows.json').read_text(encoding='utf-8'))
    doc_clauses = defaultdict(list)
    for i, r in enumerate(rows):
        if r['source'] == 'bilingual_zh':
            doc_clauses[r['doc']].append((r['para'], i))
    docs = {}
    for doc, items in doc_clauses.items():
        items.sort(key=lambda x: (x[0], x[1]))
        docs[doc] = [i for _, i in items]
    human = sorted([x for x in docs if x.startswith('ZH-H')], key=lambda x: len(docs[x]))
    ai = sorted([x for x in docs if x.startswith('ZH-A')], key=lambda x: len(docs[x]))
    return fp, rows, docs, human, ai


def seg_trajectory(fp, idx, dim):
    """段轨迹：每段句元均值 → 段序列"""
    paras = np.array([rows[i]['para'] for i in idx])
    segs = []
    for p in sorted(set(paras.tolist())):
        sel = fp[np.array(idx)[paras == p], dim]
        segs.append(float(np.mean(sel)))
    return np.array(segs)


def sliding_coupling(x, y, win=3):
    """滑动窗口相关（dim10×dim48 段级）——与 dim_flow.py 口径一致"""
    n = len(x)
    if n < win + 2:
        return float(np.corrcoef(x, y)[0, 1]) if n > 1 else 0.0
    cs = []
    for i in range(0, n - win + 1):
        a, b = x[i:i + win], y[i:i + win]
        if np.std(a) > 1e-9 and np.std(b) > 1e-9:
            cs.append(np.corrcoef(a, b)[0, 1])
    return float(np.mean(cs)) if cs else 0.0


def dist_stats(vals):
    v = np.asarray(vals, float)
    return {'mean': round(float(np.mean(v)), 4), 'std': round(float(np.std(v)), 4),
            'median': round(float(np.median(v)), 4),
            'p25': round(float(np.percentile(v, 25)), 4),
            'p75': round(float(np.percentile(v, 75)), 4)}


def main():
    global rows
    fp, rows, docs, human_docs, ai_docs = load_docs()
    print(f'句元级: 人类 {len(human_docs)} 篇——AI {len(ai_docs)} 篇')

    # ========== 1. 五指标目标分布（句元级——每篇一个值） ==========
    METRIC_DIMS = [10, 48]
    five = defaultdict(lambda: defaultdict(dict))  # metric -> grp -> doc -> val
    per_doc_stats = {'human': {}, 'ai': {}}
    for grp, dlist in (('human', human_docs), ('ai', ai_docs)):
        for doc in dlist:
            idx = np.array(docs[doc])
            t10, t48 = fp[idx, 10], fp[idx, 48]
            st = {
                'n': int(len(t10)),
                'dim10_jump': float(np.mean(np.abs(np.diff(t10)))),
                'dim48_jump': float(np.mean(np.abs(np.diff(t48)))),
                'dim10_hurst': DFS.DF.hurst(t10),
                'dim48_hurst': DFS.DF.hurst(t48),
                'te_10_to_48': DFS.transfer_entropy(t10, t48),
                'te_48_to_10': DFS.transfer_entropy(t48, t10),
                'turn10_steep': DFS.turn_points(t10)['peak_steep_mean'],
                'turn10_depth': DFS.turn_points(t10)['valley_depth_mean'],
                'turn48_steep': DFS.turn_points(t48)['peak_steep_mean'],
                'turn48_depth': DFS.turn_points(t48)['valley_depth_mean'],
                # 段级耦合（dim10×dim48 滑动窗）
                'coupling_10_48': sliding_coupling(*[seg_trajectory(fp, idx, dim) for dim in (10, 48)]),
            }
            per_doc_stats[grp][doc] = st

    METRIC_KEYS = ['dim10_jump', 'dim48_jump', 'dim10_hurst', 'dim48_hurst',
                   'te_10_to_48', 'te_48_to_10',
                   'turn10_steep', 'turn10_depth', 'turn48_steep', 'turn48_depth',
                   'coupling_10_48']
    METRIC_LABEL = {
        'dim10_jump': '跳跃度 dim10', 'dim48_jump': '跳跃度 dim48',
        'dim10_hurst': '长程记忆 Hurst dim10', 'dim48_hurst': '长程记忆 Hurst dim48',
        'te_10_to_48': '转移熵 主题→指代', 'te_48_to_10': '转移熵 指代→主题',
        'turn10_steep': '转折陡度 dim10', 'turn10_depth': '转折深度 dim10',
        'turn48_steep': '转折陡度 dim48', 'turn48_depth': '转折深度 dim48',
        'coupling_10_48': '主题×指代耦合',
    }
    five_res = {}
    print('=== 五指标目标分布（人类 vs AI——每篇一个值）===')
    for k in METRIC_KEYS:
        h = [per_doc_stats['human'][x][k] for x in human_docs]
        a = [per_doc_stats['ai'][x][k] for x in ai_docs]
        u = sc.mannwhitneyu(h, a)
        d_ = (np.mean(h) - np.mean(a)) / np.sqrt((np.var(h) + np.var(a)) / 2 + 1e-9)
        five_res[k] = {'human': dist_stats(h), 'ai': dist_stats(a),
                       'd': round(float(d_), 3), 'p': round(float(u.pvalue), 4),
                       'target_band': [dist_stats(h)['p25'], dist_stats(h)['p75']],
                       'label': METRIC_LABEL[k]}
        print(f'  {k}: 人类 {five_res[k]["human"]["mean"]}±{five_res[k]["human"]["std"]}'
              f' vs AI {five_res[k]["ai"]["mean"]}±{five_res[k]["ai"]["std"]}——d={d_:+.2f} p={u.pvalue:.3f}')

    # ========== 2. 64 维波动带（段轨迹 std = 波动宽度） ==========
    print('=== 64 维波动带（段轨迹 std）===')
    dim_bands = []
    for dim in range(64):
        h_means, h_stds, a_means, a_stds = [], [], [], []
        for doc in human_docs:
            traj = seg_trajectory(fp, np.array(docs[doc]), dim)
            if len(traj) >= 3:
                h_means.append(float(np.mean(traj)))
                h_stds.append(float(np.std(traj)))
        for doc in ai_docs:
            traj = seg_trajectory(fp, np.array(docs[doc]), dim)
            if len(traj) >= 3:
                a_means.append(float(np.mean(traj)))
                a_stds.append(float(np.std(traj)))
        if not h_stds or not a_stds:
            continue
        # 波动宽度比（人类 std / AI std——人类段数差异大——用中位数）
        hm, am = np.median(h_means), np.median(a_means)
        hs, as_ = np.median(h_stds), np.median(a_stds)
        width_ratio = hs / (as_ + 1e-9)
        # 波动宽度 d（人类 vs AI——篇级 std 分布）
        d_w = (np.mean(h_stds) - np.mean(a_stds)) / np.sqrt((np.var(h_stds) + np.var(a_stds)) / 2 + 1e-9)
        dim_bands.append({'dim': dim,
                          'human': {'mean': round(hm, 4), 'band_std': round(hs, 4)},
                          'ai': {'mean': round(am, 4), 'band_std': round(as_, 4)},
                          'width_ratio': round(float(width_ratio), 3),
                          'width_d': round(float(d_w), 3),
                          'in_human_org': dim in HUMAN_ORG,
                          'in_ai_feat': dim in AI_FEAT})

    # 极性验证：人类组织组宽度比 vs AI 特征组宽度比
    org_ratios = [b['width_ratio'] for b in dim_bands if b['in_human_org']]
    feat_ratios = [b['width_ratio'] for b in dim_bands if b['in_ai_feat']]
    print(f'=== 极性波动带验证 ===')
    print(f'  人类组织组（{HUMAN_ORG}）: 宽度比 {[round(r,2) for r in org_ratios]}——中位 {np.median(org_ratios):.2f}')
    print(f'  AI 特征组（{AI_FEAT}）: 宽度比 {[round(r,2) for r in feat_ratios]}——中位 {np.median(feat_ratios):.2f}')
    # 全部 64 维：正极性（人类宽）维度数 vs 负极性
    n_wide = sum(1 for b in dim_bands if b['width_ratio'] > 1.0)
    print(f'  64 维中人类波动宽于 AI: {n_wide}/{len(dim_bands)}')

    # ========== 3. 图 1：五指标目标分布（人类分布带 vs AI——均值±std） ==========
    fig, ax = plt.subplots(figsize=(12, 6))
    xpos = np.arange(len(METRIC_KEYS))
    for i, k in enumerate(METRIC_KEYS):
        h = five_res[k]
        hb, ab = h['human'], h['ai']
        # 人类目标带（p25-p75——半透明）
        ax.barh(i, hb['p75'] - hb['p25'], left=hb['p25'], height=0.55,
                color='#1f6fb2', alpha=0.35, label='人类目标带 p25-p75' if i == 0 else None)
        ax.scatter([hb['mean']], [i], color='#1f6fb2', s=60, zorder=3,
                   label='人类均值' if i == 0 else None)
        # AI 均值±std
        ax.errorbar(ab['mean'], i + 0.42, xerr=ab['std'], fmt='o', color='#e67e22',
                    capsize=3, ms=5, label='AI 均值±std' if i == 0 else None)
    ax.set_yticks(xpos)
    ax.set_yticklabels([METRIC_LABEL[k] for k in METRIC_KEYS], fontsize=9)
    ax.set_xlabel('指标值')
    ax.set_title('轨迹规划器目标分布：五指标（人类分布带 vs AI——规划器目标=落入人类带）')
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(PAPER / 'fig_planner_targets.png', dpi=150)
    plt.close()

    # ========== 4. 图 2：64 维波动带（宽度比散点 + 极性标注） ==========
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    dims = [b['dim'] for b in dim_bands]
    ratios = [b['width_ratio'] for b in dim_bands]
    cols = []
    for b in dim_bands:
        if b['in_human_org']:
            cols.append('#1f6fb2')
        elif b['in_ai_feat']:
            cols.append('#e67e22')
        else:
            cols.append('#999999')
    ax2.bar(dims, ratios, color=cols, alpha=0.85)
    ax2.axhline(1.0, color='#c0392b', ls='--', lw=1)
    ax2.set_xlabel('维度（64 维）')
    ax2.set_ylabel('波动宽度比（人类 std / AI std）')
    ax2.set_title('64 维波动带：人类 vs AI 段轨迹波动宽度（>1=人类波动更宽——蓝=人类组织组 橙=AI 特征组）')
    from matplotlib.patches import Patch
    ax2.legend(handles=[Patch(color='#1f6fb2', label='人类组织组（正极性）'),
                        Patch(color='#e67e22', label='AI 特征组（负极性）'),
                        Patch(color='#999999', label='其余')], fontsize=8)
    ax2.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(PAPER / 'fig_planner_bands.png', dpi=150)
    plt.close()

    # ========== 落盘 ==========
    out = {
        'five_metrics': five_res,
        'per_doc_stats': per_doc_stats,
        'dim_bands': dim_bands,
        'polarity': {
            'human_org_dims': HUMAN_ORG, 'ai_feat_dims': AI_FEAT,
            'human_org_width_ratio_median': round(float(np.median(org_ratios)), 3),
            'ai_feat_width_ratio_median': round(float(np.median(feat_ratios)), 3),
            'org_ratios': {str(d): r for d, r in zip(HUMAN_ORG, org_ratios)},
            'feat_ratios': {str(d): r for d, r in zip(AI_FEAT, feat_ratios)},
            'n_dim_human_wide': n_wide, 'n_dim_total': len(dim_bands),
        },
    }
    (OUT / 'planner_targets.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print('落盘 planner_targets.json + fig_planner_targets.png + fig_planner_bands.png ✓')


if __name__ == '__main__':
    main()
