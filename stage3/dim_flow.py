# -*- coding: utf-8 -*-
"""v0.74-2 意图流转（Flow of Thought）捕捉与深度分析——"思维心电图"

模块一：单维度轨迹（dim10 主题/dim48 指代衔接——段序列——跳跃度/趋势/Hurst 长程自相关）
模块二：维度耦合（滑动窗口 dim10×dim48 相关——耦合曲线）
模块三：思想流形（11 维 PCA → 2D 段落轨迹——螺旋 vs 随机游走——中心距离/转向角）
产出四：思维断裂点（dim48 断档检测——定位"开小差"段落）
数据：fp_matrix.npz（bilingual zh——人类 10 篇 vs AI 20 篇——段级轨迹）
纯离线——CPU——无 GPU/API
"""
import sys, json, os, re
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats as sc

BASE = Path(os.environ.get('INTENT_DYNAMICS_BASE', Path(__file__).resolve().parent.parent))
OUT = BASE / 'data' / 'dim_analysis'
FIGS = OUT / 'figs'
FIGS.mkdir(exist_ok=True)
sys.path.insert(0, str(BASE / 'stage3'))

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
TARGET = [5, 10, 11, 22, 26, 34, 43, 46, 48, 52, 59]
CH_RE = re.compile(r'^第[一二三四五六七八九十百千0-9]+[章节卷回部]')


def paras_of(text):
    """行级段落切分（≥30 字含汉字非标题）——与既有管线同口径"""
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    return [l for l in lines if re.search(r'[一-龥]', l) and len(l) >= 30 and not CH_RE.match(l)]


def hurst(series):
    """RS 分析——Hurst 指数（>0.5 长程正相关/记忆性——<0.5 反持续——=0.5 随机游走）"""
    s = np.asarray(series, float)
    s = s - s.mean()
    N = len(s)
    if N < 16:
        return np.nan
    scales = []
    rs_vals = []
    # 子区间长度从 N//2 往下到 8
    max_n = N // 2
    for n in np.unique(np.geomspace(8, max_n, 10).astype(int)):
        if n < 8:
            continue
        n_chunks = N // n
        if n_chunks < 2:
            continue
        rs = []
        for c in range(n_chunks):
            chunk = s[c * n:(c + 1) * n]
            if chunk.std() < 1e-12:
                continue
            z = np.cumsum(chunk)
            R = z.max() - z.min()
            S = chunk.std()
            rs.append(R / S)
        if rs:
            scales.append(n)
            rs_vals.append(np.mean(rs))
    if len(scales) < 3:
        return np.nan
    sl, _, rv, _, _ = sc.linregress(np.log(scales), np.log(rs_vals))
    return float(sl)


def compute_extra(enc, disc):
    """补算同题材超长轨迹：AI archives（146 段）+ 人类天行健前 6 章（1076 段）
    返回 {doc: {para: idx_list}} + fp 追加矩阵（不并入主矩阵——独立返回）"""
    import torch
    from para_dimensions import fingerprint, norm_rows
    from subclause_structure import split_subclauses
    extra_docs = {}
    extra_fp = []
    # AI archives 6 章
    texts = []
    for i in range(1, 7):
        f = BASE / 'novel-project' / 'archives' / f'vol-1-ch-{i}-draft.md'
        if f.exists():
            texts.append(f.read_text(encoding='utf-8', errors='replace'))
    ai_text = '\n'.join(texts)
    # 人类天行健前 6 章（现成切片）
    hu_texts = []
    for i in range(1, 7):
        f = BASE / 'data' / 'chapters_txj' / f'chapter_{i:02d}.txt'
        if f.exists():
            hu_texts.append(f.read_text(encoding='utf-8', errors='replace'))
    hu_text = '\n'.join(hu_texts)

    if not texts or not hu_texts:
        print('  素材未内置（novel-project/archives 或 chapters_txj）——跳过同题材补算——主分析不受影响')
        return {}, np.zeros((0, 64))
    for doc, txt in (('archives_AI', ai_text), ('txj_human_6ch', hu_text)):
        paras = paras_of(txt)
        doc_segs = {}
        for pi, p in enumerate(paras):
            ss = [s for s in split_subclauses(p) if len(s) >= 3]
            if not ss:
                continue
            with torch.no_grad():
                sv = enc.encode(ss, normalize_embeddings=True, batch_size=64,
                                show_progress_bar=False, device='cpu')
                SV = torch.from_numpy(sv.astype(np.float32)).to('cpu')
                F = fingerprint(SV, disc).cpu().numpy()
            start = len(extra_fp)
            extra_fp.append(F)
            doc_segs[pi] = list(range(start, len(extra_fp)))
        extra_docs[doc] = doc_segs
        print(f'补算 {doc}: {len(paras)} 段 ✓')
    extra_fp = np.vstack(extra_fp).astype(np.float32) if extra_fp else np.zeros((0, 64))
    return extra_docs, extra_fp


def main():
    d = np.load(OUT / 'fp_matrix.npz')
    fp = d['fp']
    rows = json.loads((OUT / 'rows.json').read_text(encoding='utf-8'))

    # 段级轨迹（bilingual zh——按 doc 按 para 排序——每段句元均值）
    from collections import defaultdict
    segs = defaultdict(dict)  # doc -> {para: idx_list}
    for i, r in enumerate(rows):
        if r['source'] == 'bilingual_zh':
            segs[r['doc']].setdefault(r['para'], []).append(i)
    docs = {}
    for doc, paras in segs.items():
        order = sorted(paras)
        # 只保留连续段（相邻 para 索引连续——跳过缺失）
        docs[doc] = {p: paras[p] for p in order}

    # 同题材超长轨迹补算（archives AI + 天行健前 6 章人类——用户路径）
    import torch
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    from para_dimensions import load_models
    enc, disc = load_models('cpu')
    extra_docs, extra_fp = compute_extra(enc, disc)
    for doc, segs_d in extra_docs.items():
        docs[doc] = segs_d
    fp_all = np.vstack([fp, extra_fp]) if len(extra_fp) else fp
    fp = fp_all

    # 人类 vs AI 篇分类（ZH-H 人类/ZH-A AI）
    human_docs = sorted([x for x in docs if x.startswith('ZH-H')], key=lambda x: len(docs[x]))
    ai_docs = sorted([x for x in docs if x.startswith('ZH-A')], key=lambda x: len(docs[x]))
    print(f'人类 {len(human_docs)} 篇——AI {len(ai_docs)} 篇')

    def traj(doc, dim):
        paras = docs[doc]
        ps = sorted(paras)
        return np.array([fp[paras[p], dim].mean() for p in ps])

    # ===== 模块一：单维度轨迹 + 跳跃度 + 趋势 + Hurst =====
    stats_all = {'human': {}, 'ai': {}}
    for grp, dlist in (('human', human_docs), ('ai', ai_docs)):
        for doc in dlist:
            t10, t48 = traj(doc, 10), traj(doc, 48)
            stats_all[grp][doc] = {
                'n_seg': len(t10),
                'dim10_jump': float(np.mean(np.abs(np.diff(t10)))),
                'dim48_jump': float(np.mean(np.abs(np.diff(t48)))),
                'dim10_slope': float(np.polyfit(np.arange(len(t10)), t10, 1)[0]),
                'dim48_slope': float(np.polyfit(np.arange(len(t48)), t48, 1)[0]),
                'dim10_hurst': hurst(t10),
                'dim48_hurst': hurst(t48),
            }
    print('=== 模块一：轨迹统计（人类 vs AI——中位数）===')
    for grp in ('human', 'ai'):
        v = stats_all[grp]
        m = {k: float(np.nanmedian([x[k] for x in v.values()])) for k in
             ('dim10_jump', 'dim48_jump', 'dim10_slope', 'dim48_slope', 'dim10_hurst', 'dim48_hurst')}
        print(f'{grp}（n={len(v)}）: dim10跳跃={m["dim10_jump"]:.4f} dim48跳跃={m["dim48_jump"]:.4f} '
              f'dim10斜率={m["dim10_slope"]:+.5f} dim48斜率={m["dim48_slope"]:+.5f} '
              f'Hurst10={m["dim10_hurst"]:.3f} Hurst48={m["dim48_hurst"]:.3f}')
    # 组间检验
    for k in ('dim10_jump', 'dim48_jump', 'dim10_hurst', 'dim48_hurst'):
        h = [stats_all['human'][x][k] for x in human_docs]
        a = [stats_all['ai'][x][k] for x in ai_docs]
        u = sc.mannwhitneyu(h, a)
        d_ = (np.mean(h) - np.mean(a)) / np.sqrt((np.var(h) + np.var(a)) / 2 + 1e-9)
        print(f'  {k}: 人类 {np.nanmedian(h):.3f} vs AI {np.nanmedian(a):.3f}——d={d_:+.2f} p={u.pvalue:.3f}')

    # 代表篇轨迹图（人类 vs AI——dim10+dim48 双轴）
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    reps_h = [human_docs[-1]] if human_docs else []
    reps_a = [x for x in ai_docs if len(docs[x]) > 40][:2]
    for ax, doc, grp in ((axes[0][0], reps_h[0] if reps_h else None, 'human'),
                         (axes[0][1], None, 'ai')):
        pass
    if reps_h:
        doc = reps_h[0]
        t10, t48 = traj(doc, 10), traj(doc, 48)
        ax = axes[0][0]
        ax.plot(t10, label='dim10 主题', color='#1f6fb2', lw=1.5)
        ax.plot(t48, label='dim48 指代', color='#e67e22', lw=1.5)
        ax.set_title(f'人类 {doc}（{len(t10)} 段——意图轨迹）')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    if reps_a:
        doc = reps_a[0]
        t10, t48 = traj(doc, 10), traj(doc, 48)
        ax = axes[0][1]
        ax.plot(t10, label='dim10 主题', color='#1f6fb2', lw=1.2)
        ax.plot(t48, label='dim48 指代', color='#e67e22', lw=1.2)
        ax.set_title(f'AI {doc}（{len(t10)} 段——意图轨迹）')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    # 耦合曲线（模块二）
    def coupling(doc, w=3):
        t10, t48 = traj(doc, 10), traj(doc, 48)
        n = len(t10)
        if n < w + 2:
            return np.array([])
        cs = []
        for i in range(0, n - w + 1):
            a, b = t10[i:i + w], t48[i:i + w]
            if a.std() < 1e-9 or b.std() < 1e-9:
                cs.append(0.0)
            else:
                cs.append(float(np.corrcoef(a, b)[0, 1]))
        return np.array(cs)
    if reps_h:
        c = coupling(reps_h[0])
        axes[1][0].plot(c, color='#2e8b57', lw=1.5)
        axes[1][0].axhline(0, color='#999', ls=':', lw=1)
        axes[1][0].set_title(f'人类耦合（dim10×dim48——窗口3——均值 {np.mean(c):+.2f}）')
        axes[1][0].grid(alpha=0.3)
    if reps_a:
        c = coupling(reps_a[0])
        axes[1][1].plot(c, color='#2e8b57', lw=1.2)
        axes[1][1].axhline(0, color='#999', ls=':', lw=1)
        axes[1][1].set_title(f'AI 耦合（dim10×dim48——窗口3——均值 {np.mean(c):+.2f}）')
        axes[1][1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGS / 'fig_flow_trajectory.png', dpi=150)
    plt.close()

    # 耦合统计（全部篇）
    coup = {}
    for grp, dlist in (('human', human_docs), ('ai', ai_docs)):
        vals = []
        for doc in dlist:
            c = coupling(doc)
            if len(c) >= 3:
                vals.append(np.mean(c))
        coup[grp] = vals
    print('=== 模块二：耦合（dim10×dim48 窗口相关——篇均值）===')
    if coup['human'] and coup['ai']:
        u = sc.mannwhitneyu(coup['human'], coup['ai'])
        print(f'人类耦合均值 {np.mean(coup["human"]):+.3f}（n={len(coup["human"])}）vs AI '
              f'{np.mean(coup["ai"]):+.3f}（n={len(coup["ai"])}）——p={u.pvalue:.3f}')

    # ===== 模块三：思想流形（11 维 PCA——2D 轨迹）=====
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    # 用全部段落句元均值拟合 PCA
    all_para = []
    all_doc = []
    for doc in docs:
        for p in sorted(docs[doc]):
            idx = np.array(docs[doc][p])
            all_para.append(fp[idx][:, TARGET].mean(0))
            all_doc.append(doc)
    X = np.array(all_para)
    X2 = pca.fit_transform(X)
    print(f'=== 模块三：思想流形（PCA——explained {pca.explained_variance_ratio_.sum():.2f}）===')
    fig2, axes2 = plt.subplots(1, 2, figsize=(13, 6))
    for ax, grp, doc_reps in ((axes2[0], 'human', reps_h), (axes2[1], 'ai', reps_a)):
        for doc in doc_reps:
            idx = [i for i, x in enumerate(all_doc) if x == doc]
            pts = X2[idx]
            ax.plot(pts[:, 0], pts[:, 1], lw=1.2, marker='o', ms=3)
            ax.scatter(pts[0, 0], pts[0, 1], c='green', s=40, marker='*', zorder=5)
            ax.scatter(pts[-1, 0], pts[-1, 1], c='red', s=40, marker='x', zorder=5)
        ax.set_title(f'{grp} 思想流形（11 维 PCA——绿*起点红×终点）')
        ax.grid(alpha=0.3)
        ax.set_aspect('equal', adjustable='datalim')
    plt.tight_layout()
    plt.savefig(FIGS / 'fig_flow_manifold.png', dpi=150)
    plt.close()

    # 流形指标：轨迹半径（贴中心——平均到质心距离）/转向角
    manifold = {}
    for grp, dlist in (('human', human_docs), ('ai', ai_docs)):
        rads, turns = [], []
        for doc in dlist:
            idx = [i for i, x in enumerate(all_doc) if x == doc]
            pts = X2[idx]
            if len(pts) < 6:
                continue
            c = pts.mean(0)
            rads.append(float(np.mean(np.linalg.norm(pts - c, axis=1))))
            # 转向角变化率（相邻段方向角差）
            v = np.diff(pts, axis=0)
            ang = np.arctan2(v[:, 1], v[:, 0])
            turns.append(float(np.mean(np.abs(np.diff(ang)))))
        manifold[grp] = {'radius': float(np.mean(rads)) if rads else None,
                         'turn_rate': float(np.mean(turns)) if turns else None}
    print(f'流形指标: 人类 radius={manifold["human"]["radius"]:.3f} turn={manifold["human"]["turn_rate"]:.3f} | '
          f'AI radius={manifold["ai"]["radius"]:.3f} turn={manifold["ai"]["turn_rate"]:.3f}')

    # ===== 产出四：思维断裂点（dim48 断档检测）=====
    breaks = {}
    for grp, dlist in (('human', human_docs), ('ai', ai_docs)):
        densities = []
        for doc in dlist:
            t48 = traj(doc, 48)
            if len(t48) < 8:
                continue
            thr = np.percentile(t48, 25)  # 低激活=指代链弱
            n_break = int((np.diff((t48 < thr).astype(int)) > 0).sum())  # 进入低区的次数
            densities.append(n_break / len(t48))
        breaks[grp] = {'n_docs': len(densities), 'break_density': float(np.mean(densities))}
    print(f'=== 产出四：思维断裂点密度（dim48 低激活区进入次数/段）===')
    print(f'人类 {breaks["human"]["break_density"]:.4f} vs AI {breaks["ai"]["break_density"]:.4f}')

    out = {'stats': stats_all, 'coupling': {k: {'mean': float(np.mean(v)), 'n': len(v)} for k, v in coup.items()},
           'manifold': manifold, 'breaks': breaks,
           'pca_explained': [round(float(x), 3) for x in pca.explained_variance_ratio_],
           'note': 'Hurst>0.5 长程记忆/反随机游走；断裂点=dim48 进入低激活区（指代链弱）次数'}
    (OUT / 'flow_analysis.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print('落盘 flow_analysis.json + figs/fig_flow_*.png ✓')


if __name__ == '__main__':
    main()
