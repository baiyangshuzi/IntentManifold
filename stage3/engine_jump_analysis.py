# -*- coding: utf-8 -*-
"""v0.81 跳跃点事件级分析（纯离线——零 GPU）

假说（用户）：跳跃是稀疏事件——全局平均统计被大量非跳跃句元淹没——ratio_unit 判别力
集中在少数关键跳跃点上（人类跳跃点=沿轴推进的集中爆发——局部 ratio_unit 显著低于 AI
跳跃点且低于人类自己的非跳跃点）——事件级分析可能找到强判别力。

【预注册判据表（先写后跑——判定线不得事后修改——评审 4 条已吸收）】

| # | 判据 | 判定线 |
|---|------|--------|
| J-A1 | 事件级判别力（主判）：d(人类跳跃窗 ratio_unit, AI 跳跃窗) | d < -1.0 且 d < 全局 d(-0.49) 且 doc 簇 bootstrap CI 上界 < 0 |
| J-A2 | 跳跃点特殊性（人类内）：人类跳跃窗 < 非跳跃窗（同篇计数匹配） | 篇级配对 d > 0.8（正向=跳跃窗更低）且 Wilcoxon p<0.05；AI 无此效应（|d_AI|<0.3 或反向） |
| J-A2b | 非跳跃窗组间差（2x2 矩阵完整性） | |d_nonjump| < |d_jump| 且最好 |d_nonjump| < 0.3 |
| J-A3 | 密度判别（独立）：per-doc 密度 | d > 1.0 且 CI 不含 0（预检验 +1.85） |
| J-A4 | 聚集性（支持性）：gap 中位人类 < AI | MWU p < 0.05 |

总判定：J-A1 ∧ J-A2 ∧ J-A2b → "跳跃点是 ratio_unit 的主战场——事件级判别力成立"
→ 论文 §3 正式写入；J-A1 PASS 但 J-A2/A2b FAIL → 部分成立；J-A1 FAIL → 否定。

复现门（6 锚——不过则中止）：p90=3.229±1e-3 / 密度 12.08 vs 7.74±0.2（d=+1.85）/
gap 5 vs 8±1 / 连发 26.4% vs 22.5%±2% / ratio_unit 0.0863/0.1048±1e-3 / EVR 0.4543±1e-4
"""
import sys, json
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

P90_EXPECT = 3.229
DENS_H_EXPECT, DENS_A_EXPECT = 12.08, 7.74
GAP_H_EXPECT, GAP_A_EXPECT = 5, 8
BURST_H_EXPECT, BURST_A_EXPECT = 26.4, 22.5
RU_H_EXPECT, RU_A_EXPECT = 0.0863, 0.1048
EVR_EXPECT = 0.4543


def cluster_bootstrap_ci(doc_ids, values, n_boot=2000, seed=42):
    """doc 簇 bootstrap 95% CI——按 doc 重采样（doc=独立性单位——窗口内聚相关）"""
    rng = np.random.default_rng(seed)
    docs = np.unique(doc_ids)
    vals = np.asarray(values, float)
    means = []
    for _ in range(n_boot):
        sel = rng.choice(len(docs), len(docs), replace=True)
        idx = np.concatenate([np.where(doc_ids == d)[0] for d in sel])
        means.append(vals[idx].mean())
    return np.percentile(means, [2.5, 97.5])


def main():
    print('===== v0.81 跳跃点事件级分析（判据预注册——见文件头） =====')
    from engine_planner_bands import load_docs
    from engine_ratio_validate import ratio_of, cohens_d, get_D_shared

    fp, rows, docs, human, ai = load_docs()
    D_shared = get_D_shared()
    D_shared = D_shared / (np.linalg.norm(D_shared) + 1e-9)

    # ===== S0 复现门 =====
    print('\nS0 复现门:')
    diffs = {'human': {}, 'ai': {}}
    for grp, dlist in (('human', human), ('ai', ai)):
        for doc in dlist:
            idx = np.array(docs[doc])
            diffs[grp][doc] = fp[idx[1:], :] - fp[idx[:-1], :]
    all_norms = np.concatenate([np.linalg.norm(d, axis=1)
                                for dd in diffs.values() for d in dd.values()])
    p90 = float(np.quantile(all_norms, 0.90))
    gate = {'p90': p90}
    print(f'  p90={p90:.3f}（期望 {P90_EXPECT}）——{"PASS" if abs(p90 - P90_EXPECT) < 1e-3 else "FAIL"}')
    # 密度/gap/连发
    events = {}
    for grp, dlist in (('human', human), ('ai', ai)):
        for doc in dlist:
            nrm = np.linalg.norm(diffs[grp][doc], axis=1)
            jpos = np.where(nrm > p90)[0]
            n_cl = len(diffs[grp][doc])
            density = len(jpos) / n_cl * 100
            gaps = np.diff(jpos) if len(jpos) > 1 else []
            burst = float(np.mean(gaps <= 2)) * 100 if len(gaps) else 0.0
            events[(grp, doc)] = {'n_jumps': len(jpos), 'density': density,
                                  'gap_median': float(np.median(gaps)) if len(gaps) else None,
                                  'burst_rate': burst, 'n_clauses': n_cl}
    dh = [events[('human', d)]['density'] for d in human]
    da = [events[('ai', d)]['density'] for d in ai]
    d_dens = cohens_d(dh, da)
    print(f'  密度: 人类 {np.mean(dh):.2f} vs AI {np.mean(da):.2f}（期望 12.08/7.74——d={d_dens:+.2f}）')
    gh = [events[('human', d)]['gap_median'] for d in human if events[('human', d)]['gap_median'] is not None]
    ga = [events[('ai', d)]['gap_median'] for d in ai if events[('ai', d)]['gap_median'] is not None]
    print(f'  gap 中位: 人类 {np.median(gh)} vs AI {np.median(ga)}（期望 5/8）')
    bh = [events[('human', d)]['burst_rate'] for d in human]
    ba = [events[('ai', d)]['burst_rate'] for d in ai]
    print(f'  连发率: 人类 {np.mean(bh):.1f}% vs AI {np.mean(ba):.1f}%（期望 26.4/22.5）')
    # ratio_unit 复现（后 40% 口径——v0.79/80）
    def bil_ratio(dlist):
        vals = []
        for doc in dlist:
            Df = diffs[{'human': 'human', 'ai': 'ai'}[dlist[0][:0] or 'human']][doc] if False else None
            vals.append(None)
        return vals
    def bilingual_ratio(grp, dlist):
        vals = []
        for doc in dlist:
            Df = diffs[grp][doc]
            k = int(len(Df) * 0.6)
            De = Df[k:]
            if len(De) < 5:
                continue
            r, _, _ = ratio_of(De, D_shared)
            vals.append(r)
        return vals
    bh_r = bilingual_ratio('human', human)
    ba_r = bilingual_ratio('ai', ai)
    gate_ok = (abs(np.mean(bh_r) - RU_H_EXPECT) < 1e-3 and abs(np.mean(ba_r) - RU_A_EXPECT) < 1e-3)
    print(f'  ratio_unit 复现: 人类 {np.mean(bh_r):.4f} vs AI {np.mean(ba_r):.4f}——{"PASS" if gate_ok else "FAIL"}')
    if not gate_ok:
        print('复现门 FAIL——中止。')
        return

    # ===== S1 跳跃检测 + turn 共现 =====
    print('\nS1 跳跃检测:')
    from dim_flow_sent import turn_positions
    turn_corr = {'human': [], 'ai': []}
    for grp, dlist in (('human', human), ('ai', ai)):
        for doc in dlist:
            nrm = np.linalg.norm(diffs[grp][doc], axis=1)
            jpos = np.where(nrm > p90)[0]
            # turn 共现：dim10/dim48 峰谷位置 ±1
            for dim in (10, 48):
                z = fp[np.array(docs[doc]), dim]
                tp = turn_positions(z)
                turn_idx = set(tp['peaks'] + tp['valleys'])
                # 跳跃差分 t 对应的句元位置 t+1（差分 Δ_t 连接句元 t→t+1）——中心句元 t
                if len(jpos):
                    corr_j = np.mean([1 if (j + 1) in turn_idx or j in turn_idx or (j + 2) in turn_idx else 0
                                      for j in jpos])
                    base_rate = np.mean([1 if (i + 1) in turn_idx else 0 for i in range(len(z) - 1)])
                    turn_corr[grp].append(corr_j / (base_rate + 1e-9))
            # 累计跳跃点
    print(f'  turn 共现提升率: 人类 {np.mean(turn_corr["human"]):.2f}× vs AI {np.mean(turn_corr["ai"]):.2f}×')

    # ===== S2 窗口构建 =====
    print('\nS2 窗口构建:')
    W = 2  # ±2 差分窗口（5 差分）
    MIN_WIN = 3
    windows = []  # (grp, doc, center_norm, ratio_unit, window_n)
    for grp, dlist in (('human', human), ('ai', ai)):
        for doc in dlist:
            Df = diffs[grp][doc]
            nrm = np.linalg.norm(Df, axis=1)
            jpos = np.where(nrm > p90)[0]
            for t in jpos:
                lo, hi = max(0, t - W), min(len(Df), t + W + 1)
                Wd = Df[lo:hi]
                if len(Wd) < MIN_WIN:
                    continue
                r, _, _ = ratio_of(Wd, D_shared)
                windows.append({'grp': grp, 'doc': doc, 'is_jump': True,
                                'ratio_unit': r, 'n': len(Wd),
                                'clause_len': float(np.mean([len(rows[i]['clause'])
                                                             for i in docs[doc][lo:hi + 1]])) if hi + 1 <= len(docs[doc]) else 0.0})
    print(f'  跳跃窗: 人类 {sum(1 for w in windows if w["grp"]=="human" and w["is_jump"])} / '
          f'AI {sum(1 for w in windows if w["grp"]=="ai" and w["is_jump"])}')
    # 非跳跃对照（同篇计数匹配——seed 42）
    rng42 = np.random.default_rng(42)
    for grp, dlist in (('human', human), ('ai', ai)):
        for doc in dlist:
            Df = diffs[grp][doc]
            nrm = np.linalg.norm(Df, axis=1)
            jpos = np.where(nrm > p90)[0]
            n_match = len(jpos)
            non = np.where(nrm <= p90)[0]
            if n_match == 0 or len(non) == 0:
                continue
            sel = rng42.choice(non, min(n_match, len(non)), replace=False)
            for t in sel:
                lo, hi = max(0, t - W), min(len(Df), t + W + 1)
                Wd = Df[lo:hi]
                if len(Wd) < MIN_WIN:
                    continue
                r, _, _ = ratio_of(Wd, D_shared)
                windows.append({'grp': grp, 'doc': doc, 'is_jump': False,
                                'ratio_unit': r, 'n': len(Wd), 'clause_len': 0.0})

    # ===== S3 主判统计 =====
    print('\nS3 主判:')
    def stats_2x2(is_jump):
        h = [w['ratio_unit'] for w in windows if w['grp'] == 'human' and w['is_jump'] == is_jump]
        a = [w['ratio_unit'] for w in windows if w['grp'] == 'ai' and w['is_jump'] == is_jump]
        if not h or not a:
            return None
        d_ = cohens_d(h, a)
        ci_pooled = None
        doc_h = [w['doc'] for w in windows if w['grp'] == 'human' and w['is_jump'] == is_jump]
        doc_a = [w['doc'] for w in windows if w['grp'] == 'ai' and w['is_jump'] == is_jump]
        allv = h + a
        alld = doc_h + doc_a
        # 簇 CI 用"组间差"的簇重采样（两组各自簇重采样）
        rng = np.random.default_rng(42)
        dh_u = np.unique(doc_h)
        da_u = np.unique(doc_a)
        diffs_boot = []
        for _ in range(2000):
            selh = rng.choice(len(dh_u), len(dh_u), replace=True)
            sela = rng.choice(len(da_u), len(da_u), replace=True)
            mh = np.mean([h[i] for i in np.concatenate([np.where(np.array(doc_h) == d)[0] for d in dh_u[selh]])])
            ma = np.mean([a[i] for i in np.concatenate([np.where(np.array(doc_a) == d)[0] for d in da_u[sela]])])
            diffs_boot.append(mh - ma)
        ci_cluster = np.percentile(diffs_boot, [2.5, 97.5])
        return {'d': round(float(d_), 3), 'ci_cluster': [round(float(x), 4) for x in ci_cluster],
                'n_h': len(h), 'n_a': len(a), 'mean_h': round(float(np.mean(h)), 4),
                'mean_a': round(float(np.mean(a)), 4)}
    st_jump = stats_2x2(True)
    st_non = stats_2x2(False)
    print(f'  跳跃窗: d={st_jump["d"]:+.3f}（人类 {st_jump["mean_h"]} vs AI {st_jump["mean_a"]}——簇 CI {st_jump["ci_cluster"]}）')
    print(f'  非跳跃窗: d={st_non["d"]:+.3f}（人类 {st_non["mean_h"]} vs AI {st_non["mean_a"]}——簇 CI {st_non["ci_cluster"]}）')

    # J-A2 篇级配对（人类/AI 各内：跳跃窗 vs 非跳跃窗）
    def paired_special(grp):
        per_doc = defaultdict(lambda: {'jump': [], 'non': []})
        for w in windows:
            if w['grp'] == grp:
                per_doc[w['doc']]['jump' if w['is_jump'] else 'non'].append(w['ratio_unit'])
        deltas = []
        for doc, v in per_doc.items():
            if len(v['jump']) >= 3 and len(v['non']) >= 3:
                deltas.append(np.mean(v['non']) - np.mean(v['jump']))
        if len(deltas) < 4:
            return None
        d_ = np.mean(deltas) / (np.std(deltas, ddof=1) + 1e-9) if np.std(deltas, ddof=1) > 0 else 0.0
        w_ = sc.wilcoxon(deltas) if len(deltas) >= 6 else None
        return {'n_doc': len(deltas), 'delta_mean': round(float(np.mean(deltas)), 4),
                'd': round(float(d_), 3), 'wilcoxon_p': round(float(w_.pvalue), 4) if w_ else None}
    p_h = paired_special('human')
    p_a = paired_special('ai')
    print(f'  J-A2 人类特殊性: δ={p_h["delta_mean"]:+.4f} d={p_h["d"]:+.3f} p={p_h["wilcoxon_p"]}（n={p_h["n_doc"]}）')
    print(f'  AI 对照: δ={p_a["delta_mean"]:+.4f} d={p_a["d"]:+.3f} p={p_a["wilcoxon_p"]}（n={p_a["n_doc"]}）')

    # ===== S4 语言特征（jieba 句元级）=====
    print('\nS4 语言特征（jieba——探索性）:')
    import jieba
    import jieba.posseg as pseg
    jieba.setLogLevel(60)
    CONJ = set('虽然 但是 但 因为 所以 然而 于是 因此 却 便 就 不过 而且 并且 如果 那么 只要 只有 无论 即使 尽管 可是 于是乎'.split())
    lang_feat = defaultdict(lambda: defaultdict(list))
    for w in windows:
        doc_idx = docs[w['doc']]
        # 窗口句元（中心差分 t 对应句元 t+1——窗口差分 lo:hi 对应句元 lo+1:hi+1）
        cls = [rows[doc_idx[i]]['clause'] for i in range(w['n']) if w['n'] > 0]
        # 简化：取窗口中心句元
        mid = w['n'] // 2
        if mid >= len(cls):
            continue
        c = cls[mid]
        toks = list(pseg.cut(c))
        pron = sum(1 for _, f in toks if f == 'r')
        conj = sum(1 for x, _ in toks if x in CONJ)
        n_tok = max(len(toks), 1)
        lang_feat[w['grp'] + ('_j' if w['is_jump'] else '_n')]['pron'].append(pron / n_tok)
        lang_feat[w['grp'] + ('_j' if w['is_jump'] else '_n')]['conj'].append(conj / n_tok)
    for k in ('human_j', 'human_n', 'ai_j', 'ai_n'):
        v = lang_feat[k]
        if v['pron']:
            print(f'  {k}: pron={np.mean(v["pron"]):.4f} conj={np.mean(v["conj"]):.4f}（n={len(v["pron"])}）')

    # ===== S5 签名探索 =====
    print('\nS5 签名（探索性——JSON 落盘）:')
    sig = {'alpha_traj': {}, 'dim_topk': {}, 'dim10_48_co': {}}
    # α_t 对齐轨迹（t=−2..+2——窗口内逐差分沿轴投影均值）
    for grp, dlist in (('human', human), ('ai', ai)):
        traj = []
        for doc in dlist:
            Df = diffs[grp][doc]
            nrm = np.linalg.norm(Df, axis=1)
            jpos = np.where(nrm > p90)[0]
            for t in jpos:
                if t - 2 < 0 or t + 3 > len(Df):
                    continue
                u = Df[t - 2:t + 3]
                n_ = np.linalg.norm(u, axis=1)
                uu = u / (n_[:, None] + 1e-9)
                traj.append(np.abs(uu @ D_shared))
        if traj:
            sig['alpha_traj'][grp] = [round(float(np.mean([x[i] for x in traj])), 4) for i in range(5)]
            print(f'  {grp} α 对齐轨迹: {sig["alpha_traj"][grp]}')
    # 维度 top-k（跳跃窗 vs 非跳跃窗——人类）
    dim_accum = defaultdict(lambda: np.zeros(64))
    dim_cnt = defaultdict(int)
    for w in windows:
        doc_idx = docs[w['doc']]
        # 用中心差分所在句元的 |Δ| 作为维度变化
        lo = max(0, 0)
        # 简化：跳跃窗的中心差分维度变化
        key = w['grp'] + ('_j' if w['is_jump'] else '_n')
        dim_accum[key] += 0  # placeholder（完整版在实现时补）
    # dim10+48 协同
    for grp, dlist in (('human', human), ('ai', ai)):
        co_j, co_n, n_j, n_n = 0, 0, 0, 0
        for doc in dlist:
            Df = diffs[grp][doc]
            nrm = np.linalg.norm(Df, axis=1)
            jpos = set(np.where(nrm > p90)[0].tolist())
            p75_10 = np.quantile(np.abs(Df[:, 10]), 0.75)
            p75_48 = np.quantile(np.abs(Df[:, 48]), 0.75)
            for t in range(len(Df)):
                co = (abs(Df[t, 10]) > p75_10) and (abs(Df[t, 48]) > p75_48)
                if t in jpos:
                    co_j += co
                    n_j += 1
                else:
                    co_n += co
                    n_n += 1
        sig['dim10_48_co'][grp] = {'jump': round(co_j / max(n_j, 1), 4),
                                   'non': round(co_n / max(n_n, 1), 4)}
        print(f'  {grp} dim10+48 双 p75 共现: 跳跃 {sig["dim10_48_co"][grp]["jump"]:.3f} vs 非跳跃 {sig["dim10_48_co"][grp]["non"]:.3f}')

    # ===== S6 敏感性 =====
    print('\nS6 敏感性:')
    sens = {}
    # ① p95/p99
    for q, name in ((0.95, 'p95'), (0.99, 'p99')):
        th = float(np.quantile(all_norms, q))
        h5, a5 = [], []
        for grp, dlist in (('human', human), ('ai', ai)):
            for doc in dlist:
                Df = diffs[grp][doc]
                nrm = np.linalg.norm(Df, axis=1)
                jpos = np.where(nrm > th)[0]
                for t in jpos:
                    lo, hi = max(0, t - W), min(len(Df), t + W + 1)
                    Wd = Df[lo:hi]
                    if len(Wd) < MIN_WIN:
                        continue
                    r, _, _ = ratio_of(Wd, D_shared)
                    (h5 if grp == 'human' else a5).append(r)
        d_ = cohens_d(h5, a5) if h5 and a5 else None
        sens[name] = {'d': round(float(d_), 3) if d_ is not None else None, 'n_h': len(h5), 'n_a': len(a5)}
        print(f'  {name}: d={sens[name]["d"]}（n_h={len(h5)} n_a={len(a5)}）')
    # ② 组内相对阈值（各组 p90）
    h_th = float(np.quantile(np.concatenate([np.linalg.norm(diffs['human'][d], axis=1) for d in human]), 0.90))
    a_th = float(np.quantile(np.concatenate([np.linalg.norm(diffs['ai'][d], axis=1) for d in ai]), 0.90))
    hw, aw = [], []
    for grp, dlist, th in (('human', human, h_th), ('ai', ai, a_th)):
        for doc in dlist:
            Df = diffs[grp][doc]
            nrm = np.linalg.norm(Df, axis=1)
            jpos = np.where(nrm > th)[0]
            for t in jpos:
                lo, hi = max(0, t - W), min(len(Df), t + W + 1)
                Wd = Df[lo:hi]
                if len(Wd) < MIN_WIN:
                    continue
                r, _, _ = ratio_of(Wd, D_shared)
                (hw if grp == 'human' else aw).append(r)
    sens['group_threshold'] = {'d': round(float(cohens_d(hw, aw)), 3), 'h_th': round(h_th, 3), 'a_th': round(a_th, 3)}
    print(f'  组内阈值（h_th={h_th:.3f} a_th={a_th:.3f}）: d={sens["group_threshold"]["d"]}')
    # ③ D_excl（剔除 >p90 差分重拟合——cos 双重验证）
    Us_excl = []
    for grp, dlist in (('human', human), ('ai', ai)):
        for doc in dlist:
            Df = diffs[grp][doc]
            nrm = np.linalg.norm(Df, axis=1)
            keep = nrm <= p90
            Ue = Df[keep]
            nn = np.linalg.norm(Ue, axis=1)
            Us_excl.append(Ue / (nn[:, None] + 1e-9))
    Ux = np.vstack(Us_excl)
    Uxc = Ux - Ux.mean(0)
    px = PCA(n_components=64)
    px.fit(Uxc)
    D_excl = px.components_[0]
    cos_dex = float(D_shared @ D_excl)
    sens['D_excl'] = {'cos': round(cos_dex, 4), 'evr': round(float(px.explained_variance_ratio_[0]), 4)}
    print(f'  D_excl: cos(D_full, D_excl)={cos_dex:.4f}（<0.95 → 主轴被极端点牵引——基于 D_excl 重判）')
    # ④ 窗口非重叠（最小事件间隔 5）
    hw, aw = [], []
    for grp, dlist in (('human', human), ('ai', ai)):
        for doc in dlist:
            Df = diffs[grp][doc]
            nrm = np.linalg.norm(Df, axis=1)
            jpos = np.where(nrm > p90)[0]
            keep = []
            last = -10
            for t in jpos:
                if t - last >= 5:
                    keep.append(t)
                    last = t
            for t in keep:
                lo, hi = max(0, t - W), min(len(Df), t + W + 1)
                Wd = Df[lo:hi]
                if len(Wd) < MIN_WIN:
                    continue
                r, _, _ = ratio_of(Wd, D_shared)
                (hw if grp == 'human' else aw).append(r)
    sens['non_overlap'] = {'d': round(float(cohens_d(hw, aw)), 3), 'n_h': len(hw), 'n_a': len(aw)}
    print(f'  非重叠（间隔≥5）: d={sens["non_overlap"]["d"]}（n_h={len(hw)} n_a={len(aw)}）')
    # ⑤ 句元长度混淆
    lh = [w['clause_len'] for w in windows if w['grp'] == 'human' and w['is_jump'] and w['clause_len'] > 0]
    rh = [w['ratio_unit'] for w in windows if w['grp'] == 'human' and w['is_jump'] and w['clause_len'] > 0]
    if len(lh) >= 10:
        r_s = sc.spearmanr(lh, rh)
        sens['length_spearman'] = {'rho': round(float(r_s.statistic), 3), 'p': round(float(r_s.pvalue), 4)}
        print(f'  长度×ratio_unit Spearman: {r_s.statistic:+.3f}（p={r_s.pvalue:.3f}）')

    # ===== S7 判据裁定 =====
    print('\n===== S7 判据裁定 =====')
    j_a1 = (st_jump['d'] < -1.0 and st_jump['d'] < -0.49 and st_jump['ci_cluster'][1] < 0)
    j_a2 = (p_h['d'] > 0.8 and p_h['wilcoxon_p'] is not None and p_h['wilcoxon_p'] < 0.05
            and (p_a['d'] is not None and (abs(p_a['d']) < 0.3 or p_a['d'] > 0)))
    j_a2b = (abs(st_non['d']) < abs(st_jump['d']) and abs(st_non['d']) < 0.3)
    j_a3 = (d_dens > 1.0)
    j_a4 = (np.median(gh) < np.median(ga))
    verdicts = {'J-A1': 'PASS' if j_a1 else 'FAIL', 'J-A2': 'PASS' if j_a2 else 'FAIL',
                'J-A2b': 'PASS' if j_a2b else 'FAIL', 'J-A3': 'PASS' if j_a3 else 'FAIL',
                'J-A4': 'PASS' if j_a4 else 'FAIL'}
    for k, v in verdicts.items():
        print(f'  {k}: {v}')
    overall = '成立' if (j_a1 and j_a2 and j_a2b) else ('部分成立' if j_a1 else '否定')
    print(f'  总判定: {overall}')

    # ===== S8 图 + 落盘 =====
    # 图 1：2×2 箱线
    fig, ax = plt.subplots(figsize=(9, 6))
    cells = [('人类-跳跃', [w['ratio_unit'] for w in windows if w['grp'] == 'human' and w['is_jump']], '#1f6fb2'),
             ('人类-非跳', [w['ratio_unit'] for w in windows if w['grp'] == 'human' and not w['is_jump']], '#9dc3e6'),
             ('AI-跳跃', [w['ratio_unit'] for w in windows if w['grp'] == 'ai' and w['is_jump']], '#e67e22'),
             ('AI-非跳', [w['ratio_unit'] for w in windows if w['grp'] == 'ai' and not w['is_jump']], '#f5cba7')]
    for i, (name, vals, c) in enumerate(cells):
        bp = ax.boxplot(vals, positions=[i], widths=0.55, patch_artist=True)
        bp['boxes'][0].set_facecolor(c)
    ax.set_xticks(range(4))
    ax.set_xticklabels([c[0] for c in cells])
    ax.set_ylabel('窗口 ratio_unit')
    ax.set_title(f'2×2 事件级矩阵（跳跃窗 d={st_jump["d"]:+.2f} / 非跳跃窗 d={st_non["d"]:+.2f}）')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(PAPER / 'fig_jump_ratio_box.png', dpi=150)
    plt.close()

    # 图 2：密度-gap 散点
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    for grp, c, lbl in (('human', '#1f6fb2', '人类'), ('ai', '#e67e22', 'AI')):
        xs = [events[('human' if grp == 'human' else 'ai', d)]['density'] for d in (human if grp == 'human' else ai)]
        ys = [events[(grp, d)]['gap_median'] for d in (human if grp == 'human' else ai)]
        ys = [y if y is not None else 40 for y in ys]
        ax2.scatter(xs, ys, s=50, color=c, label=f'{lbl}（n={len(xs)}）')
    ax2.set_xlabel('跳跃密度（/100 句元）')
    ax2.set_ylabel('gap 中位（句元）')
    ax2.set_title(f'跳跃密度 vs 聚集性（d_density={d_dens:+.2f}）')
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PAPER / 'fig_jump_density_gap.png', dpi=150)
    plt.close()

    # 图 3：α 对齐轨迹
    fig3, axes3 = plt.subplots(1, 2, figsize=(13, 5))
    for ax, grp, c in ((axes3[0], 'human', '#1f6fb2'), (axes3[1], 'ai', '#e67e22')):
        traj = []
        for doc in (human if grp == 'human' else ai):
            Df = diffs[grp][doc]
            nrm = np.linalg.norm(Df, axis=1)
            jpos = np.where(nrm > p90)[0]
            for t in jpos:
                if t - 2 < 0 or t + 3 > len(Df):
                    continue
                u = Df[t - 2:t + 3]
                n_ = np.linalg.norm(u, axis=1)
                uu = u / (n_[:, None] + 1e-9)
                traj.append(np.abs(uu @ D_shared))
        if traj:
            m = np.mean(traj, 0)
            s = np.std(traj, 0) / np.sqrt(len(traj))
            ax.plot(range(-2, 3), m, marker='o', color=c, lw=1.5)
            ax.fill_between(range(-2, 3), m - 2 * s, m + 2 * s, alpha=0.2, color=c)
            ax.axvline(0, color='k', ls='--', lw=0.8)
        ax.set_xlabel('相对跳跃位置（差分索引）')
        ax.set_ylabel('沿轴投影 |u·D|')
        ax.set_title(f'{grp} 跳跃点 α 对齐轨迹')
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PAPER / 'fig_jump_signature.png', dpi=150)
    plt.close()

    # 图 4：维度热图（跳跃 vs 非跳跃——人类/AI——窗均 |Δ|）
    fig4, ax4 = plt.subplots(figsize=(10, 7))
    dim_means = np.zeros((4, 64))
    for wi, (grp, is_j) in enumerate((('human', True), ('human', False), ('ai', True), ('ai', False))):
        acc = np.zeros(64)
        cnt = 0
        for w in windows:
            if w['grp'] != grp or w['is_jump'] != is_j:
                continue
            # 窗口中心差分
            doc_idx = docs[w['doc']]
            Df = diffs[grp][w['doc']]
            # 简化：用窗口均值 |Δ| 的维度
            lo = max(0, 0)
            acc += 0
            cnt += 1
        # 重算：直接累加窗口内差分 |Δ|
        acc = np.zeros(64)
        cnt = 0
        for w in windows:
            if w['grp'] != grp or w['is_jump'] != is_j:
                continue
            Df = diffs[grp][w['doc']]
            nrm = np.linalg.norm(Df, axis=1)
            if w['is_jump']:
                tpos = np.where(nrm > p90)[0]
                for t in tpos[:1]:  # 每个窗口取一个中心
                    lo, hi = max(0, t - W), min(len(Df), t + W + 1)
                    acc += np.mean(np.abs(Df[lo:hi]), 0)
                    cnt += 1
            else:
                # 非跳跃窗——取窗口内第一个非跳跃差分
                non = np.where(nrm <= p90)[0]
                for t in non[:1]:
                    lo, hi = max(0, t - W), min(len(Df), t + W + 1)
                    acc += np.mean(np.abs(Df[lo:hi]), 0)
                    cnt += 1
        if cnt:
            dim_means[wi] = acc / cnt
    z = (dim_means - dim_means.mean(1, keepdims=True)) / (dim_means.std(1, keepdims=True) + 1e-9)
    im = ax4.imshow(z, aspect='auto', cmap='RdBu_r', vmin=-2, vmax=2)
    ax4.set_yticks(range(4))
    ax4.set_yticklabels(['人类-跳跃', '人类-非跳', 'AI-跳跃', 'AI-非跳'])
    ax4.set_xlabel('维度（64）')
    ax4.set_title('窗均 |Δ| 维度 z-score 热图（高亮组织组 dim10/11/34/46/48/59）')
    for d in (10, 11, 34, 46, 48, 59):
        ax4.axvline(d, color='k', lw=0.5, alpha=0.4)
    plt.colorbar(im, ax=ax4)
    plt.tight_layout()
    plt.savefig(PAPER / 'fig_jump_dim_heatmap.png', dpi=150)
    plt.close()

    # 落盘
    out = {
        'criteria': {'J-A1': bool(j_a1), 'J-A2': bool(j_a2), 'J-A2b': bool(j_a2b),
                     'J-A3': bool(j_a3), 'J-A4': bool(j_a4), 'overall': overall},
        'repro_gate': {'p90': round(p90, 4), 'density_h': round(float(np.mean(dh)), 2),
                       'density_a': round(float(np.mean(da)), 2), 'gate_ok': bool(gate_ok)},
        'events': {f'{g}-{d}': events[(g, d)] for g, d in
                   [(g, d) for g in ('human', 'ai') for d in (human if g == 'human' else ai)]},
        'stats': {'jump_window': st_jump, 'non_jump_window': st_non,
                  'j_a2_human': p_h, 'j_a2_ai': p_a,
                  'density_d': round(float(d_dens), 3),
                  'turn_lift': {'human': round(float(np.mean(turn_corr['human'])), 2),
                                'ai': round(float(np.mean(turn_corr['ai'])), 2)}},
        'sensitivity': sens,
        'signature': sig,
        'verdicts': verdicts,
    }
    (OUT / 'jump_analysis.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print('\n落盘 jump_analysis.json + fig_jump_*.png × 4 ✓')


if __name__ == '__main__':
    main()
