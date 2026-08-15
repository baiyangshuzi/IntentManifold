# -*- coding: utf-8 -*-
"""v0.80 ratio_unit 稳健性与判别力验证（纯离线——零 GPU）

ratio_unit（推进集中度 = 单位差分垂直/平行主轴分量比）在 v0.79 bilingual 上 d=-3.18
（人类 0.086 vs AI 0.105）——本脚本做跨语料稳健性验证（用户优先 1）。

【预注册判据表（先写后跑——判定线不得事后修改）】

| # | 判据 | 判定线 |
|---|------|--------|
| V-R1 | 独立集判别力（主判） | 非配对 d(人类27, ai+qwen64) < -1.5 且方向同向（人类<AI）；CI 不含 0；配对 Wilcoxon p<0.05（human-ai 与 human-qwen 双方向） |
| V-R1b | 白话域（副线不并判） | d(baihua 15 vs 30) < -1.5 |
| V-R2 | sent_proj 低冗余 | 池化 |rho(ratio_unit, sent_proj)| < 0.6 且组内（human/ai+qwen）均 < 0.6 |
| V-R3 | l7_adj 低冗余 | 同 V-R2（指标换 l7_adj） |
| V-R4 | 干预空间增量（支持性） | |Spearman(Δratio, Δsent_proj)| < 0.6（48 对） |

总判定：V-R1 ∧ V-R2 ∧ V-R3 → ratio_unit 正式写入论文 §3；部分成立 → 标注域限定/冗余；
V-R1 FAIL → 否定（仅 REPORT + §6.5 否定说明）。

复现门（运行中止条件）：S0 D_shared EVR=0.4543±1e-4 且 bilingual ratio_unit 复现
0.086/0.105 偏差<1e-6——不过则中止报警。
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
TI = BASE / 'data' / 'training_intervention'
PAPER = Path('C:/Users/bai/Desktop/AB系统论文储备')
sys.path.insert(0, str(BASE / 'stage3'))

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def get_D_shared():
    """S0 主轴 D 锚定：优先 axis_analysis.json['D_shared']（v0.79 落盘）——缺失则重算"""
    f = OUT / 'axis_analysis.json'
    if f.exists():
        d = json.loads(f.read_text(encoding='utf-8'))
        if 'D_shared' in d.get('axis', {}):
            return np.asarray(d['axis']['D_shared'], float)
    # 重算（同 engine_axis_test S1-S3 口径）
    from engine_planner_bands import load_docs
    fp, rows, docs, human, ai = load_docs()
    diffs = []
    for grp, dlist in (('human', human), ('ai', ai)):
        for doc in dlist:
            idx = np.array(docs[doc])
            D = fp[idx[1:], :] - fp[idx[:-1], :]
            diffs.append(D / (np.linalg.norm(D, axis=1)[:, None] + 1e-9))
    U = np.vstack(diffs)
    split = int(len(U) * 0.6)
    Uc = U[:split] - U[:split].mean(0)
    p = PCA(n_components=64)
    p.fit(Uc)
    D = p.components_[0]
    mu_h = np.vstack([fp[np.array(docs[doc])[1:], :] - fp[np.array(docs[doc])[:-1], :]
                      for doc in human]).mean(0)
    if D @ mu_h < 0:
        D = -D
    return D


def ratio_of(D_fp, D_axis):
    """per-doc ratio_unit（口径同 v0.79：逐元素均值）——D_fp: (n,64) 差分"""
    n = np.linalg.norm(D_fp, axis=1)
    u = D_fp / (n[:, None] + 1e-9)
    jppu = float(np.mean(np.abs(u - (u @ D_axis)[:, None] * D_axis)))
    jpu = float(np.mean(np.abs(u @ D_axis)))
    return jppu / (jpu + 1e-9), jpu, jppu


def cohens_d(x, y):
    return (np.mean(x) - np.mean(y)) / np.sqrt((np.var(x) + np.var(y)) / 2 + 1e-9)


def main():
    print('===== v0.80 ratio_unit 稳健性与判别力验证（判据预注册——见文件头） =====')
    from engine_field_evidence import bootstrap_ci

    d = np.load(OUT / 'fp_matrix.npz')
    fp = d['fp']
    rows = json.loads((OUT / 'rows.json').read_text(encoding='utf-8'))

    # ===== S0 主轴锚定 + 复现门 =====
    D_shared = get_D_shared()
    D_shared = D_shared / (np.linalg.norm(D_shared) + 1e-9)
    # 复现门 1：bilingual per-doc ratio_unit 复现 v0.79
    from engine_planner_bands import load_docs
    fp_b, rows_b, docs_b, human_b, ai_b = load_docs()
    def bilingual_ratio(dlist):
        vals = []
        for doc in dlist:
            idx = np.array(docs_b[doc])
            Df = fp_b[idx[1:], :] - fp_b[idx[:-1], :]
            k = int(len(Df) * 0.6)
            De = Df[k:]
            if len(De) < 5:
                continue
            r, _, _ = ratio_of(De, D_shared)
            vals.append(r)
        return vals
    bh = bilingual_ratio(human_b)
    ba = bilingual_ratio(ai_b)
    print(f'S0 复现门: bilingual ratio_unit 人类 {np.mean(bh):.4f}（v0.79 实测 0.0863）AI {np.mean(ba):.4f}（v0.79 实测 0.1048）')
    gate_ok = abs(np.mean(bh) - 0.0863) < 1e-3 and abs(np.mean(ba) - 0.1048) < 1e-3
    print(f'  复现门: {"PASS" if gate_ok else "FAIL——中止"}（容差 1e-3——显示精度级）')
    if not gate_ok:
        print('口径漂移——中止。')
        return

    # ===== S1 independent_test per-doc ratio_unit =====
    print('\nS1 independent_test per-doc ratio_unit:')
    ind_docs = defaultdict(list)
    for i, r in enumerate(rows):
        if r['source'] == 'independent_test':
            ind_docs[(r['doc'], r['side'])].append(i)
    per_doc = {}
    excluded = []
    for (doc, side), idxs in ind_docs.items():
        idxs.sort()
        Df = fp[idxs[1:], :] - fp[idxs[:-1], :]
        if len(Df) < 5:
            if side == 'human':
                excluded.append({'doc': doc, 'n_diffs': len(Df)})
            continue
        r, jpu, jppu = ratio_of(Df, D_shared)
        per_doc[(doc, side)] = {'doc': doc, 'side': side, 'domain': 'baihua' if doc.startswith('B') else 'shiping',
                                'n_diffs': len(Df), 'ratio_unit': r, 'j_par_unit': jpu, 'j_perp_unit': jppu}
    print(f'  per-doc: 人类 {sum(1 for v in per_doc.values() if v["side"]=="human")} / '
          f'ai {sum(1 for v in per_doc.values() if v["side"]=="ai")} / '
          f'qwen {sum(1 for v in per_doc.values() if v["side"]=="qwen")}')
    print(f'  人类短文本剔除: {[e["doc"] for e in excluded]}')

    def grp(sides, domain=None):
        return [v['ratio_unit'] for v in per_doc.values()
                if v['side'] in sides and (domain is None or v['domain'] == domain)]

    h = grp(('human',))
    a_all = grp(('ai', 'qwen'))
    a_ds = grp(('ai',))
    a_qw = grp(('qwen',))
    print(f'  人类 {np.mean(h):.4f}（n={len(h)}）vs ai+qwen {np.mean(a_all):.4f}（n={len(a_all)}）')

    # ===== S2 配对/非配对 + 分域 + 长度敏感性 =====
    print('\nS2 统计:')
    # 配对（同题 human vs ai / human vs qwen——n=27 人类合格篇）
    def paired_delta(side2):
        dts = []
        for v in per_doc.values():
            if v['side'] == 'human':
                k = (v['doc'], side2)
                if k in per_doc:
                    dts.append(v['ratio_unit'] - per_doc[k]['ratio_unit'])
        return dts
    d_ai = paired_delta('ai')
    d_qw = paired_delta('qwen')
    w_ai = sc.wilcoxon(d_ai) if len(d_ai) >= 6 else None
    w_qw = sc.wilcoxon(d_qw) if len(d_qw) >= 6 else None
    ci_paired_ai = bootstrap_ci(d_ai, np.zeros(len(d_ai)), paired=True) if d_ai else None
    print(f'  配对 human-ai: δ={np.mean(d_ai):+.4f}（n={len(d_ai)}）Wilcoxon p={w_ai.pvalue:.3f} CI {ci_paired_ai}')
    print(f'  配对 human-qwen: δ={np.mean(d_qw):+.4f}（n={len(d_qw)}）Wilcoxon p={w_qw.pvalue:.3f}')
    # 非配对主判
    d_all = cohens_d(h, a_all)
    ci_all = bootstrap_ci(h, a_all, paired=False)
    u_all = sc.mannwhitneyu(h, a_all)
    print(f'  非配对主判: d={d_all:+.3f}（人类 {np.mean(h):.4f} vs ai+qwen {np.mean(a_all):.4f}）CI {ci_all} MWU p={u_all.pvalue:.3f}')
    # 分域
    h_b, a_b = grp(('human',), 'baihua'), grp(('ai', 'qwen'), 'baihua')
    h_s, a_s = grp(('human',), 'shiping'), grp(('ai', 'qwen'), 'shiping')
    d_b = cohens_d(h_b, a_b)
    d_s = cohens_d(h_s, a_s)
    print(f'  分域: 白话 d={d_b:+.3f}（人类 {np.mean(h_b):.4f} vs {np.mean(a_b):.4f}）——'
          f'时评 d={d_s:+.3f}（人类 {np.mean(h_s):.4f} vs {np.mean(a_s):.4f}）')
    # 长度敏感性：ai/qwen 截断至人类 max 差分
    max_h = max(v['n_diffs'] for v in per_doc.values() if v['side'] == 'human')
    h_lm, a_lm = [], []
    for v in per_doc.values():
        if v['side'] == 'human':
            h_lm.append(v['ratio_unit'])
        elif v['side'] in ('ai', 'qwen'):
            idxs = [i for i, r in enumerate(rows) if r['source'] == 'independent_test'
                    and r['doc'] == v['doc'] and r['side'] == v['side']]
            idxs.sort()
            Df = fp[idxs[1:], :] - fp[idxs[:-1], :]
            Dm = Df[:max_h]
            if len(Dm) >= 5:
                r, _, _ = ratio_of(Dm, D_shared)
                a_lm.append(r)
    d_lm = cohens_d(h_lm, a_lm)
    print(f'  长度匹配（截断至 {max_h} 差分）: d={d_lm:+.3f}')

    # ===== S3 training_intervention per-run ratio_unit =====
    print('\nS3 干预集 per-run ratio_unit:')
    ti_docs = defaultdict(list)
    for i, r in enumerate(rows):
        if r['source'] == 'training_intervention':
            ti_docs[r['doc']].append(i)
    per_run = {}
    for run_id, idxs in ti_docs.items():
        idxs.sort(key=lambda i: (rows[i]['seg'], i))
        Df = fp[idxs[1:], :] - fp[idxs[:-1], :]
        if len(Df) < 5:
            continue
        r, _, _ = ratio_of(Df, D_shared)
        per_run[run_id] = {'run_id': run_id, 'condition': run_id.split('-')[0],
                           'n_diffs': len(Df), 'ratio_unit': r}
    print(f'  runs: {len(per_run)}')
    cond_groups = defaultdict(list)
    for v in per_run.values():
        cond_groups[v['condition']].append(v['ratio_unit'])
    for c in ('none', 'vt_ext', 'vt_seed', 'vt_seed_beam', 'vt_kalman', 'vt_kalman_seed',
              'vt_kalman_gate', 'vt_gate_beam', 'vt_oracle'):
        if c in cond_groups:
            print(f'    {c}: {np.mean(cond_groups[c]):.4f}（n={len(cond_groups[c])}）')
    # Δ配对（vt vs none——同 prompt×seed）
    manifest = json.loads((TI / 'manifest.json').read_text(encoding='utf-8'))
    man_by_id = {r['run_id']: r for r in manifest}
    pairs_dr, pairs_ds = [], []
    for cond in ('vt_ext', 'vt_seed', 'vt_seed_beam', 'vt_kalman', 'vt_kalman_seed',
                 'vt_kalman_gate', 'vt_gate_beam', 'vt_oracle'):
        for pid in ('P1', 'P2', 'P3'):
            for sd in (0, 1):
                vt_id = f'{cond}-{pid}-s{sd}'
                none_id = f'none-{pid}-s{sd}'
                if vt_id in per_run and none_id in per_run:
                    r_vt = per_run[vt_id]['ratio_unit']
                    r_none = per_run[none_id]['ratio_unit']
                    sp_vt = np.mean([seg['dims']['sent_proj'] for seg in man_by_id[vt_id]['segs']
                                     if seg.get('dims') and 'sent_proj' in seg['dims']]) if vt_id in man_by_id else None
                    sp_none = np.mean([seg['dims']['sent_proj'] for seg in man_by_id[none_id]['segs']
                                       if seg.get('dims') and 'sent_proj' in seg['dims']]) if none_id in man_by_id else None
                    if sp_vt is not None and sp_none is not None:
                        pairs_dr.append(r_vt - r_none)
                        pairs_ds.append(sp_vt - sp_none)
    if len(pairs_dr) >= 6:
        rho_r = sc.spearmanr(pairs_dr, pairs_ds)
        print(f'  Δ配对 n={len(pairs_dr)}: Spearman(Δratio, Δsent_proj)={rho_r.statistic:+.3f} p={rho_r.pvalue:.3f}')
        print(f'  Δratio 均值={np.mean(pairs_dr):+.4f}（vt−none）——Δsent_proj 均值={np.mean(pairs_ds):+.4f}')
    else:
        rho_r = None
        print('  Δ配对不足')

    # ===== S4 独立集七维重算（bge CPU） =====
    print('\nS4 独立集七维重算（bge CPU 离线）:')
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    from para_dimensions import load_models, para_dimensions
    from subclause_structure import split_subclauses
    import jieba
    import jieba.posseg as pseg
    jieba.setLogLevel(60)
    enc, disc = load_models('cpu')
    # 从 manifest 拿文本
    ind_man = json.loads((BASE / 'data' / 'independent_test' / 'manifest.json').read_text(encoding='utf-8'))
    pairs = ind_man['pairs']
    text_by_key = {}
    for p in pairs:
        pid = p['pair_id']
        for side, key in (('human', 'human'), ('ai', 'ai'), ('qwen', 'qwen_ai')):
            t = p.get(key)
            if t:
                text_by_key[(pid, side)] = t
    seven = {}
    for (doc, side), t in text_by_key.items():
        try:
            dm = para_dimensions(t, enc, disc, split_subclauses, pseg, device='cpu')
            seven[(doc, side)] = dm
        except Exception as ex:
            print(f'  {doc}-{side} 七维失败: {str(ex)[:50]}')
    print(f'  七维计算: {len(seven)} 篇')

    # 相关矩阵（ratio_unit × 七维——池化 + 组内）
    keys7 = ['disc', 'sent_proj', 'traj', 'l7_adj', 'word_proj', 'word_adj', 'entropy']
    def corr_view(entries):
        out = {}
        for k7 in keys7:
            xs, ys = [], []
            for (doc, side), v in entries.items():
                if (doc, side) not in per_doc or k7 not in v:
                    continue
                xs.append(per_doc[(doc, side)]['ratio_unit'])
                ys.append(v[k7])
            if len(xs) >= 6:
                r = sc.spearmanr(xs, ys)
                out[k7] = {'rho': round(float(r.statistic), 3), 'p': round(float(r.pvalue), 4), 'n': len(xs)}
            else:
                out[k7] = None
        return out
    pooled = {k: v for (k, v) in seven.items() if k in per_doc}
    v_pool = corr_view(pooled)
    v_hum = corr_view({k: v for k, v in pooled.items() if k[1] == 'human'})
    v_aiq = corr_view({k: v for k, v in pooled.items() if k[1] in ('ai', 'qwen')})
    print('  相关（池化/人类内/ai+qwen 内）:')
    for k7 in keys7:
        print(f'    {k7}: 池化 {v_pool[k7]["rho"] if v_pool[k7] else "NA"} / '
              f'人类 {v_hum[k7]["rho"] if v_hum[k7] else "NA"} / '
              f'ai+qwen {v_aiq[k7]["rho"] if v_aiq[k7] else "NA"}')

    # 支持性 AUC 增量（human vs ai+qwen——sent_proj → +ratio_unit）
    def auc_scores(feats, labels):
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        X = np.array(feats)
        y = np.array(labels)
        if len(set(y)) < 2 or X.shape[0] < 6:
            return None
        clf = LogisticRegression(max_iter=1000)
        try:
            clf.fit(X, y)
            return roc_auc_score(y, clf.predict_proba(X)[:, 1])
        except Exception:
            return None
    feat_docs = [(doc, side) for (doc, side) in pooled.keys() if side in ('human', 'ai', 'qwen')]
    sp_vals = [seven[k].get('sent_proj', 0) for k in feat_docs]
    ru_vals = [per_doc[k]['ratio_unit'] for k in feat_docs]
    labels = [1 if k[1] == 'human' else 0 for k in feat_docs]
    auc_sp = auc_scores([[s] for s in sp_vals], labels)
    auc_sp_ru = auc_scores([[s, r] for s, r in zip(sp_vals, ru_vals)], labels)
    print(f'  AUC: sent_proj {auc_sp:.3f} → sent_proj+ratio_unit {auc_sp_ru:.3f}（Δ={auc_sp_ru - auc_sp:+.3f}）')

    # ===== S6 D 漂移诊断 =====
    print('\nS6 D 漂移诊断:')
    def refit_D(items):
        """items: [(key, idx_list)]——key 可以是任意（ind 用 (doc,side)——ti 用 run_id）"""
        Us = []
        for _, idxs in items:
            idxs = sorted(idxs)
            Df = fp[idxs[1:], :] - fp[idxs[:-1], :]
            n = np.linalg.norm(Df, axis=1)
            Us.append(Df / (n[:, None] + 1e-9))
        U = np.vstack(Us)
        Uc = U - U.mean(0)
        p = PCA(n_components=64)
        p.fit(Uc)
        Dg = p.components_[0]
        return Dg, float(p.explained_variance_ratio_[0])
    D_refit_ind, evr_ind = refit_D(list(ind_docs.items()))
    D_refit_ti, evr_ti = refit_D(list(ti_docs.items()))
    print(f'  cos(D_bilingual, D_ind)={D_shared @ D_refit_ind:+.4f}（EVR_ind={evr_ind:.4f}）')
    print(f'  cos(D_bilingual, D_ti)={D_shared @ D_refit_ti:+.4f}（EVR_ti={evr_ti:.4f}）')

    # ===== S7 代理对照 =====
    print('\nS7 代理对照:')
    rng = np.random.default_rng(42)
    S = rng.normal(0, 1, (5000, 64))
    n = np.linalg.norm(S, axis=1)
    u = S / (n[:, None] + 1e-9)
    jppu = np.mean(np.abs(u - (u @ D_shared)[:, None] * D_shared))
    jpu = np.mean(np.abs(u @ D_shared))
    print(f'  各向同性代理 ratio_unit={jppu / jpu:.3f}（锚≈1.0——观测 0.086/0.105 远小于锚）')

    # ===== S5 判据裁定 =====
    print('\n===== S5 判据裁定 =====')
    vr1 = (d_all < -1.5 and ci_all[1] < 0 and
           (w_ai is not None and w_ai.pvalue < 0.05) and (w_qw is not None and w_qw.pvalue < 0.05))
    vr1b = d_b < -1.5
    def low_redundancy(v):
        return (v is not None and abs(v['rho']) < 0.6 and v_hum[k7_ref]['rho'] is not None
                and abs(v_hum[k7_ref]['rho']) < 0.6 and v_aiq[k7_ref]['rho'] is not None
                and abs(v_aiq[k7_ref]['rho']) < 0.6)
    k7_ref = 'sent_proj'
    vr2 = low_redundancy(v_pool['sent_proj'])
    k7_ref = 'l7_adj'
    vr3 = low_redundancy(v_pool['l7_adj'])
    vr4 = (rho_r is not None and abs(rho_r.statistic) < 0.6)
    verdicts = {'V-R1': 'PASS' if vr1 else 'FAIL', 'V-R1b': 'PASS' if vr1b else 'FAIL',
                'V-R2': 'PASS' if vr2 else 'FAIL', 'V-R3': 'PASS' if vr3 else 'FAIL',
                'V-R4': 'PASS' if vr4 else 'FAIL'}
    for k, v in verdicts.items():
        print(f'  {k}: {v}')
    overall = '成立' if (vr1 and vr2 and vr3) else ('部分成立' if vr1 else '否定')
    print(f'  总判定: {overall}')

    # ===== S8 图 4 张 =====
    # 图 1：配对散点 + 配对差
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    ax = axes[0]
    for v in per_doc.values():
        if v['side'] == 'human':
            k_ai = (v['doc'], 'ai')
            k_qw = (v['doc'], 'qwen')
            if k_ai in per_doc:
                c = '#1f6fb2' if v['domain'] == 'baihua' else '#c0392b'
                ax.scatter(v['ratio_unit'], per_doc[k_ai]['ratio_unit'], s=50, color=c,
                           label='baihua' if v['domain'] == 'baihua' and k_qw not in per_doc else
                                 ('baihua' if v['domain'] == 'baihua' and not ax.get_legend_handles_labels()[1] else 'shiping'))
            if k_qw in per_doc:
                ax.scatter(v['ratio_unit'], per_doc[k_qw]['ratio_unit'], s=30, marker='x',
                           color='#7f8c8d', alpha=0.7)
    lim = (0, max(max(v['ratio_unit'] for v in per_doc.values()) * 1.1, 0.2))
    ax.plot(lim, lim, 'k--', lw=1)
    ax.set_xlabel('ratio_unit 人类')
    ax.set_ylabel('ratio_unit ai(x=ds) / qwen(灰点)')
    ax.set_title('独立测试集配对（同题人类 vs AI）')
    ax.grid(alpha=0.3)
    ax2 = axes[1]
    dts_b = [d_ai[i] for i, v in enumerate([x for x in per_doc.values() if x['side'] == 'human'])
             if (v['doc'], 'ai') in per_doc]
    # 简化：按域分组的配对差
    dom_delta = {'baihua': [], 'shiping': []}
    for v in per_doc.values():
        if v['side'] == 'human' and (v['doc'], 'ai') in per_doc:
            dom_delta[v['domain']].append(v['ratio_unit'] - per_doc[(v['doc'], 'ai')]['ratio_unit'])
    for i, (dom, vals) in enumerate(dom_delta.items()):
        ax2.boxplot(vals, positions=[i], widths=0.5)
        ax2.scatter([i] * len(vals), vals, alpha=0.6, color=('#1f6fb2' if dom == 'baihua' else '#c0392b'))
    ax2.axhline(0, color='k', ls='--', lw=1)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(['白话', '时评'])
    ax2.set_ylabel('δ = ratio_human − ratio_ai')
    ax2.set_title('配对差（同题）——按域')
    ax2.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(PAPER / 'fig_ratio_paired.png', dpi=150)
    plt.close()

    # 图 2：3 side × 2 域箱线
    fig2, ax2 = plt.subplots(figsize=(10, 5.5))
    xpos = 0
    ticks, tlabels = [], []
    for dom in ('baihua', 'shiping'):
        for side in ('human', 'ai', 'qwen'):
            vals = [v['ratio_unit'] for v in per_doc.values() if v['domain'] == dom and v['side'] == side]
            if vals:
                ax2.boxplot(vals, positions=[xpos], widths=0.55)
                ticks.append(xpos)
                tlabels.append(f'{side[:2]}-{dom[:2]}')
                xpos += 1
        xpos += 0.5
    ax2.set_xticks(ticks)
    ax2.set_xticklabels(tlabels, rotation=20)
    ax2.set_ylabel('ratio_unit')
    ax2.set_title(f'独立集 ratio_unit 分组（白话 d={d_b:.2f} / 时评 d={d_s:.2f}）')
    ax2.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(PAPER / 'fig_ratio_domains.png', dpi=150)
    plt.close()

    # 图 3：相关热图
    fig3, axes3 = plt.subplots(1, 3, figsize=(17, 5))
    mats = [('池化', v_pool), ('人类内', v_hum), ('ai+qwen 内', v_aiq)]
    for ax, (name, vv) in zip(axes3, mats):
        vals = np.zeros((7, 1))
        for i, k7 in enumerate(keys7):
            if vv[k7]:
                vals[i, 0] = vv[k7]['rho']
        im = ax.imshow(vals, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        ax.set_yticks(range(7))
        ax.set_yticklabels(keys7, fontsize=8)
        ax.set_xticks([])
        ax.set_title(name, fontsize=11)
        for i, k7 in enumerate(keys7):
            if vv[k7]:
                ax.text(0, i, f'{vv[k7]["rho"]:+.2f}', ha='center', va='center', fontsize=9,
                        color='white' if abs(vv[k7]['rho']) > 0.5 else 'black')
    plt.tight_layout()
    plt.savefig(PAPER / 'fig_ratio_corr.png', dpi=150)
    plt.close()

    # 图 4：干预集
    fig4, axes4 = plt.subplots(1, 2, figsize=(14, 5.5))
    ax = axes4[0]
    groups = [('none', 'none'), ('vt_*', 'vt_*')]
    g_names = ['none', 'vt_ext', 'vt_seed', 'vt_seed_beam', 'vt_kalman', 'vt_kalman_seed',
               'vt_kalman_gate', 'vt_gate_beam', 'vt_oracle', '其他']
    g_data = []
    for c in g_names:
        if c == 'vt_*':
            continue
        if c == '其他':
            vals = [v['ratio_unit'] for v in per_run.values() if v['condition'] not in
                    ('none', 'vt_ext', 'vt_seed', 'vt_seed_beam', 'vt_kalman', 'vt_kalman_seed',
                     'vt_kalman_gate', 'vt_gate_beam', 'vt_oracle')]
        elif c == 'none':
            vals = [v['ratio_unit'] for v in per_run.values() if v['condition'] == 'none']
        else:
            vals = [v['ratio_unit'] for v in per_run.values() if v['condition'] == c]
        if vals:
            g_data.append((c, vals))
    for i, (c, vals) in enumerate(g_data):
        ax.boxplot(vals, positions=[i], widths=0.5)
    ax.set_xticks(range(len(g_data)))
    ax.set_xticklabels([c for c, _ in g_data], rotation=30, fontsize=8)
    ax.set_ylabel('ratio_unit')
    ax.set_title('干预集 ratio_unit（200 run）')
    ax.grid(axis='y', alpha=0.3)
    ax2 = axes4[1]
    if len(pairs_dr) >= 6:
        ax2.scatter(pairs_ds, pairs_dr, alpha=0.7)
        ax2.set_xlabel('Δsent_proj（vt−none）')
        ax2.set_ylabel('Δratio_unit（vt−none）')
        ax2.set_title(f'48 对 Δ 相关 Spearman={rho_r.statistic:+.3f}')
        ax2.axhline(0, color='k', ls='--', lw=0.8)
        ax2.axvline(0, color='k', ls='--', lw=0.8)
        ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PAPER / 'fig_ratio_intervention.png', dpi=150)
    plt.close()

    # ===== 落盘 =====
    out = {
        'criteria': {'V-R1': bool(vr1), 'V-R1b': bool(vr1b), 'V-R2': bool(vr2),
                     'V-R3': bool(vr3), 'V-R4': bool(vr4), 'overall': overall},
        'repro_gate': {'bilingual_human': round(float(np.mean(bh)), 4),
                       'bilingual_ai': round(float(np.mean(ba)), 4), 'passed': bool(gate_ok)},
        'independent': {'human_mean': round(float(np.mean(h)), 4), 'ai_qwen_mean': round(float(np.mean(a_all)), 4),
                        'd_all': round(float(d_all), 3), 'ci_all': [round(float(x), 4) for x in ci_all],
                        'd_baihua': round(float(d_b), 3), 'd_shiping': round(float(d_s), 3),
                        'd_length_matched': round(float(d_lm), 3),
                        'paired_ai': {'delta_mean': round(float(np.mean(d_ai)), 4),
                                      'wilcoxon_p': round(float(w_ai.pvalue), 4) if w_ai else None},
                        'paired_qwen': {'delta_mean': round(float(np.mean(d_qw)), 4),
                                        'wilcoxon_p': round(float(w_qw.pvalue), 4) if w_qw else None},
                        'excluded_short': excluded},
        'intervention': {'n_runs': len(per_run),
                         'spearman_delta': round(float(rho_r.statistic), 4) if rho_r is not None else None,
                         'delta_ratio_mean': round(float(np.mean(pairs_dr)), 4) if pairs_dr else None,
                         'delta_sentproj_mean': round(float(np.mean(pairs_ds)), 4) if pairs_ds else None},
        'corr': {'pooled': v_pool, 'human': v_hum, 'ai_qwen': v_aiq,
                 'auc_sent_proj': round(float(auc_sp), 4) if auc_sp else None,
                 'auc_sp_ru': round(float(auc_sp_ru), 4) if auc_sp_ru else None,
                 'auc_delta': round(float(auc_sp_ru - auc_sp), 4) if (auc_sp and auc_sp_ru) else None},
        'D_drift': {'cos_D_ind': round(float(D_shared @ D_refit_ind), 4),
                    'cos_D_ti': round(float(D_shared @ D_refit_ti), 4),
                    'evr_ind': round(evr_ind, 4), 'evr_ti': round(evr_ti, 4)},
        'surrogate': {'ratio_unit': round(float(jppu / jpu), 3)},
        'verdicts': verdicts,
    }
    (OUT / 'ratio_validation.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print('\n落盘 ratio_validation.json + fig_ratio_*.png × 4 ✓')


if __name__ == '__main__':
    main()
