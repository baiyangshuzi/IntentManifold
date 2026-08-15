# -*- coding: utf-8 -*-
"""v0.78-1 根意图势场·离线证据（阶段 A——零 GPU）

回答："根据现有数据，持续表层意图指向能否传导到根意图"的可行性判断。

A.1 表层→根传导证据（201 runs 指纹集——主分析限定 v_dir=prompt 核心子集）：
  vt_ext / vt_seed_beam / vt_seed（注入目标=prompt 核心——离线复算同口径）vs none（同 prompt×seed 配对）
  1) 轨迹系统性偏转 δ_align：vt 的 mean(cos(S2,S3,v_dir)) − none 同式
  2) 扰动维激活偏移 Δact：v_dir 高激活维+dim10/48——[mean(F2,3)−mean(F1)]_vt − 同式_none（人类 band_std 单位化）
  3) 惯性静态代理 I = cos(F3−F1, v_dir−F1)
  副证据（单独报告——注入目标漂移）：vt_kalman 系（vt_kalman/vt_kalman_seed/vt_kalman_gate/vt_gate_beam——在线 EMA 方向）

A.2 核心稳定性（bilingual_zh 人类 10 篇 vs AI 20 篇）：
  贴核心度 cos(S_k, C_doc) + 核心漂移率 mean(angle(S_{k+1},S_k))——d + bootstrap 95% CI——探索性措辞

A.3 产出：field_evidence.json（三判据裁定）+ field_target.json（T=人类核心——independent_test B-01..B-32 指纹均值）+ 图 2 张
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

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# PROMPTS 与 gen_theme_guidance 一致（离线复算 v_dir 用）
PROMPTS = {
    'P1': '雨夜，刑警老周在城中追缉一名连续作案的凶手。他沿着水洼里的脚印，走进一条旧巷。',
    'P2': '老渡口，江水浑黄，渡船在雾中缓缓靠岸。守渡人老陈坐在棚下，等一个三天没有出现的乘客。',
    'P3': '星舰在深空迷航，窗外是无边星海。导航官望着陌生的星图，试图辨认来路。',
}
MAIN_VT = ['vt_ext', 'vt_seed_beam', 'vt_seed']          # v_dir=prompt 核心（主分析）
KALMAN_VT = ['vt_kalman', 'vt_kalman_seed', 'vt_kalman_gate', 'vt_gate_beam']  # 副证据（注入目标漂移）
DIM10, DIM48 = 10, 48


def load_planner_bands():
    p = json.loads((OUT / 'planner_targets.json').read_text(encoding='utf-8'))
    bands = {b['dim']: b for b in p['dim_bands']}
    return bands


def seg_means(fp, rows, doc):
    """doc 的段均值指纹序列（seg 分组）——返回 [(seg, S_unit)]"""
    idx = [i for i, r in enumerate(rows) if r['doc'] == doc]
    segs = defaultdict(list)
    for i in idx:
        segs[rows[i]['seg']].append(i)
    out = []
    for s in sorted(segs):
        m = fp[segs[s], :].mean(0)
        n = np.linalg.norm(m)
        if n > 1e-9:
            out.append((s, m / n))
    return out


def core_of_texts_simple(enc, disc, texts):
    """离线复算 prompt 核心（同 core_of_texts 口径：分行→句元→指纹→mean→归一）"""
    from para_dimensions import fingerprint, norm_rows
    import torch
    from subclause_structure import split_subclauses
    Fs = []
    for t in texts:
        for line in t.split('\n'):
            for c in split_subclauses(line):
                if len(c) < 3:
                    continue
                sv = enc.encode([c], normalize_embeddings=True, batch_size=1,
                                show_progress_bar=False, device='cpu')
                SV = torch.from_numpy(sv.astype(np.float32))
                with torch.no_grad():
                    F = norm_rows(fingerprint(SV, disc)).detach().cpu().numpy()[0]
                Fs.append(F)
    if not Fs:
        return None
    c = np.mean(Fs, 0)
    return c / (np.linalg.norm(c) + 1e-9)


def bootstrap_ci(x, y, n_boot=2000, seed=42, paired=False):
    """bootstrap 95% CI——paired=True 用配对差；否则两组独立重采样"""
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x, float), np.asarray(y, float)
    if paired:
        d = x - y
        means = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)]
    else:
        means = []
        for _ in range(n_boot):
            xb = x[rng.integers(0, len(x), len(x))].mean()
            yb = y[rng.integers(0, len(y), len(y))].mean()
            means.append(xb - yb)
    return np.percentile(means, [2.5, 97.5])


def main():
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    d = np.load(OUT / 'fp_matrix.npz')
    fp = d['fp']
    rows = json.loads((OUT / 'rows.json').read_text(encoding='utf-8'))
    bands = load_planner_bands()

    # ===== 加载判别器（CPU——算 prompt 核心用） =====
    print('加载判别器（CPU）……')
    from para_dimensions import load_models
    enc, disc = load_models('cpu')
    vdirs = {}
    for pid, text in PROMPTS.items():
        c = core_of_texts_simple(enc, disc, [text])
        vdirs[pid] = c
        print(f'  {pid} 核心范数: {np.linalg.norm(c):.3f}（单位化后）')

    # ===== A.1 表层→根传导证据 =====
    print('\n=== A.1 传导证据（主分析：vt_ext/vt_seed_beam/vt_seed vs none——同 prompt×seed 配对）===')
    # 配对集：vt 条件 × {P1,P2,P3} × {s0,s1}
    pairs = []
    for cond in MAIN_VT:
        for pid in ['P1', 'P2', 'P3']:
            for sd in [0, 1]:
                vt_doc = f'{cond}-{pid}-s{sd}'
                none_doc = f'none-{pid}-s{sd}'
                vt_segs = seg_means(fp, rows, vt_doc)
                none_segs = seg_means(fp, rows, none_doc)
                if len(vt_segs) >= 2 and len(none_segs) >= 2:
                    pairs.append((cond, pid, sd, vt_segs, none_segs, vdirs[pid]))
    print(f'配对 runs: {len(pairs)}')

    # 1) 轨迹系统性偏转 δ_align（S2,S3 对 v_dir）
    def align_of(segs, vdir):
        vals = [float(s @ vdir) for _, s in segs[1:]]  # S2,S3
        return float(np.mean(vals)) if vals else 0.0

    align_vt, align_none, deltas = [], [], []
    for cond, pid, sd, vt_s, none_s, vdir in pairs:
        a_vt, a_none = align_of(vt_s, vdir), align_of(none_s, vdir)
        align_vt.append(a_vt)
        align_none.append(a_none)
        deltas.append(a_vt - a_none)
    delta_align = float(np.mean(deltas))
    ci_align = bootstrap_ci(align_vt, align_none, paired=True)
    print(f'δ_align = {delta_align:+.4f}（95% CI [{ci_align[0]:+.4f}, {ci_align[1]:+.4f}]）——方向一致 runs: '
          f'{sum(1 for x in deltas if x > 0)}/{len(deltas)}')

    # 2) 扰动维激活偏移 Δact（v_dir 高激活维 + dim10/48——人类 band_std 单位化）
    # F2,3 − F1 的偏移差（vt − none）
    def act_shift(segs, vdir, dims):
        F1 = dict(segs).get(1)
        later = [S for s, S in segs if s >= 2]
        if F1 is None or not later:
            return {}
        F1u = F1 / (np.linalg.norm(F1) + 1e-9)
        res = {}
        for k in dims:
            base = float(F1u[k])
            later_mean = float(np.mean([S[k] for S in later]))
            res[k] = later_mean - base
        return res

    # v_dir 高激活维（|v_dir| 前 5）
    top_dims = set()
    for vdir in vdirs.values():
        top = np.argsort(np.abs(vdir))[::-1][:5]
        top_dims.update(top.tolist())
    dims_of_interest = sorted(top_dims | {DIM10, DIM48})
    delta_acts = {k: [] for k in dims_of_interest}
    for cond, pid, sd, vt_s, none_s, vdir in pairs:
        av = act_shift(vt_s, vdir, dims_of_interest)
        an = act_shift(none_s, vdir, dims_of_interest)
        for k in dims_of_interest:
            delta_acts[k].append(av.get(k, 0.0) - an.get(k, 0.0))
    # band_std 单位化
    act_d_stats = {}
    for k in dims_of_interest:
        bstd = bands[k]['human']['band_std']
        vals = np.array(delta_acts[k]) / (bstd + 1e-9)
        act_d_stats[k] = {
            'dim': k, 'delta_act': round(float(np.mean(delta_acts[k])), 4),
            'd_bandstd': round(float(np.mean(vals)), 3),
            'direction_agree': f'{sum(1 for v in vals if v > 0)}/{len(vals)}',
        }
        print(f'  dim{k}: Δact={act_d_stats[k]["delta_act"]:+.4f}（d={act_d_stats[k]["d_bandstd"]:+.2f} band_std——'
              f'方向一致 {act_d_stats[k]["direction_agree"]}）')
    act_mean_d = float(np.mean([v['d_bandstd'] for v in act_d_stats.values()]))

    # 3) 惯性静态代理 I = cos(F3−F1, v_dir−F1)
    def inertia(segs, vdir):
        segmap = dict(segs)
        if 1 not in segmap or 3 not in segmap:
            return None
        F1, F3 = segmap[1], segmap[3]
        disp = F3 - F1
        toward = vdir - F1
        n1, n2 = np.linalg.norm(disp), np.linalg.norm(toward)
        if n1 < 1e-9 or n2 < 1e-9:
            return None
        return float(disp @ toward / (n1 * n2))

    I_vt, I_none = [], []
    for cond, pid, sd, vt_s, none_s, vdir in pairs:
        i1, i2 = inertia(vt_s, vdir), inertia(none_s, vdir)
        if i1 is not None and i2 is not None:
            I_vt.append(i1)
            I_none.append(i2)
    I_vt_m, I_none_m = float(np.mean(I_vt)), float(np.mean(I_none))
    print(f'惯性静态代理 I: vt {I_vt_m:.3f} vs none {I_none_m:.3f}（差 {I_vt_m - I_none_m:+.3f}）')

    # ===== 副证据：vt_kalman 系（单独报告——注入目标漂移）=====
    print('\n副证据（vt_kalman 系——注入目标为在线 EMA 方向——仅报告不判据）：')
    k_pairs = []
    for cond in KALMAN_VT:
        for pid in ['P1', 'P2', 'P3']:
            for sd in [0, 1]:
                vt_doc = f'{cond}-{pid}-s{sd}'
                none_doc = f'none-{pid}-s{sd}'
                vt_s = seg_means(fp, rows, vt_doc)
                none_s = seg_means(fp, rows, none_doc)
                if len(vt_s) >= 2 and len(none_s) >= 2:
                    k_pairs.append((cond, vt_s, none_s, vdirs[pid]))
    k_deltas = []
    for cond, vt_s, none_s, vdir in k_pairs:
        k_deltas.append(align_of(vt_s, vdir) - align_of(none_s, vdir))
    print(f'  {len(k_pairs)} 配对——δ_align(EMA 系) = {np.mean(k_deltas):+.4f}')

    # ===== A.2 核心稳定性（bilingual_zh——人类 vs AI）=====
    print('\n=== A.2 核心稳定性（bilingual_zh 人类 vs AI——探索性）===')
    def core_stats(fp, rows, docs):
        out = {}
        for doc in docs:
            segs = []
            idx = [i for i, r in enumerate(rows) if r['doc'] == doc]
            paras = defaultdict(list)
            for i in idx:
                paras[rows[i]['para']].append(i)
            for p in sorted(paras):
                m = fp[paras[p], :].mean(0)
                n = np.linalg.norm(m)
                if n > 1e-9:
                    segs.append(m / n)
            if len(segs) < 3:
                continue
            C = np.mean(segs, 0)
            C = C / (np.linalg.norm(C) + 1e-9)
            cling = [float(s @ C) for s in segs]
            drift = [float(np.arccos(np.clip(a @ b, -1, 1))) for a, b in zip(segs[:-1], segs[1:])]
            out[doc] = {'n_seg': len(segs), 'cling_mean': float(np.mean(cling)),
                        'drift_mean': float(np.mean(drift))}
        return out

    docs = defaultdict(list)
    for i, r in enumerate(rows):
        if r['source'] == 'bilingual_zh':
            docs[r['doc']].append(i)
    human_docs = [x for x in docs if x.startswith('ZH-H')]
    ai_docs = [x for x in docs if x.startswith('ZH-A')]
    hs = core_stats(fp, rows, human_docs)
    as_ = core_stats(fp, rows, ai_docs)
    h_cling = [v['cling_mean'] for v in hs.values()]
    a_cling = [v['cling_mean'] for v in as_.values()]
    h_drift = [v['drift_mean'] for v in hs.values()]
    a_drift = [v['drift_mean'] for v in as_.values()]
    d_cling = (np.mean(h_cling) - np.mean(a_cling)) / np.sqrt((np.var(h_cling) + np.var(a_cling)) / 2 + 1e-9)
    d_drift = (np.mean(h_drift) - np.mean(a_drift)) / np.sqrt((np.var(h_drift) + np.var(a_drift)) / 2 + 1e-9)
    ci_drift = bootstrap_ci(h_drift, a_drift, paired=False)
    u_drift = sc.mannwhitneyu(h_drift, a_drift)
    print(f'贴核心度: 人类 {np.mean(h_cling):.4f} vs AI {np.mean(a_cling):.4f}——d={d_cling:+.2f}')
    print(f'核心漂移率(rad/段): 人类 {np.mean(h_drift):.4f} vs AI {np.mean(a_drift):.4f}——d={d_drift:+.2f} '
          f'p={u_drift.pvalue:.3f}——95% CI [{ci_drift[0]:+.4f}, {ci_drift[1]:+.4f}]')

    # ===== 判据裁定 =====
    print('\n=== 判据裁定 ===')
    c_a1 = (delta_align >= 0.05 and act_mean_d >= 0.3 and
            sum(1 for x in deltas if x > 0) / len(deltas) >= 0.8)
    c_a2 = (I_vt_m >= 0.6 and I_vt_m > I_none_m + 0.3 * np.std(I_none))
    c_a3 = (np.mean(h_cling) >= 0.9 and ci_drift[1] < 0)
    print(f'C-A1 传导: δ_align={delta_align:+.4f}(≥0.05) Δact d={act_mean_d:+.2f}(≥0.3) '
          f'方向一致 {sum(1 for x in deltas if x > 0)}/{len(deltas)}(≥80%) → {"PASS" if c_a1 else "FAIL"}')
    print(f'C-A2 惯性代理: I_vt={I_vt_m:.3f}(≥0.6) 且 {I_vt_m:.3f} > {I_none_m + 0.3 * np.std(I_none):.3f} → {"PASS" if c_a2 else "FAIL"}')
    print(f'C-A3 核心稳定: 贴核心 {np.mean(h_cling):.3f}(≥0.9) 且漂移 CI [{ci_drift[0]:+.4f},{ci_drift[1]:+.4f}] 不含 0 → {"PASS" if c_a3 else "FAIL"}')

    # ===== field_target.json（T=人类核心——independent_test B- 文档）=====
    ind = [i for i, r in enumerate(rows) if r['source'] == 'independent_test' and r['side'] == 'human']
    T_raw = fp[ind, :].mean(0)
    T = T_raw / (np.linalg.norm(T_raw) + 1e-9)
    # mean_cos_to_doc_cores：每篇对自身核心的贴合
    ind_docs = sorted(set(rows[i]['doc'] for i in ind))
    per_doc_cos = []
    for doc in ind_docs:
        idx = [i for i in ind if rows[i]['doc'] == doc]
        c = fp[idx, :].mean(0)
        c = c / (np.linalg.norm(c) + 1e-9)
        per_doc_cos.append(float(T @ c))
    target = {
        'human_core': T.tolist(),
        'source': f'independent_test human B-01..B-{len(ind_docs):02d}',
        'n_docs': len(ind_docs), 'n_rows': len(ind),
        'mean_cos_to_doc_cores': round(float(np.mean(per_doc_cos)), 4),
    }
    (OUT / 'field_target.json').write_text(json.dumps(target, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'\nfield_target.json: T=人类核心（{len(ind_docs)} 篇/{len(ind)} 句元——mean_cos_to_doc_cores={target["mean_cos_to_doc_cores"]}）')

    # ===== 图 1：配对偏转 + 激活偏移 =====
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # 左：配对 δ_align（每配对一条——vt vs none）
    ax = axes[0]
    x = np.arange(len(align_vt))
    ax.bar(x - 0.2, align_vt, 0.4, label='vt（注入）', color='#1f6fb2')
    ax.bar(x + 0.2, align_none, 0.4, label='none（对照）', color='#e67e22')
    ax.set_xlabel('配对 runs（同 prompt×seed）')
    ax.set_ylabel('cos(S2,S3, v_dir)——对注入目标的贴合')
    ax.set_title(f'轨迹系统性偏转（δ_align={delta_align:+.3f}——vt 贴目标 vs none）')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    # 右：扰动维激活偏移（band_std 单位化）
    ax2 = axes[1]
    dims = [v['dim'] for v in act_d_stats.values()]
    vals = [v['d_bandstd'] for v in act_d_stats.values()]
    cols = ['#1f6fb2' if x in (DIM10, DIM48) else '#7f8c8d' for x in dims]
    ax2.bar([str(x) for x in dims], vals, color=cols)
    ax2.axhline(0.3, color='#c0392b', ls='--', lw=1, label='判据线 0.3')
    ax2.set_xlabel('维度')
    ax2.set_ylabel('Δact（人类 band_std 单位）')
    ax2.set_title('扰动维激活偏移（vt−none——蓝=dim10/48）')
    ax2.legend(fontsize=8)
    ax2.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(PAPER / 'fig_field_deflection.png', dpi=150)
    plt.close()

    # ===== 图 2：核心稳定性（贴核心度 vs 漂移率散点）=====
    fig2, ax = plt.subplots(figsize=(8, 6))
    ax.scatter([v['cling_mean'] for v in hs.values()], [v['drift_mean'] for v in hs.values()],
               c='#1f6fb2', label=f'人类（n={len(hs)}）', s=60)
    ax.scatter([v['cling_mean'] for v in as_.values()], [v['drift_mean'] for v in as_.values()],
               c='#e67e22', label=f'AI（n={len(as_)}）', s=60)
    ax.axvline(0.9, color='#c0392b', ls='--', lw=1, label='贴核心判据线 0.9')
    ax.set_xlabel('贴核心度（段对全篇核心的余弦）')
    ax.set_ylabel('核心漂移率（rad/段）')
    ax.set_title('核心稳定性：人类 vs AI（探索性——根意图构造的数据基础）')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PAPER / 'fig_field_core_stability.png', dpi=150)
    plt.close()

    # ===== 落盘 =====
    out = {
        'pair_align': {'vt': align_vt, 'none': align_none, 'deltas': deltas,
                       'delta_align': round(delta_align, 4),
                       'ci95': [round(float(x), 4) for x in ci_align],
                       'direction_agree': f'{sum(1 for x in deltas if x > 0)}/{len(deltas)}'},
        'act_shift': act_d_stats, 'act_mean_d_bandstd': round(act_mean_d, 3),
        'inertia': {'vt_mean': round(I_vt_m, 4), 'none_mean': round(I_none_m, 4),
                    'diff': round(I_vt_m - I_none_m, 4)},
        'kalman_sub_evidence': {'n_pairs': len(k_pairs),
                                'delta_align_ema': round(float(np.mean(k_deltas)), 4)},
        'core_stability': {'human_n': len(hs), 'ai_n': len(as_),
                           'human_cling': round(float(np.mean(h_cling)), 4),
                           'ai_cling': round(float(np.mean(a_cling)), 4),
                           'd_cling': round(float(d_cling), 3),
                           'human_drift': round(float(np.mean(h_drift)), 4),
                           'ai_drift': round(float(np.mean(a_drift)), 4),
                           'd_drift': round(float(d_drift), 3),
                           'p_drift': round(float(u_drift.pvalue), 4),
                           'ci95_drift': [round(float(x), 4) for x in ci_drift]},
        'verdict': {'C-A1_传导': 'PASS' if c_a1 else 'FAIL',
                    'C-A2_惯性代理': 'PASS' if c_a2 else 'FAIL',
                    'C-A3_核心稳定': 'PASS' if c_a3 else 'FAIL',
                    'decision_gate': 'GO（进入阶段 B persist 命题）' if c_a1 else 'NO-GO（persist 降级）'},
    }
    (OUT / 'field_evidence.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print('\n落盘 field_evidence.json + field_target.json + fig_field_deflection.png + fig_field_core_stability.png ✓')


if __name__ == '__main__':
    main()
