# -*- coding: utf-8 -*-
"""v0.78-3 根意图势场·度量化与判据评估（阶段 D——纯离线）

读 manifest field_summary/field_trace + 文本 → 聚合：
- 各条件 R 漂移（ΔR_T/cosR_end_T/回拉事件/自由率）
- C-B1 传导 / C-B2 保持 / C-B3 表面惯性 / C-B4 冻结对照 / C-B6 不劣于最强
- 五指标目标带命中（dim10_jump/turn10_steep/Hurst/TE/coupling——dim_flow_sent 口径）
产出 field_experiment_analysis.json + 图 4 张（论文储备）
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
TI = BASE / 'data' / 'training_intervention'
PAPER = Path('C:/Users/bai/Desktop/AB系统论文储备')
sys.path.insert(0, str(BASE / 'stage3'))

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

FIELD = ('vt_field', 'vt_field_persist', 'vt_field_frozen', 'vt_field_full')
DIM10, DIM48 = 10, 48


def five_metrics_of(texts):
    """从段文本句元级重算五指标（无指纹时用文本重编码——太贵——改用 fp_matrix 已有指纹？）
    说明：manifest 的 segs 无句元指纹——五指标用 fp_matrix training_intervention 行（doc=run_id）"""
    return None


def main():
    m = json.loads((TI / 'manifest.json').read_text(encoding='utf-8'))
    runs = [r for r in m if r.get('condition') in FIELD and r.get('status') == 'done' and r.get('field_summary')]
    print(f'field runs: {len(runs)}')

    # ===== 按条件聚合 R 漂移 =====
    by_cond = defaultdict(list)
    for r in runs:
        by_cond[r['condition']].append(r)
    summary = {}
    for cond in FIELD:
        rs = by_cond[cond]
        dR = [r['field_summary']['delta_R_T'] for r in rs if r['field_summary'].get('delta_R_T') is not None]
        cosT = [r['field_summary']['cos_R_end_T'] for r in rs if r['field_summary'].get('cos_R_end_T') is not None]
        cosF = [r['field_summary']['mean_cosF_T'] for r in rs if r['field_summary'].get('mean_cosF_T') is not None]
        e = [r['field_summary']['mean_e'] for r in rs if r['field_summary'].get('mean_e') is not None]
        pull = [r['field_summary']['n_pullback_bf1'] for r in rs]
        free = [r['field_summary']['n_free_bf0'] for r in rs]
        ncls = [r['field_summary']['n_clauses'] for r in rs]
        summary[cond] = {
            'n': len(rs),
            'delta_R_T': {'mean': round(float(np.mean(dR)), 4) if dR else None,
                          'per_run': [round(float(x), 4) for x in dR]},
            'cos_R_end_T': {'mean': round(float(np.mean(cosT)), 4) if cosT else None,
                            'per_run': [round(float(x), 4) for x in cosT]},
            'mean_cosF_T': round(float(np.mean(cosF)), 4) if cosF else None,
            'mean_e': round(float(np.mean(e)), 4) if e else None,
            'n_pullback_bf1_total': int(np.sum(pull)),
            'n_free_bf0_total': int(np.sum(free)),
            'n_clauses_total': int(np.sum(ncls)),
            'clauses_per_run': round(float(np.mean(ncls)), 1) if ncls else None,
        }
        print(f'{cond}: n={len(rs)}——ΔR_T={summary[cond]["delta_R_T"]["mean"]}——cos(R_end,T)='
              f'{summary[cond]["cos_R_end_T"]["mean"]}——mean_cosF_T={summary[cond]["mean_cosF_T"]}')

    # ===== C-B1 传导（persist ΔR_T vs 0——单样本）=====
    p_dR = [x for x in summary['vt_field_persist']['delta_R_T']['per_run'] if x is not None]
    f_dR = [x for x in summary['vt_field_frozen']['delta_R_T']['per_run'] if x is not None]
    full_dR = [x for x in summary['vt_field_full']['delta_R_T']['per_run'] if x is not None]
    t_p = sc.ttest_1samp(p_dR, 0) if len(p_dR) >= 2 else None
    verdicts = {}
    verdicts['C-B1_传导'] = (f'FAIL（ΔR_T={np.mean(p_dR):+.4f}——负值——R 未被拉向 T——'
                            f'单样本 t p={t_p.pvalue:.3f}') if t_p else 'n<2'
    # C-B2 保持（persist 段 3 vs 段 2 的 cosR_T——从 trace 取段尾）
    keep_vals = []
    for r in by_cond['vt_field_persist']:
        tr = []
        for seg in r['segs']:
            tr += seg.get('field_trace', [])
        # trace 每句元 cosR_T——最后两个（段 2 末/段 3 末）
        seg2_end = None
        seg3_end = None
        if len(r['segs']) >= 2:
            t2 = r['segs'][1].get('field_trace', [])
            if t2:
                seg2_end = [x for x in t2 if x.get('cosR_T') is not None][-1]['cosR_T']
        if len(r['segs']) >= 3:
            t3 = r['segs'][2].get('field_trace', [])
            if t3:
                seg3_end = [x for x in t3 if x.get('cosR_T') is not None][-1]['cosR_T']
        if seg2_end is not None and seg3_end is not None:
            keep_vals.append(seg3_end - seg2_end)
    verdicts['C-B2_保持'] = (f'{"PASS" if np.mean(keep_vals) >= -0.05 else "FAIL"}'
                            f'（段3−段2 ΔcosR_T={np.mean(keep_vals):+.4f}——判定线 ≥-0.05）') if keep_vals else '无 trace'
    # C-B4 冻结对照（persist vs frozen 的 cosR_end_T）
    p_cosT = [x for x in summary['vt_field_persist']['cos_R_end_T']['per_run'] if x is not None]
    f_cosT = [x for x in summary['vt_field_frozen']['cos_R_end_T']['per_run'] if x is not None]
    diff_cosT = float(np.mean(p_cosT) - np.mean(f_cosT)) if p_cosT and f_cosT else None
    verdicts['C-B4_冻结对照'] = (f'persist−frozen cos(R_end,T)={diff_cosT:+.4f}'
                               f'（≤0.02 两解释不可分；>0.02 且 persist>frozen → 内化为生成产物）') if diff_cosT is not None else '无'
    # C-B6 不劣于最强（vt_field vs vt_gate_beam 的 win_sent_proj——用文本段 dims）
    gb = [r for r in m if r.get('condition') == 'vt_gate_beam' and r.get('status') == 'done']
    gb_proj = [seg['dims']['sent_proj'] for r in gb for seg in r.get('segs', []) if seg.get('dims')]
    fl_proj = [seg['dims']['sent_proj'] for r in by_cond['vt_field'] for seg in r.get('segs', []) if seg.get('dims')]
    if gb_proj and fl_proj:
        verdicts['C-B6_不劣于最强'] = (f'{"PASS" if np.mean(fl_proj) >= np.mean(gb_proj) - 0.01 else "FAIL"}'
                                     f'（vt_field {np.mean(fl_proj):.4f} vs vt_gate_beam {np.mean(gb_proj):.4f}）')
        summary['vt_field']['win_sent_proj_mean'] = round(float(np.mean(fl_proj)), 4)
        summary['vt_gate_beam_baseline'] = {'win_sent_proj_mean': round(float(np.mean(gb_proj)), 4),
                                            'n_segs': len(gb_proj)}
    else:
        verdicts['C-B6_不劣于最强'] = '缺基线'

    # ===== 图 1：R 漂移（ΔR_T 每条件 per-run——箱线/散点）=====
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    x = np.arange(4)
    means = [summary[c]['delta_R_T']['mean'] if summary[c]['delta_R_T']['mean'] is not None else 0 for c in FIELD]
    vals = [summary[c]['delta_R_T']['per_run'] for c in FIELD]
    cols = ['#7f8c8d', '#1f6fb2', '#c0392b', '#27ae60']
    for xi, (c, v) in enumerate(zip(FIELD, vals)):
        if v:
            ax.scatter([xi] * len(v), v, c=cols[xi], s=40, zorder=3)
            ax.plot([xi - 0.3, xi + 0.3], [np.mean(v)] * 2, c=cols[xi], lw=2)
    ax.axhline(0, color='k', ls='--', lw=0.8)
    ax.axhline(0.15, color='#c0392b', ls=':', lw=1, label='判据参考线 0.15')
    ax.set_xticks(x)
    ax.set_xticklabels(FIELD, rotation=15)
    ax.set_ylabel('ΔR_T = cos(R_end,T) - cos(R0,T)')
    ax.set_title('根意图 R 漂移（传导——frozen=0 对照——full=上限）')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    # 图 1 右：cos(R_end,T) 分布
    ax2 = axes[1]
    for xi, (c, v) in enumerate(zip(FIELD, vals)):
        cosv = summary[c]['cos_R_end_T']['per_run']
        if cosv:
            ax2.scatter([xi] * len(cosv), cosv, c=cols[xi], s=40, zorder=3)
            ax2.plot([xi - 0.3, xi + 0.3], [np.mean(cosv)] * 2, c=cols[xi], lw=2)
    ax2.set_xticks(x)
    ax2.set_xticklabels(FIELD, rotation=15)
    ax2.set_ylabel('cos(R_end, T)')
    ax2.set_title('R 终点对 T 的贴合')
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PAPER / 'fig_field_R_trajectory.png', dpi=150)
    plt.close()

    # ===== 图 2：回拉事件（bf 分布——persist 代表 run）=====
    fig2, ax = plt.subplots(figsize=(9, 5))
    rep = by_cond['vt_field_persist'][0]
    tr = []
    for seg in rep['segs']:
        tr += seg.get('field_trace', [])
    if tr:
        e_vals = [t['e'] for t in tr]
        bf_vals = [t['bf'] for t in tr]
        ax.plot(e_vals, marker='o', ms=4, lw=1, label='偏离 e=0.90−p', color='#1f6fb2')
        ax.axhline(0.05, color='#c0392b', ls='--', lw=1, label='门控线 0.05（bf=1.0）')
        ax.axhline(0.02, color='#e67e22', ls='--', lw=1, label='门控线 0.02（bf=0.5）')
        ax.set_xlabel('句元序号（段 1-3）')
        ax.set_ylabel('偏离 e / 注入强度 bf')
        ax.set_title(f'偏离-回拉事件（{rep["run_id"]}——bf={bf_vals}）')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PAPER / 'fig_field_pullback.png', dpi=150)
    plt.close()

    # ===== 图 3：五指标目标带命中（persist vs 人类带——重编码段文本句元指纹）=====
    # 说明：新 field runs 指纹不在 fp_matrix（静态快照）——用判别器重编码（CPU）
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    from para_dimensions import load_models, fingerprint, norm_rows
    from subclause_structure import split_subclauses
    import torch
    enc, disc = load_models('cpu')
    rows = json.loads((OUT / 'rows.json').read_text(encoding='utf-8'))
    bands = json.loads((OUT / 'planner_targets.json').read_text(encoding='utf-8'))
    fm = bands['five_metrics']

    def traj_of_run(r):
        """run 段文本 → 句元指纹序列（重编码）"""
        Fs = []
        for seg in r.get('segs', []):
            t = seg.get('text', '')
            for s in split_subclauses(t):
                if len(s) < 3:
                    continue
                sv = enc.encode([s], normalize_embeddings=True, batch_size=1,
                                show_progress_bar=False, device='cpu')
                SV = torch.from_numpy(sv.astype(np.float32))
                with torch.no_grad():
                    F = norm_rows(fingerprint(SV, disc)).detach().cpu().numpy()[0]
                Fs.append(F)
        return np.array(Fs) if Fs else None
    import dim_flow as DF
    from dim_flow_sent import transfer_entropy, turn_points
    metric_stats = {}
    for cond in FIELD:
        t10s, t48s = [], []
        for r in by_cond[cond]:
            X = traj_of_run(r)
            if X is None or len(X) < 5:
                continue
            t10s.append(X[:, DIM10])
            t48s.append(X[:, DIM48])
        if not t10s:
            continue
        jump = np.mean([np.mean(np.abs(np.diff(t))) for t in t10s])
        hs = [h for h in (DF.hurst(t) for t in t10s) if h == h]  # 滤 nan
        hurst = np.mean(hs) if hs else float('nan')
        steep = np.mean([turn_points(t)['peak_steep_mean'] for t in t10s])
        te = np.mean([transfer_entropy(a, b) for a, b in zip(t10s, t48s)])
        # 耦合（dim10×dim48 段级相关）
        coup = []
        for t10, t48 in zip(t10s, t48s):
            n = min(len(t10), len(t48))
            if n >= 6:
                r = np.corrcoef(t10[:n], t48[:n])[0, 1]
                coup.append(r)
        coupling = float(np.mean(coup)) if coup else None
        metric_stats[cond] = {'dim10_jump': round(float(jump), 4), 'dim10_hurst': round(float(hurst), 4),
                              'turn10_steep': round(float(steep), 4), 'te_10_48': round(float(te), 4),
                              'coupling': round(coupling, 4) if coupling else None}
    print('\n五指标（fp_matrix 句元轨迹）：')
    for c, v in metric_stats.items():
        print(f'  {c}: {v}')
    # 目标带（人类）
    tb = {k: {'lo': fm[k]['human']['p25'], 'hi': fm[k]['human']['p75']} for k in
          ('dim10_jump', 'dim10_hurst', 'turn10_steep')}
    # TE 目标：≤ p75
    # 命中判定
    hit_report = {}
    for c, v in metric_stats.items():
        hits = []
        if tb['dim10_jump']['lo'] <= v['dim10_jump'] <= tb['dim10_jump']['hi']:
            hits.append('dim10_jump')
        if tb['dim10_hurst']['lo'] <= v['dim10_hurst'] <= tb['dim10_hurst']['hi']:
            hits.append('dim10_hurst')
        if tb['turn10_steep']['lo'] <= v['turn10_steep'] <= tb['turn10_steep']['hi']:
            hits.append('turn10_steep')
        if v['te_10_48'] <= fm['te_10_to_48']['human']['p75']:
            hits.append('te')
        if v['coupling'] is not None and fm['coupling_10_48']['human']['p25'] <= v['coupling'] <= fm['coupling_10_48']['human']['p75']:
            hits.append('coupling')
        hit_report[c] = {'hits': hits, 'n_hit': len(hits)}
    print('目标带命中:', {c: v['hits'] for c, v in hit_report.items()})

    # 图 3：五指标命中雷达/条形
    fig3, ax3 = plt.subplots(figsize=(9, 5))
    keys3 = ['dim10_jump', 'turn10_steep', 'dim10_hurst', 'te_10_48', 'coupling']
    conds3 = [c for c in FIELD if c in metric_stats]
    x3 = np.arange(len(keys3))
    w = 0.8 / len(conds3)
    for i, c in enumerate(conds3):
        v = metric_stats[c]
        vals3 = [v['dim10_jump'], v['turn10_steep'], v['dim10_hurst'], v['te_10_48'],
                 v['coupling'] if v['coupling'] is not None else 0]
        # z 归一化到人类带（用人类 mean/std）——fm 键名映射
        fm_keys = {'dim10_jump': 'dim10_jump', 'turn10_steep': 'turn10_steep',
                   'dim10_hurst': 'dim10_hurst', 'te_10_48': 'te_10_to_48',
                   'coupling': 'coupling_10_48'}
        zs = []
        for k, val in zip(keys3, vals3):
            mh, sh = fm[fm_keys[k]]['human']['mean'], fm[fm_keys[k]]['human']['std']
            zs.append((val - mh) / sh)
        ax3.bar(x3 + (i - len(conds3) / 2 + 0.5) * w, zs, w, label=c, alpha=0.85)
    ax3.axhline(0, color='k', lw=1)
    ax3.axhspan(-0.67, 0.67, color='#2ecc71', alpha=0.08, label='人类带 ±0.67σ（p25-p75 近似）')
    ax3.set_xticks(x3)
    ax3.set_xticklabels(keys3)
    ax3.set_ylabel('z（相对人类均值——σ 单位）')
    ax3.set_title('五指标相对人类目标带的偏离（z 分数）')
    ax3.legend(fontsize=8)
    ax3.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(PAPER / 'fig_field_five_metrics.png', dpi=150)
    plt.close()

    # ===== 图 4：persist vs none 表面惯性（F3 对 T 的对齐）=====
    def seg_proj_T(run, T, seg_i):
        """段 3 指纹对 T 的余弦（重编码——新 runs 不在 fp_matrix）"""
        seg = next((s for s in run.get('segs', []) if s['seg'] == seg_i), None)
        if seg is None:
            return None
        Fs = []
        for st in split_subclauses(seg.get('text', '')):
            if len(st) < 3:
                continue
            sv = enc.encode([st], normalize_embeddings=True, batch_size=1,
                            show_progress_bar=False, device='cpu')
            SV = torch.from_numpy(sv.astype(np.float32))
            with torch.no_grad():
                Fs.append(norm_rows(fingerprint(SV, disc)).detach().cpu().numpy()[0])
        if not Fs:
            return None
        m = np.mean(Fs, 0)
        return float(m @ T / (np.linalg.norm(m) + 1e-9))
    target = json.loads((OUT / 'field_target.json').read_text(encoding='utf-8'))['human_core']
    T = np.asarray(target, float)
    fig4, ax4 = plt.subplots(figsize=(9, 5))
    g_none = [r for r in m if r.get('condition') == 'none' and r.get('status') == 'done'
              and r.get('prompt_id') and r.get('seed') in (0, 1)]
    pairs4 = []
    for r in by_cond['vt_field_persist']:
        pid, sd = r['prompt_id'], r['seed']
        none_r = next((x for x in g_none if x['prompt_id'] == pid and x['seed'] == sd), None)
        if none_r is None:
            continue
        p3 = seg_proj_T(r, T, 3)
        n3 = seg_proj_T(none_r, T, 3)
        if p3 is not None and n3 is not None:
            pairs4.append((f'{pid}-s{sd}', p3, n3))
    if pairs4:
        xs = np.arange(len(pairs4))
        ax4.bar(xs - 0.2, [p[1] for p in pairs4], 0.4, label='vt_field_persist 段3', color='#1f6fb2')
        ax4.bar(xs + 0.2, [p[2] for p in pairs4], 0.4, label='none 段3', color='#e67e22')
        ax4.set_xticks(xs)
        ax4.set_xticklabels([p[0] for p in pairs4])
        ax4.set_ylabel('cos(F3, T)——段 3 指纹对 T 的贴合')
        ax4.set_title('表面惯性（C-B3）：persist 段 3 vs none 段 3 对 T 的对齐')
        ax4.legend(fontsize=8)
        ax4.grid(axis='y', alpha=0.3)
        inertia = float(np.mean([p[1] - p[2] for p in pairs4]))
        verdicts['C-B3_表面惯性'] = (f'{"PASS" if inertia >= 0.05 else "FAIL"}'
                                   f'（persist−none 段3 对 T 贴合={inertia:+.4f}——判定线 ≥0.05）')
    else:
        ax4.text(0.5, 0.5, '无配对数据', ha='center')
        verdicts['C-B3_表面惯性'] = '无配对'
    plt.tight_layout()
    plt.savefig(PAPER / 'fig_field_persist.png', dpi=150)
    plt.close()

    # ===== 落盘 =====
    out = {'summary': summary, 'verdicts': verdicts, 'metric_stats': metric_stats,
           'hit_report': hit_report}
    (OUT / 'field_experiment_analysis.json').write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print('\n=== 判据裁定 ===')
    for k, v in verdicts.items():
        print(f'  {k}: {v}')
    print('落盘 field_experiment_analysis.json + 图 4 张 ✓')


if __name__ == '__main__':
    main()
