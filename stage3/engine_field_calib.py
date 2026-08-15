# -*- coding: utf-8 -*-
"""v0.78-2 α 校准分析：从校准跑 manifest 的 field_summary 选 α + 功效分析定 C-B1 阈值

用法：校准跑后运行——读 data/training_intervention/manifest.json 的 field_summary
——输出 α 选定 + MDE（配对 t 检验 α=0.05/power=0.8）——阈值 = max(MDE, 0.15)
"""
import sys, json
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path
import numpy as np
from scipy import stats as sc

BASE = Path('C:/Users/bai/Desktop/小说系统')
MAN = BASE / 'data' / 'training_intervention' / 'manifest.json'


def main():
    runs = json.loads(MAN.read_text(encoding='utf-8'))
    calib = [r for r in runs if r.get('condition') in ('vt_field_persist', 'vt_field_frozen', 'vt_field_full')
             and r.get('status') == 'done' and r.get('field_summary')]
    print(f'校准/势场 runs 数: {len(calib)}')
    if not calib:
        print('无 field_summary——先跑校准/主实验')
        return
    # 按 α 分组（校准跑用 --field-alpha 覆盖）
    by_alpha = {}
    for r in calib:
        a = r['field_summary']['alpha']
        by_alpha.setdefault(a, []).append(r)
    for a in sorted(by_alpha):
        rs = by_alpha[a]
        dR = [r['field_summary'].get('delta_R_T') for r in rs]
        dR = [x for x in dR if x is not None]
        if dR:
            print(f'  α={a}: n={len(dR)}——ΔR_T = {np.mean(dR):+.4f}±{np.std(dR):.4f}（[{min(dR):+.4f}, {max(dR):+.4f}]）')
    # 功效分析（配对 t 检验——H0: ΔR_T=0——n=6 主矩阵 per-prompt 配对）
    # 校准跑各 α 的 ΔR_T 合并分布（保守 sd）
    all_dR = [r['field_summary']['delta_R_T'] for r in calib
              if r['field_summary'].get('delta_R_T') is not None]
    if len(all_dR) >= 2:
        sd = np.std(all_dR, ddof=1)
        mean = np.mean(all_dR)
        for n in (6, 8):
            df = n - 1
            t_crit = sc.t.ppf(0.975, df)          # 双侧 0.05
            t_power = sc.t.ppf(0.8, df, loc=0)    # power 0.8 近似
            mde = t_crit * sd / np.sqrt(n)
            thresh = max(mde, 0.15)
            print(f'  功效分析 n={n}: sd={sd:.4f}——MDE={mde:.4f}——C-B1 阈值={thresh:.4f}')
        print(f'  校准跑均值 ΔR_T={mean:+.4f}（若 ≥ 阈值则效应可测；否则报告"弱传导"）')
    # 全部势场 run 的传导/保持汇总（主实验后调用）
    for cond in ('vt_field', 'vt_field_persist', 'vt_field_frozen', 'vt_field_full'):
        rs = [r for r in calib if r.get('condition') == cond]
        if not rs:
            continue
        dR = [r['field_summary'].get('delta_R_T') for r in rs]
        dR = [x for x in dR if x is not None]
        cosT = [r['field_summary'].get('cos_R_end_T') for r in rs]
        cosT = [x for x in cosT if x is not None]
        if dR:
            print(f'  {cond}: n={len(dR)}——ΔR_T={np.mean(dR):+.4f}±{np.std(dR):.4f}——cos(R_end,T)={np.mean(cosT):.4f}')


if __name__ == '__main__':
    main()
