# -*- coding: utf-8 -*-
"""v0.82-3 温度扫描分析（温度 → 跳跃密度/幅度的因果——可控变量最终验证）

读 data/temp_sweep/ 归档（4 温度 × P1 × s0,s1）——重编码句元指纹（判别器 CPU）
→ 跳跃密度（p90 用 bilingual 池 3.229 或干预集池）→ 温度 vs 密度的单调性 + 效应量
→ 与 char_diversity 对照（机制一致性）
"""
import sys, json, os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path
import numpy as np
from scipy import stats as sc

BASE = Path('C:/Users/bai/Desktop/小说系统')
OUT = BASE / 'data' / 'dim_analysis'
SWEEP = BASE / 'data' / 'temp_sweep'
sys.path.insert(0, str(BASE / 'stage3'))

P90_BI = 3.229  # bilingual 池 p90（v0.79 预检验）


def cohens_d(x, y):
    return (np.mean(x) - np.mean(y)) / np.sqrt((np.var(x) + np.var(y)) / 2 + 1e-9)


def main():
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    # 干预集池 p90（fp_matrix training_intervention 行——现成）
    fp = np.load(OUT / 'fp_matrix.npz')['fp']
    rows = json.loads((OUT / 'rows.json').read_text(encoding='utf-8'))
    from collections import defaultdict
    ti = defaultdict(list)
    for i, r in enumerate(rows):
        if r['source'] == 'training_intervention':
            ti[r['doc']].append(i)
    norms_all = []
    for idxs in ti.values():
        idxs.sort()
        D = fp[idxs[1:], :] - fp[idxs[:-1], :]
        norms_all.append(np.linalg.norm(D, axis=1))
    P90_TI = float(np.quantile(np.concatenate(norms_all), 0.90))
    print(f'干预集池 p90 = {P90_TI:.3f}（bilingual 3.229——短文本分布不同——用干预集池）')

    from para_dimensions import load_models, fingerprint
    from subclause_structure import split_subclauses
    import torch
    enc, disc = load_models('cpu')

    results = {}
    p90_use = P90_TI
    for f in sorted(SWEEP.glob('t*.json')):
        temp = f.stem.replace('t', '').replace('', '')
        temp = f.stem[1:]
        temp = float(f.stem[1:]) / 10 if len(f.stem) == 3 else float(f.stem[1:2] + '.' + f.stem[2:])
        entries = json.loads(f.read_text(encoding='utf-8'))
        dens, amps = [], []
        for e in entries:
            texts = [seg.get('text', '') for seg in e.get('segs', [])]
            Fs = []
            for t in texts:
                for s in split_subclauses(t):
                    if len(s) < 3:
                        continue
                    sv = enc.encode([s], normalize_embeddings=True, batch_size=1,
                                    show_progress_bar=False, device='cpu')
                    SV = torch.from_numpy(sv.astype(np.float32))
                    with torch.no_grad():
                        Fs.append(fingerprint(SV, disc).detach().cpu().numpy()[0])
            if len(Fs) < 6:
                continue
            F = np.array(Fs)
            nrm = np.linalg.norm(F[1:] - F[:-1], axis=1)
            dens.append(np.sum(nrm > p90_use) / len(nrm) * 100)
            amps.append(float(np.mean(nrm[nrm > p90_use])) if np.any(nrm > p90_use) else 0.0)
        if dens:
            results[temp] = {'n': len(dens), 'density': round(float(np.mean(dens)), 2),
                             'amp': round(float(np.mean(amps)), 3),
                             'per_run_density': [round(float(x), 2) for x in dens]}
            print(f'温度 {temp}: 密度 {np.mean(dens):.2f}/100 句元——幅度 {np.mean(amps):.3f}（n={len(dens)}）')

    # 单调性（Spearman 温度 × 密度）
    temps = sorted(results)
    if len(temps) >= 3:
        xs = [results[t]['density'] for t in temps]
        r = sc.spearmanr(temps, xs)
        # 全 run 级
        all_d = []
        for t in temps:
            for d in results[t]['per_run_density']:
                all_d.append((t, d))
        r2 = sc.spearmanr([x[0] for x in all_d], [x[1] for x in all_d])
        print(f'\n温度 × 密度 Spearman（温度级）={r.statistic:+.3f} p={r.pvalue:.3f}')
        print(f'温度 × 密度 Spearman（run 级 n={len(all_d)}）={r2.statistic:+.3f} p={r2.pvalue:.3f}')
        # 最高 vs 最低温度 d
        d_ext = cohens_d([x[1] for x in all_d if x[0] == temps[0]], [x[1] for x in all_d if x[0] == temps[-1]])
        print(f'最低 vs 最高温度密度 d={d_ext:+.3f}')

    (OUT / 'temp_analysis.json').write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding='utf-8')
    print('\n落盘 temp_analysis.json ✓')


if __name__ == '__main__':
    main()
