# -*- coding: utf-8 -*-
"""v0.82-1 跳跃点确定性审查（回答"当前数据能保证确定性正确的跳跃点吗"）

审查缺口（v0.81 未算的）：
A1 段边界混淆：跳跃点中段边界差分（para 变化处）占比——人类 vs AI——人类段数多→段边界跳跃多？
A2 |Δ| × 句元长度：长句元是否更容易跳跃（指纹范数偏置）
A3 跳跃的维度贡献：哪些维度主导跳跃（|Δ| 的维度分解——dim10/48 占比 vs 其他）
A4 模型因素：DS vs Qwen 的跳跃密度/幅度差异（模型是生成侧变量）
A5 跳跃点文本抽查：跳跃点是否真实对应主题/指代转折（语义验证——打印样例）
"""
import sys, json
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy import stats as sc

BASE = Path('C:/Users/bai/Desktop/小说系统')
OUT = BASE / 'data' / 'dim_analysis'

ORG = [10, 11, 34, 46, 48, 59]


def main():
    fp = np.load(OUT / 'fp_matrix.npz')['fp']
    rows = json.loads((OUT / 'rows.json').read_text(encoding='utf-8'))
    docs = defaultdict(list)
    for i, r in enumerate(rows):
        if r['source'] == 'bilingual_zh':
            docs[r['doc']].append(i)
    human = sorted([x for x in docs if x.startswith('ZH-H')], key=lambda x: len(docs[x]))
    ai = sorted([x for x in docs if x.startswith('ZH-A')], key=lambda x: len(docs[x]))
    all_norms = []
    for dlist in (human, ai):
        for doc in dlist:
            idx = np.array(docs[doc])
            D = fp[idx[1:], :] - fp[idx[:-1], :]
            all_norms.append(np.linalg.norm(D, axis=1))
    p90 = float(np.quantile(np.concatenate(all_norms), 0.90))
    print(f'p90 = {p90:.3f}')

    # ===== A1 段边界混淆 =====
    print('\nA1 段边界跳跃占比（para 变化处——段边界差分）:')
    for grp, dlist in (('human', human), ('ai', ai)):
        n_jump, n_bd_jump, n_bd_total, n_total = 0, 0, 0, 0
        for doc in dlist:
            idx = np.array(docs[doc])
            paras = [rows[i]['para'] for i in idx]
            D = fp[idx[1:], :] - fp[idx[:-1], :]
            nrm = np.linalg.norm(D, axis=1)
            bd = [i for i in range(len(paras) - 1) if paras[i + 1] != paras[i]]
            n_bd_total += len(bd)
            n_total += len(nrm)
            for i in bd:
                if nrm[i] > p90:
                    n_bd_jump += 1
            n_jump += int(np.sum(nrm > p90))
        print(f'  {grp}: 段边界差分 {n_bd_total}（占全部 {n_bd_total / n_total * 100:.1f}%）——'
              f'段边界跳跃 {n_bd_jump}（占跳跃 {n_bd_jump / max(n_jump, 1) * 100:.1f}%）——'
              f'段边界跳跃率 {n_bd_jump / max(n_bd_total, 1) * 100:.1f}% vs 非边界跳跃率 {(n_jump - n_bd_jump) / max(n_total - n_bd_total, 1) * 100:.1f}%')

    # ===== A2 |Δ| × 句元长度 =====
    print('\nA2 |Δ| 与句元长度相关:')
    for grp, dlist in (('human', human), ('ai', ai)):
        lens, norms = [], []
        for doc in dlist:
            idx = np.array(docs[doc])
            D = fp[idx[1:], :] - fp[idx[:-1], :]
            nrm = np.linalg.norm(D, axis=1)
            for i in range(len(nrm)):
                lens.append(len(rows[idx[i + 1]]['clause']))
                norms.append(nrm[i])
        r = sc.spearmanr(lens, norms)
        print(f'  {grp}: Spearman(|Δ|, 句元长度)={r.statistic:+.3f} p={r.pvalue:.3f}')

    # ===== A3 跳跃的维度贡献 =====
    print('\nA3 跳跃的维度贡献（|Δ| 维度分解——top 维占比）:')
    for grp, dlist in (('human', human), ('ai', ai)):
        acc = np.zeros(64)
        cnt = 0
        for doc in dlist:
            idx = np.array(docs[doc])
            D = fp[idx[1:], :] - fp[idx[:-1], :]
            nrm = np.linalg.norm(D, axis=1)
            jpos = np.where(nrm > p90)[0]
            if len(jpos):
                acc += np.sum(np.abs(D[jpos]), 0)
                cnt += len(jpos)
        top = np.argsort(acc)[::-1][:8]
        org_share = acc[ORG].sum() / (acc.sum() + 1e-9)
        print(f'  {grp}: 跳跃主导维 top-8={top.tolist()}——org 组贡献 {org_share * 100:.1f}%——'
              f'dim10 占比 {acc[10] / (acc.sum() + 1e-9) * 100:.1f}%——dim48 占比 {acc[48] / (acc.sum() + 1e-9) * 100:.1f}%')

    # ===== A4 模型因素（DS vs Qwen）=====
    print('\nA4 模型因素（DS vs Qwen）:')
    ds = [x for x in ai if 'DS' in x]
    qw = [x for x in ai if 'QWEN' in x]
    for mdl, dlist in (('DS', ds), ('Qwen', qw)):
        dens, amps = [], []
        for doc in dlist:
            idx = np.array(docs[doc])
            D = fp[idx[1:], :] - fp[idx[:-1], :]
            nrm = np.linalg.norm(D, axis=1)
            dens.append(np.sum(nrm > p90) / len(nrm) * 100)
            amps.append(float(np.mean(nrm[nrm > p90])) if np.any(nrm > p90) else 0.0)
        print(f'  {mdl}（n={len(dlist)}）: 密度 {np.mean(dens):.2f}/100 句元——跳跃幅度均值 {np.mean(amps):.3f}')
    d_ds = cohens_d([np.sum(np.linalg.norm(fp[np.array(docs[d])[1:], :] - fp[np.array(docs[d])[:-1], :], axis=1) > p90) / len(np.array(docs[d])) * 100 for d in ds],
                    [np.sum(np.linalg.norm(fp[np.array(docs[d])[1:], :] - fp[np.array(docs[d])[:-1], :], axis=1) > p90) / len(np.array(docs[d])) * 100 for d in qw]) if ds and qw else None

    # ===== A5 跳跃点文本抽查（人类 vs AI 各 3 个最大跳跃）=====
    print('\nA5 跳跃点文本抽查（最大跳跃——语义验证）:')
    for grp, dlist in (('human', human), ('ai', ai)):
        samples = []
        for doc in dlist:
            idx = np.array(docs[doc])
            D = fp[idx[1:], :] - fp[idx[:-1], :]
            nrm = np.linalg.norm(D, axis=1)
            t = int(np.argmax(nrm))
            samples.append((nrm[t], doc, rows[idx[t]]['clause'], rows[idx[t + 1]]['clause']))
        samples.sort(reverse=True)
        print(f'  [{grp}] 最大跳跃 3 个:')
        for nrm, doc, c1, c2 in samples[:3]:
            print(f'    ({nrm:.2f} | {doc}):')
            print(f'      ← {c1[:50]}')
            print(f'      → {c2[:50]}')


def cohens_d(x, y):
    return (np.mean(x) - np.mean(y)) / np.sqrt((np.var(x) + np.var(y)) / 2 + 1e-9)


if __name__ == '__main__':
    main()
