# -*- coding: utf-8 -*-
"""v0.82-2 跳跃机制分析（为什么 AI 跳跃更少——找可控变量）

1. 干预集解码策略 × 跳跃密度/幅度（现成 200 run——beam/logits β/概率 β/温度固定 0.9）
2. 语言特征归因：per-doc 语言特征 × 跳跃密度相关（句长/标点/代词/连接词/词熵）
3. AI vs 人类语言特征差异（哪些特征与跳跃密度共变——机制线索）
4. 可控变量清单（推理期：温度/top_k/top_p/beam/β——训练期：损失项）
"""
import sys, json
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy import stats as sc

BASE = Path('C:/Users/bai/Desktop/小说系统')
OUT = BASE / 'data' / 'dim_analysis'


def cohens_d(x, y):
    return (np.mean(x) - np.mean(y)) / np.sqrt((np.var(x) + np.var(y)) / 2 + 1e-9)


def main():
    fp = np.load(OUT / 'fp_matrix.npz')['fp']
    rows = json.loads((OUT / 'rows.json').read_text(encoding='utf-8'))
    docs = defaultdict(list)
    for i, r in enumerate(rows):
        if r['source'] == 'training_intervention':
            docs[r['doc']].append(i)
    all_norms = []
    for idxs in docs.values():
        idxs.sort(key=lambda i: (rows[i]['seg'], i))
        D = fp[idxs[1:], :] - fp[idxs[:-1], :]
        all_norms.append(np.linalg.norm(D, axis=1))
    p90 = float(np.quantile(np.concatenate(all_norms), 0.90))
    print(f'干预集 p90 = {p90:.3f}')

    # ===== 1. 解码策略 × 跳跃密度/幅度 =====
    print('\n1. 解码策略 × 跳跃密度/幅度（200 run）:')
    groups = {
        'none（基线）': ['none'],
        'beam5（束搜索）': ['beam5', 'beam5_ctl'],
        'logits β1.0': ['lg10', 'lg10u'],
        'logits β0.5': ['lg05'],
        '概率层 β0.5': ['b05'],
        '概率层 β0.3': ['b03'],
        '种子+beam': ['sentence_seed_beam', 'seed_only'],
        'vt 注入系': ['vt_ext', 'vt_seed', 'vt_seed_beam', 'vt_kalman', 'vt_kalman_seed',
                     'vt_kalman_gate', 'vt_gate_beam'],
        '外部锚 logits': ['t2_prompt', 't3_prompt', 't3_human', 't3_self'],
        'PID/Kalman 闭环': ['pid_kalman', 'pid_kalman_ext', 'p_kalman_strategy'],
    }
    stats = {}
    for gname, conds in groups.items():
        dens, amps = [], []
        for run_id, idxs in docs.items():
            cond = run_id.split('-')[0]
            if cond not in conds:
                continue
            idxs.sort(key=lambda i: (rows[i]['seg'], i))
            D = fp[idxs[1:], :] - fp[idxs[:-1], :]
            nrm = np.linalg.norm(D, axis=1)
            dens.append(np.sum(nrm > p90) / len(nrm) * 100)
            amps.append(float(np.mean(nrm[nrm > p90])) if np.any(nrm > p90) else 0.0)
        if dens:
            stats[gname] = {'n': len(dens), 'density': round(float(np.mean(dens)), 2),
                            'amp': round(float(np.mean(amps)), 3)}
            print(f'  {gname}（n={len(dens)}）: 密度 {np.mean(dens):.2f}/100 句元——幅度 {np.mean(amps):.3f}')

    # none vs 各组的 d
    none_d = stats.get('none（基线）', {}).get('density')
    if none_d:
        print('\n  相对 none 的密度差（d）:')
        for gname in groups:
            if gname == 'none（基线）':
                continue
            s = stats.get(gname)
            if not s:
                continue
            print(f'    {gname}: {s["density"] - none_d:+.2f}/100 句元')

    # ===== 2. 语言特征 × 跳跃密度（per-doc 相关）=====
    print('\n2. 语言特征归因（jieba——干预集 per-run）:')
    import jieba
    import jieba.posseg as pseg
    jieba.setLogLevel(60)
    CONJ = set('虽然 但是 但 因为 所以 然而 于是 因此 却 便 就 不过 而且 并且 如果 那么 只要 只有 无论 即使 尽管 可是 于是乎'.split())
    feats = []
    for run_id, idxs in docs.items():
        idxs.sort(key=lambda i: (rows[i]['seg'], i))
        D = fp[idxs[1:], :] - fp[idxs[:-1], :]
        nrm = np.linalg.norm(D, axis=1)
        dens = np.sum(nrm > p90) / len(nrm) * 100
        amp = float(np.mean(nrm[nrm > p90])) if np.any(nrm > p90) else 0.0
        # 语言特征（全 run 句元）
        clauses = [rows[i]['clause'] for i in idxs]
        lens = [len(c) for c in clauses]
        pron_n = conj_n = punct_n = 0
        n_tok = 0
        for c in clauses:
            toks = list(pseg.cut(c))
            pron_n += sum(1 for _, f in toks if f == 'r')
            conj_n += sum(1 for x, _ in toks if x in CONJ)
            punct_n += sum(1 for ch in c if ch in '。！？，、；：')
            n_tok += max(len(toks), 1)
        # 词级熵（字符级近似——per clause 唯一字符比例）
        ent = np.mean([len(set(c)) / max(len(c), 1) for c in clauses])
        feats.append({'run': run_id, 'density': dens, 'amp': amp,
                      'clause_len': float(np.mean(lens)), 'pron': pron_n / n_tok,
                      'conj': conj_n / n_tok, 'punct': punct_n / max(len(clauses), 1),
                      'char_diversity': ent})
    # 相关
    keys = ['clause_len', 'pron', 'conj', 'punct', 'char_diversity']
    print(f'  runs: {len(feats)}——Spearman(特征, 跳跃密度):')
    for k in keys:
        xs = [f[k] for f in feats]
        ys = [f['density'] for f in feats]
        r = sc.spearmanr(xs, ys)
        print(f'    {k}: ρ={r.statistic:+.3f} p={r.pvalue:.3f}')

    # ===== 3. AI vs 人类语言特征（机制线索——人类语料）=====
    print('\n3. 人类 vs AI 语言特征（bilingual——跳跃密度差异的机制线索）:')
    docs_b = defaultdict(list)
    for i, r in enumerate(rows):
        if r['source'] == 'bilingual_zh':
            docs_b[r['doc']].append(i)
    human = sorted([x for x in docs_b if x.startswith('ZH-H')], key=lambda x: len(docs_b[x]))
    ai = sorted([x for x in docs_b if x.startswith('ZH-A')], key=lambda x: len(docs_b[x]))
    all_b = []
    for dlist in (human, ai):
        for doc in dlist:
            all_b.append(np.linalg.norm(fp[np.array(docs_b[doc])[1:], :] - fp[np.array(docs_b[doc])[:-1], :], axis=1))
    p90_b = float(np.quantile(np.concatenate(all_b), 0.90))
    for grp, dlist in (('人类', human), ('AI', ai)):
        dens, lens, pron, conj, punct, ent = [], [], [], [], [], []
        for doc in dlist:
            idxs = np.array(docs_b[doc])
            D = fp[idxs[1:], :] - fp[idxs[:-1], :]
            nrm = np.linalg.norm(D, axis=1)
            dens.append(np.sum(nrm > p90_b) / len(nrm) * 100)
            clauses = [rows[i]['clause'] for i in idxs]
            lens.append(np.mean([len(c) for c in clauses]))
            pron_n = conj_n = n_tok = 0
            for c in clauses:
                toks = list(pseg.cut(c))
                pron_n += sum(1 for _, f in toks if f == 'r')
                conj_n += sum(1 for x, _ in toks if x in CONJ)
                n_tok += max(len(toks), 1)
            pron.append(pron_n / max(n_tok, 1))
            conj.append(conj_n / max(n_tok, 1))
            punct.append(np.mean([sum(1 for ch in c if ch in '。！？，、；：') for c in clauses]))
            ent.append(np.mean([len(set(c)) / max(len(c), 1) for c in clauses]))
        print(f'  {grp}（n={len(dlist)}）: 密度 {np.mean(dens):.2f}——句长 {np.mean(lens):.1f}——'
              f'代词 {np.mean(pron):.4f}——连接词 {np.mean(conj):.4f}——标点 {np.mean(punct):.1f}——'
              f'字符多样性 {np.mean(ent):.3f}')

    # 可控变量清单落盘
    out = {
        'p90': round(p90, 3),
        'decoding_strategies': stats,
        'lang_feat_corr': {k: {'rho': round(float(sc.spearmanr([f[k] for f in feats],
                                                               [f['density'] for f in feats]).statistic), 3),
                                'p': round(float(sc.spearmanr([f[k] for f in feats],
                                                               [f['density'] for f in feats]).pvalue), 4)}
                           for k in keys},
        'human_ai_lang': {},
    }
    (OUT / 'jump_mechanism.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print('\n落盘 jump_mechanism.json ✓')


if __name__ == '__main__':
    main()
