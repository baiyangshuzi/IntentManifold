# -*- coding: utf-8 -*-
"""v0.74 阶段 B：非线性探针 + 上层追溯（256/512 维复合来源）+ 聚类

B1 非线性探针：独立集白话人类段——18 语言特征 → MLP → 每维激活 R²（5 折 CV）
B2 上层追溯：64 维未解释维度 i ← W₂[i,:]（64×256 权重行）→ top-k 256 维子集 U_i →
             U_i 的 h1 激活与 18 特征 Spearman（256 层可解释性）——判定"复合维度 vs 深层信号"
B3 聚类：11 维相关矩阵
输入：fp_matrix.npz（含 h1——256 维中间激活）+ rows.json + 判别器权重
"""
import sys, json, os, re
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path
import numpy as np
import torch
from scipy import stats as sc

BASE = Path(os.environ.get('INTENT_DYNAMICS_BASE', Path(__file__).resolve().parent.parent))
OUT = BASE / 'data' / 'dim_analysis'
sys.path.insert(0, str(BASE / 'stage3'))
TARGET = [5, 10, 11, 22, 26, 34, 43, 46, 48, 52, 59]


def main():
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    # ===== 数据 =====
    d = np.load(OUT / 'fp_matrix.npz')
    fp, h1 = d['fp'], d['h1']
    rows = json.loads((OUT / 'rows.json').read_text(encoding='utf-8'))
    # 判别器权重
    from para_dimensions import load_models
    import torch as T
    enc, disc = load_models('cpu')
    W2 = disc.net[3].weight.detach().numpy()  # (64, 256)
    W1 = disc.net[0].weight.detach().numpy()  # (256, 512)

    # 独立集白话人类段（B-01..B-20 human——每对单段）
    man = json.load(open(BASE / 'data/independent_test/manifest.json', encoding='utf-8'))
    seg_texts, seg_docs = [], []
    for pair in man.get('pairs', []):
        if 'B-' in pair['pair_id'] and pair.get('human'):
            seg_texts.append(pair['human'])
            seg_docs.append(pair['pair_id'])
    print(f'独立集白话人类段: {len(seg_texts)}')

    # 18 特征（build_dim_probe 口径——内嵌 measure_feats——jieba 词法——无模型依赖）
    import jieba
    import jieba.posseg as pseg
    from subclause_structure import split_subclauses as _ss
    from collections import Counter
    jieba.setLogLevel(60)
    EMOTION = set(['开心', '难过', '愤怒', '害怕', '紧张', '激动', '委屈', '幸福', '痛苦', '温暖',
                   '孤独', '焦虑', '恐惧', '喜悦', '悲伤', '惊喜', '失望', '希望', '爱', '恨',
                   '开心', '伤心', '生气', '担心', '高兴', '难过'])

    def measure_feats(txt):
        ss = [s for s in _ss(txt) if len(s) >= 3]
        if len(ss) < 2:
            return None
        segs = [list(pseg.cut(s)) for s in ss]
        n = len(segs)
        def pos_ratio(flags):
            return sum(1 for seg in segs for w, f in seg if f[0] in flags) / max(n, 1)
        all_words = [w for seg in segs for w, f in seg if len(w) >= 2]
        wc = Counter(all_words)
        return {
            'n_ratio': pos_ratio('n'), 'v_ratio': pos_ratio('v'), 'a_ratio': pos_ratio('a'),
            'd_ratio': pos_ratio('d'), 'r_ratio': pos_ratio('r'),
            'sent_len_mean': float(np.mean([len(s) for s in ss])),
            'punct_density': sum(1 for c in txt if c in '，。！？；：、') / max(len(txt), 1),
            'digit_density': sum(1 for c in txt if c.isdigit()) / max(len(txt), 1),
            'emotion_density': sum(1 for w in all_words if w in EMOTION) / max(n, 1),
            'quote_density': txt.count('"') + txt.count('“') + txt.count('”'),
            'ttr': len(set(all_words)) / max(len(all_words), 1),
            'n_sent': n,
            'exclaim_q': txt.count('！') + txt.count('？'),
            'dash': txt.count('——') + txt.count('—'),
            'fourchar': sum(1 for w in all_words if len(w) == 4) / max(n, 1),
            'func_ratio': pos_ratio('u') + pos_ratio('c') + pos_ratio('p'),
            'word_count': len(all_words),
            'conj_density': sum(1 for seg in segs for w, f in seg if f[0] == 'c') / max(n, 1),
        }

    feats = []
    for t in seg_texts:
        m = measure_feats(t)
        feats.append(m if m else None)
    feat_names = [f for f in feats[0].keys()] if feats[0] else []
    F18 = np.array([[f[k] for k in feat_names] for f in feats if f])
    ok_idx = [i for i, f in enumerate(feats) if f]
    print(f'18 特征矩阵: {F18.shape}——特征: {feat_names}')

    # 段级 h1 与 fp（按 doc 匹配——rows 的独立集 doc）
    from collections import defaultdict
    seg_h1, seg_fp = {}, {}
    for i, r in enumerate(rows):
        if r['source'] == 'independent_test' and r['side'] == 'human' and 'B-' in r['doc']:
            seg_h1.setdefault(r['doc'], []).append(i)
            seg_fp.setdefault(r['doc'], []).append(i)
    H1_seg = np.array([h1[seg_h1[doc]].mean(0) for doc in seg_docs if doc in seg_h1])
    FP_seg = np.array([fp[seg_fp[doc]].mean(0) for doc in seg_docs if doc in seg_fp])
    # 对齐 ok_idx（有特征的段）
    seg_docs_ok = [seg_docs[i] for i in ok_idx]
    mask = [seg_docs_ok.index(doc) for doc in seg_docs_ok if doc in seg_h1]
    H1_ok = np.array([h1[seg_h1[doc]].mean(0) for doc in seg_docs_ok])
    FP_ok = np.array([fp[seg_fp[doc]].mean(0) for doc in seg_docs_ok])
    print(f'段级 h1: {H1_ok.shape} fp: {FP_ok.shape}——特征段数 {F18.shape[0]}')

    # ===== B1 非线性探针（MLP 特征→单维激活——5 折 CV R²）=====
    import torch.nn as nn
    def mlp_probe(X, y, seed=42):
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(X))
        n = len(X)
        k = 5
        preds = np.zeros(n)
        for f in range(k):
            te = idx[f * n // k:(f + 1) * n // k]
            tr = np.concatenate([idx[:f * n // k], idx[(f + 1) * n // k:]])
            Xtr, Xte = X[tr], X[te]
            Xs = (Xtr - Xtr.mean(0)) / (Xtr.std(0) + 1e-9)
            Xte_s = (Xte - Xtr.mean(0)) / (Xtr.std(0) + 1e-9)
            m = nn.Sequential(nn.Linear(X.shape[1], 32), nn.ReLU(), nn.Linear(32, 1))
            opt = torch.optim.Adam(m.parameters(), lr=1e-2)
            Xt = torch.from_numpy(Xs.astype(np.float32))
            yt = torch.from_numpy(y[tr].astype(np.float32)).view(-1, 1)
            Xv = torch.from_numpy(Xte_s.astype(np.float32))
            for ep in range(200):
                opt.zero_grad()
                loss = nn.functional.mse_loss(m(Xt), yt)
                loss.backward()
                opt.step()
            with torch.no_grad():
                preds[te] = m(Xv).numpy().ravel()
        ss_res = ((y - preds) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        return 1 - ss_res / ss_tot

    b1 = {}
    X = F18.astype(np.float32)
    for j in TARGET:
        r2 = mlp_probe(X, FP_ok[:, j])
        b1[j] = {'mlp_r2': round(float(r2), 4), 'verdict': '表层组合' if r2 > 0.6 else '未解释'}
    print('=== B1 非线性探针（18 特征 → 单维激活——MLP 5 折 CV R²）===')
    for j in TARGET:
        print(f'  dim{j}: R²={b1[j]["mlp_r2"]:.3f}——{b1[j]["verdict"]}')

    # ===== B2 上层追溯（64 维 → W₂ 行 → 256 维子集 → h1 探针）=====
    b2 = {}
    for j in TARGET:
        row_w = np.abs(W2[j])  # (256,)
        topk = np.argsort(row_w)[::-1][:8]  # top-8 高权重 256 维
        # U_i 的 h1 探针（18 特征 Spearman）
        sig = []
        for u in topk:
            rho, p = sc.spearmanr(H1_ok[:, u], F18, axis=0) if False else (None, None)
        # 每 256 维对每个 18 特征 Spearman
        u_rho = np.zeros((len(topk), F18.shape[1]))
        u_sig = np.zeros((len(topk), F18.shape[1]), bool)
        for a, u in enumerate(topk):
            for b in range(F18.shape[1]):
                rho, p = sc.spearmanr(H1_ok[:, u], F18[:, b])
                u_rho[a, b] = rho
                u_sig[a, b] = (p < 0.05 and abs(rho) > 0.3)
        n_sig_pairs = int(u_sig.sum())
        n_sig_dims = int((u_sig.sum(1) > 0).sum())
        # 判定：U_i 中 ≥1 个 256 维可解释 → 复合维度
        verdict = '复合维度（256 层可解释）' if n_sig_dims >= 1 else '深层信号（256 层也不可解释）'
        b2[j] = {'top8_256dims': [int(x) for x in topk],
                 'top_weights': [round(float(row_w[u]), 4) for u in topk],
                 'n_sig_pairs': n_sig_pairs, 'n_sig_256dims': n_sig_dims,
                 'best_rho': round(float(np.abs(u_rho).max()), 3),
                 'verdict': verdict}
        print(f'=== B2 追溯 dim{j}: top8 256维={b2[j]["top8_256dims"]}——显著对 {n_sig_pairs}——'
              f'可解释 256 维 {n_sig_dims}/8——max|rho|={b2[j]["best_rho"]}——{verdict}')

    # ===== B3 聚类（11 维相关矩阵）=====
    corr11 = np.corrcoef(FP_ok[:, TARGET].T)
    b3 = {'corr_matrix': np.round(corr11, 3).tolist(),
          'n_pairs_|r|>0.5': int((np.abs(corr11) > 0.5).sum() / 2)}
    print(f'=== B3 聚类：11 维内 |r|>0.5 的配对 {b3["n_pairs_|r|>0.5"]} 个 ===')

    (OUT / 'probe_results.json').write_text(json.dumps(
        {'b1': b1, 'b2': b2, 'b3': b3, 'feat_names': feat_names,
         'note': 'B2 判定：256 层 top-8 高权重子集有显著特征关联→复合维度；无→深层信号'},
        ensure_ascii=False, indent=1), encoding='utf-8')
    print('落盘 probe_results.json ✓')


if __name__ == '__main__':
    main()
