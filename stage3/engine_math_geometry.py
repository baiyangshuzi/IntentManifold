# -*- coding: utf-8 -*-
"""v0.85 数学运算几何结构预实验（纯离线——零 GPU——评审 14 条全吸收）

用户问题：数学运算的符号/数字组合在现有意图空间中是否具备可区分的几何结构？
预实验（最低门槛）：4 类算式（加/减/乘/除各 24 条）——"数字=实体，运算符=链"结构化转写
——bge 编码 → fingerprint——PCA/t-SNE 2D——定量分离判据。

【预注册判据表（先写后跑——两轮评审 14 条吸收）】

| # | 判据 | 判定线 |
|---|------|--------|
| M-A1 | 真词超占位词基线（主判） | sil(真词) > 0.25 且 Δsil > 0.10；或 acc(真词) > 0.7 且 Δacc > 0.15——且置换检验（1000 次）真实值 > 随机 95 分位 |
| M-A2 | 数学域自成一簇（仅参考） | 数学 intra cos > 数学-人类 inter + 0.1（数学全新生成/叙事沿用已有——声明） |

评审吸收：①占位词基线（词汇分类陷阱——Δ 增量判据）；②除法整除限制（C 全整数——无数字类型混入）；
③占位词 P0 预检（甲乙丙丁有顺序语义——acc>0.6/sil>0.3 换随机拼音）；④全新 seed 2026 防数据窥探；
⑤结论措辞收敛（"运算符语义编码的额外可分离信息"——不称"数学结构贡献"）；⑥置换 1000 次；
⑦LDA 前 PCA 降维 95% + 最近邻留一法双报；⑧C 数字分布显式报告；⑨指纹 L2 归一化后 cosine（写死）；
⑩t-SNE 仅可视化不判据；⑪类中心距离比仅报告不判据；⑫M-A2 仅参考不作主判定。

总判定：M-A1 PASS → 预实验成功（最低门槛）→ 正式启动数学意图分析；FAIL → 否定（词面差异即可解释）。
"""
import sys, json
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path
import numpy as np
from scipy import stats as sc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path('C:/Users/bai/Desktop/小说系统')
OUT = BASE / 'data' / 'dim_analysis'
PAPER = Path('C:/Users/bai/Desktop/AB系统论文储备')
sys.path.insert(0, str(BASE / 'stage3'))

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

SEED = 2026
OPS = {'加号': 'add', '减号': 'sub', '乘号': 'mul', '除号': 'div'}
PLACEHOLDERS = ['甲', '乙', '丙', '丁']
RANDOM_PH = ['wu', 'ji', 'pan', 'kuo']
N_PER = 24


def gen_pairs(rng):
    """四类运算数字对（评审 2：除法只选整除对——C 全整数——减法 A>=B）"""
    pairs = {k: [] for k in OPS.values()}
    while len(pairs['add']) < N_PER:
        a, b = rng.integers(2, 100, 2)
        pairs['add'].append((int(a), int(b), int(a + b)))
    while len(pairs['sub']) < N_PER:
        a, b = rng.integers(2, 100, 2)
        if a >= b:
            pairs['sub'].append((int(a), int(b), int(a - b)))
    while len(pairs['mul']) < N_PER:
        a, b = rng.integers(2, 100, 2)
        pairs['mul'].append((int(a), int(b), int(a * b)))
    while len(pairs['div']) < N_PER:
        a, b = rng.integers(2, 100, 2)
        if b != 0 and a % b == 0:
            pairs['div'].append((int(a), int(b), int(a // b)))
    return pairs


def make_structured(pairs, op_words):
    """结构化转写（数字=实体/运算符=链）——同一批数字——op_words 按 OPS 顺序"""
    texts, labels = [], []
    for i, (op_cn, opw) in enumerate(op_words.items()):
        for a, b, c in pairs[OPS[op_cn]]:
            texts.append(f'数字 {a} {opw} 数字 {b} 等于 数字 {c}')
            labels.append(i)
    return texts, np.array(labels)


def mask_digits(text):
    """数字掩码（v0.85-2：全掩码→[NUM]——注意：类内退化 sil=1.0——仅诊断用）"""
    import re
    return re.sub(r'\d+', '[NUM]', text)


def bin_digits(text):
    """区间映射（v0.85-3 修复：数字→小/中/大 3 档——消除值域差异但保留类内多样性——
    全掩码使类内零方差（sil=1.0 数学必然——模板退化）——区间映射是正确口径）"""
    import re
    def rep(m):
        v = int(m.group())
        return '小' if v <= 33 else ('中' if v <= 66 else '大')
    return re.sub(r'\d+', rep, text)


def make_raw(pairs):
    """原始式对照（评审 5：补充展示不判据）"""
    sym = {'add': '+', 'sub': '-', 'mul': '×', 'div': '÷'}
    texts, labels = [], []
    for i, op in enumerate(OPS):
        for a, b, c in pairs[op]:
            texts.append(f'{a} {sym[op]} {b} = {c}')
            labels.append(i)
    return texts, np.array(labels)


def fingerprints_of(texts, enc, disc):
    """整条结构化转写 → bge → fingerprint（原始未归一化——评审 9）"""
    from para_dimensions import fingerprint
    import torch
    Fs = []
    for t in texts:
        sv = enc.encode([t], normalize_embeddings=True, batch_size=1,
                        show_progress_bar=False, device='cpu')
        SV = torch.from_numpy(sv.astype(np.float32))
        with torch.no_grad():
            Fs.append(fingerprint(SV, disc).detach().cpu().numpy()[0])
    return np.array(Fs)


def l2norm(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


def sil_acc(X, labels, n_comp=0.95):
    """silhouette（cosine）+ LDA 留一（PCA 95% 前）+ 最近邻留一——评审 7/14"""
    from sklearn.metrics import silhouette_score, accuracy_score
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.decomposition import PCA
    from sklearn.neighbors import KNeighborsClassifier
    Xn = l2norm(X)
    sil = float(silhouette_score(Xn, labels, metric='cosine'))
    # PCA 降维 95% 方差（评审 14）
    pca = PCA(n_components=n_comp)
    Z = pca.fit_transform(Xn)
    # LDA 留一
    lda = LinearDiscriminantAnalysis()
    preds = []
    for i in range(len(Z)):
        lda.fit(np.delete(Z, i, 0), np.delete(labels, i))
        preds.append(lda.predict(Z[i:i + 1])[0])
    acc_lda = float(accuracy_score(labels, preds))
    # 最近邻留一（评审 14 双报）
    knn = KNeighborsClassifier(n_neighbors=1)
    preds_k = []
    for i in range(len(Xn)):
        knn.fit(np.delete(Xn, i, 0), np.delete(labels, i))
        preds_k.append(knn.predict(Xn[i:i + 1])[0])
    acc_knn = float(accuracy_score(labels, preds_k))
    # 类中心距离比（评审 11：仅报告不判据）
    centers = np.array([Xn[labels == k].mean(0) for k in range(4)])
    centers = l2norm(centers)
    inter = np.mean([1 - c1 @ c2 for i, c1 in enumerate(centers) for c2 in centers[i + 1:]])
    intra = np.mean([1 - Xn[labels == k] @ centers[k] for k in range(4)])
    return {'sil': sil, 'acc_lda': acc_lda, 'acc_knn': acc_knn,
            'center_ratio': round(float(inter / (intra + 1e-9)), 3),
            'inter_cos': round(float(inter), 3), 'intra_cos': round(float(intra), 3)}


def permutation_test(X, labels, n_perm=1000, seed=42):
    """置换检验（评审 6/14）：标签随机打乱 1000 次——silhouette 分布——真实值分位"""
    from sklearn.metrics import silhouette_score
    Xn = l2norm(X)
    true_sil = float(silhouette_score(Xn, labels, metric='cosine'))
    rng = np.random.default_rng(seed)
    sils = []
    for _ in range(n_perm):
        perm = rng.permutation(labels)
        sils.append(float(silhouette_score(Xn, perm, metric='cosine')))
    p_val = float(np.mean(np.array(sils) >= true_sil))
    return {'true_sil': true_sil, 'p': p_val, 'p95': float(np.percentile(sils, 95)),
            'mean_null': float(np.mean(sils))}


def main():
    print('===== v0.85 数学运算几何结构预实验（判据预注册——见文件头） =====')
    import os
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    from para_dimensions import load_models
    enc, disc = load_models('cpu')

    rng = np.random.default_rng(SEED)
    pairs = gen_pairs(rng)

    # ===== P0 占位词预检（独立步骤——评审 12）=====
    print('\nP0 占位词预检:')
    ph_words = dict(zip(OPS.keys(), PLACEHOLDERS))
    ph_texts, ph_labels = make_structured(pairs, ph_words)
    F_ph = fingerprints_of(ph_texts, enc, disc)
    st_ph = sil_acc(F_ph, ph_labels)
    print(f'  甲乙丙丁: sil={st_ph["sil"]:.3f} acc_lda={st_ph["acc_lda"]:.3f}')
    if st_ph['sil'] > 0.3 or st_ph['acc_lda'] > 0.6:
        print('  → 甲乙丙丁有语义——更换随机拼音')
        ph_words = dict(zip(OPS.keys(), RANDOM_PH))
        ph_texts, ph_labels = make_structured(pairs, ph_words)
        F_ph = fingerprints_of(ph_texts, enc, disc)
        st_ph = sil_acc(F_ph, ph_labels)
        print(f'  随机拼音: sil={st_ph["sil"]:.3f} acc_lda={st_ph["acc_lda"]:.3f}')
    ph_ok = not (st_ph['sil'] > 0.3 or st_ph['acc_lda'] > 0.6)
    print(f'  占位词基线合格: {ph_ok}（sil={st_ph["sil"]:.3f} acc={st_ph["acc_lda"]:.3f}）')

    # ===== P1 真词四类 + C 分布 =====
    print('\nP1 真词四类:')
    true_texts, true_labels = make_structured(pairs, OPS)
    F_true = fingerprints_of(true_texts, enc, disc)
    # C 分布（评审 13）
    c_dist = {}
    for op_cn in OPS:
        cs = [c for _, _, c in pairs[OPS[op_cn]]]
        c_dist[op_cn] = {'min': min(cs), 'max': max(cs), 'mean': round(float(np.mean(cs)), 1)}
    print(f'  C 分布: {c_dist}')

    # ===== P2 定量 + Δ 增量 + 置换 =====
    print('\nP2 定量:')
    st_true = sil_acc(F_true, true_labels)
    print(f'  真词: sil={st_true["sil"]:.3f} acc_lda={st_true["acc_lda"]:.3f} '
          f'acc_knn={st_true["acc_knn"]:.3f} 中心比={st_true["center_ratio"]}')
    delta_sil = st_true['sil'] - st_ph['sil']
    delta_acc = st_true['acc_lda'] - st_ph['acc_lda']
    print(f'  Δsil={delta_sil:+.3f} Δacc={delta_acc:+.3f}')
    perm = permutation_test(F_true, true_labels, n_perm=1000)
    print(f'  置换检验: 真实 sil={perm["true_sil"]:.3f} 随机 p95={perm["p95"]:.3f} p={perm["p"]:.4f}')
    # M-A2（仅参考）
    fp = np.load(OUT / 'fp_matrix.npz')['fp']
    rows = json.loads((OUT / 'rows.json').read_text(encoding='utf-8'))
    from collections import defaultdict
    docs = defaultdict(list)
    for i, r in enumerate(rows):
        if r['source'] == 'bilingual_zh':
            docs[r['doc']].append(i)
    human = sorted([x for x in docs if x.startswith('ZH-H')], key=lambda x: len(docs[x]))
    ai = sorted([x for x in docs if x.startswith('ZH-A')], key=lambda x: len(docs[x]))
    h_idx = np.concatenate([np.array(docs[d]) for d in human])
    Fn = l2norm(F_true)
    Hn = l2norm(fp[h_idx])
    math_intra = float(np.mean([Fn[i] @ Fn[j] for i in range(len(Fn)) for j in range(i + 1, len(Fn))]))
    math_human_inter = float(np.mean(Fn @ Hn.T))
    m_a2 = math_intra > math_human_inter + 0.1
    print(f'  M-A2（仅参考）: 数学 intra={math_intra:.3f} vs 数学-人类 inter={math_human_inter:.3f}——'
          f'{"支持" if m_a2 else "不支持"}')

    # ===== P2.5 数字掩码口径（v0.85-2 修复——成本最低——消除数字表面特征）=====
    print('\nP2.5 数字掩码（所有数字→[NUM]——数字表面特征消除）:')
    mt_texts = [mask_digits(t) for t in true_texts]
    mp_texts = [mask_digits(t) for t in ph_texts]
    F_mt = fingerprints_of(mt_texts, enc, disc)
    F_mp = fingerprints_of(mp_texts, enc, disc)
    st_mt = sil_acc(F_mt, true_labels)
    st_mp = sil_acc(F_mp, ph_labels)
    print(f'  掩码真词: sil={st_mt["sil"]:.3f} acc_lda={st_mt["acc_lda"]:.3f} acc_knn={st_mt["acc_knn"]:.3f}')
    print(f'  掩码占位词: sil={st_mp["sil"]:.3f} acc_lda={st_mp["acc_lda"]:.3f}')
    delta_mt = st_mt['sil'] - st_mp['sil']
    delta_ma = st_mt['acc_lda'] - st_mp['acc_lda']
    print(f'  掩码 Δsil={delta_mt:+.3f} Δacc={delta_ma:+.3f}')
    perm_m = permutation_test(F_mt, true_labels, n_perm=1000)
    print(f'  掩码置换: 真实 sil={perm_m["true_sil"]:.3f} p={perm_m["p"]:.4f}')
    # 注意：全掩码使类内零方差（同模板同 [NUM]——每类 24 条相同——sil=1.0 数学必然——口径无效仅诊断）
    # 区间映射（v0.85-3 正确口径——保留类内多样性）
    bt_texts = [bin_digits(t) for t in true_texts]
    bp_texts = [bin_digits(t) for t in ph_texts]
    F_bt = fingerprints_of(bt_texts, enc, disc)
    F_bp = fingerprints_of(bp_texts, enc, disc)
    st_bt = sil_acc(F_bt, true_labels)
    st_bp = sil_acc(F_bp, ph_labels)
    print(f'  区间映射真词: sil={st_bt["sil"]:.3f} acc_lda={st_bt["acc_lda"]:.3f} acc_knn={st_bt["acc_knn"]:.3f}')
    print(f'  区间映射占位词: sil={st_bp["sil"]:.3f} acc_lda={st_bp["acc_lda"]:.3f}')
    delta_bt = st_bt['sil'] - st_bp['sil']
    delta_ba = st_bt['acc_lda'] - st_bp['acc_lda']
    print(f'  区间映射 Δsil={delta_bt:+.3f} Δacc={delta_ba:+.3f}')
    perm_b = permutation_test(F_bt, true_labels, n_perm=1000)
    print(f'  区间映射置换: 真实 sil={perm_b["true_sil"]:.3f} p={perm_b["p"]:.4f}')
    cond_b1 = (st_bt['sil'] > 0.25 and delta_bt > 0.10) or \
              (st_bt['acc_lda'] > 0.7 and delta_ba > 0.15)
    cond_b2 = perm_b['p'] < 0.05
    m_a1_binned = cond_b1 and cond_b2
    print(f'  区间映射 M-A1: {"PASS" if m_a1_binned else "FAIL"}（sil={st_bt["sil"]:.3f} Δsil={delta_bt:+.3f} '
          f'acc={st_bt["acc_lda"]:.3f} Δacc={delta_ba:+.3f} 置换 p={perm_b["p"]:.4f}）')

    # ===== 裁定 =====
    print('\n===== 裁定 =====')
    cond1 = (st_true['sil'] > 0.25 and delta_sil > 0.10) or \
            (st_true['acc_lda'] > 0.7 and delta_acc > 0.15)
    cond2 = perm['p'] < 0.05
    m_a1 = cond1 and cond2
    print(f'  M-A1: {"PASS" if m_a1 else "FAIL"}（sil={st_true["sil"]:.3f}>0.25 Δsil={delta_sil:+.3f}>0.10 '
          f'acc={st_true["acc_lda"]:.3f}>0.7 Δacc={delta_acc:+.3f}>0.15——置换 p={perm["p"]:.4f}）')
    overall = '成功（最低门槛）' if m_a1 else '否定'
    print(f'  总判定: {overall}')
    if m_a1:
        print('  结论措辞（评审 4 收敛）：在结构化转写条件下，真运算符词的分离度显著超过占位词基线，'
              '说明 BGE 对数学运算符的语义编码提供了额外可分离信息；这是最低门槛，'
              '不代表真实推导轨迹或运算规则本身被捕捉。')

    # ===== 图 3 张 =====
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2).fit(l2norm(np.vstack([F_true, F_ph])))
    Zt = pca.transform(l2norm(F_true))
    Zp = pca.transform(l2norm(F_ph))
    colors = ['#1f6fb2', '#e67e22', '#27ae60', '#8e44ad']
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, Z, lbl in ((axes[0], Zt, '真词'), (axes[1], Zp, '占位词')):
        for k in range(4):
            m = true_labels == k
            ax.scatter(Z[m, 0], Z[m, 1], s=40, color=colors[k],
                       label=list(OPS.keys())[k] if lbl == '真词' else None, alpha=0.8)
        ax.set_title(f'{lbl}四类（PCA 2D）')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        if lbl == '真词':
            ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PAPER / 'fig_math_pca.png', dpi=150)
    plt.close()

    # t-SNE（仅可视化——评审 7/10）
    try:
        from sklearn.manifold import TSNE
        fig2, ax2 = plt.subplots(figsize=(7, 6))
        T = TSNE(n_components=2, perplexity=10, random_state=0).fit_transform(l2norm(F_true))
        for k in range(4):
            m = true_labels == k
            ax2.scatter(T[m, 0], T[m, 1], s=40, color=colors[k], label=list(OPS.keys())[k], alpha=0.8)
        ax2.set_title('真词四类（t-SNE——仅可视化不判据）')
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(PAPER / 'fig_math_tsne.png', dpi=150)
        plt.close()
    except Exception as e:
        print(f'  t-SNE 失败: {str(e)[:60]}')

    # Δ 对比柱状
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    names = ['silhouette', 'LDA acc']
    vals_t = [st_true['sil'], st_true['acc_lda']]
    vals_p = [st_ph['sil'], st_ph['acc_lda']]
    x = np.arange(2)
    ax3.bar(x - 0.2, vals_t, 0.4, label='真词', color='#1f6fb2')
    ax3.bar(x + 0.2, vals_p, 0.4, label='占位词', color='#7f8c8d')
    for xi, (vt, vp) in enumerate(zip(vals_t, vals_p)):
        ax3.text(xi, max(vt, vp) + 0.02, f'Δ={vt - vp:+.2f}', ha='center', fontsize=9)
    ax3.set_xticks(x)
    ax3.set_xticklabels(names)
    ax3.set_ylabel('值')
    ax3.set_title('真词 vs 占位词基线（Δ=数学结构贡献——置换 p 标注）')
    ax3.legend(fontsize=9)
    ax3.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(PAPER / 'fig_math_delta.png', dpi=150)
    plt.close()

    # ===== 落盘 =====
    out = {
        'criteria': {'M-A1': bool(m_a1), 'M-A2_ref': bool(m_a2), 'overall': overall},
        'c_dist': c_dist,
        'placeholder': {'words': list(ph_words.values()), 'sil': st_ph['sil'],
                        'acc_lda': st_ph['acc_lda'], 'ok': bool(ph_ok)},
        'true_words': {'sil': st_true['sil'], 'acc_lda': st_true['acc_lda'],
                       'acc_knn': st_true['acc_knn'], 'center_ratio': st_true['center_ratio']},
        'delta': {'sil': round(float(delta_sil), 3), 'acc': round(float(delta_acc), 3)},
        'permutation': perm,
        'm_a2': {'math_intra': round(math_intra, 3),
                 'math_human_inter': round(math_human_inter, 3)},
        'verdict': overall,
        'masked': {'true_sil': st_mt['sil'], 'true_acc': st_mt['acc_lda'],
                   'ph_sil': st_mp['sil'], 'ph_acc': st_mp['acc_lda'],
                   'delta_sil': round(float(delta_mt), 3),
                   'delta_acc': round(float(delta_ma), 3),
                   'perm': perm_m, 'm_a1_masked': bool(m_a1_masked),
                   'note': '全掩码类内零方差——sil=1.0 数学必然——口径仅诊断'},
        'binned': {'true_sil': st_bt['sil'], 'true_acc': st_bt['acc_lda'],
                   'ph_sil': st_bp['sil'], 'ph_acc': st_bp['acc_lda'],
                   'delta_sil': round(float(delta_bt), 3),
                   'delta_acc': round(float(delta_ba), 3),
                   'perm': perm_b, 'm_a1_binned': bool(m_a1_binned)},
    }
    (OUT / 'math_geometry.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print('\n落盘 math_geometry.json + fig_math_*.png × 3 ✓')


if __name__ == '__main__':
    main()
