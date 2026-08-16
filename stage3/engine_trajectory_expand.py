# -*- coding: utf-8 -*-
"""v0.92 轨迹展开预实验（静态池化 → 轨迹函数——核心问题：BGE 512 句元级路径里有没有思想留下的动态痕迹）

【预注册判据表（先写后跑——用户评审 5 关键 + Plan 评审冻结）】

| # | 判据 | 判定线 | 角色 |
|---|------|--------|------|
| M-T0 | 复现门：corpus 73/113；M-I3 协议静态探针 ≈0.650±0.02；replay 边界匹配 100% | 前置门 |
| M-T1 | Δ_i = acc(观测序) − acc(段内乱序)（A_center——共享 25 配对）：Wilcoxon p<0.05 且配对保持置换 p<0.05（B=500） | 主判 |
| M-T2 | Δ_marg = acc(A_full 含均值) − acc(静态均值) ≥ 0.05；Δ_i≈0 而 Δ_marg>0 → 降级"仅静态异质性" | 增量门 |
| M-T3 | 同口径边界归因：Δ_i_neg = acc_neg(观测) − acc_neg(段内乱序)；Δ_b = acc_neg(段内乱序) − acc_neg(全乱序)——Δ_b>0 且 Δ_i_neg<Δ_b → "构造痕迹" | 归因守卫 |
| M-T4 | 签名臂：同 M-T1 口径 Δ_i（level-2 项）；L1-full 报告不判 | 副判 |
| M-T5 | 源探针（正样本内实践论 vs 矛盾论——A_center——5 折+簇置换 B=200）——高分离 → "源驱动"标记（**残余污染声明**） | 归因诊断 |
| M-T6 | 域调整：logits ~ {n_seg, 句元均长, chars} Ridge(α=10) 残差 d_res；<0.4 → 降级"学到规模" | 守卫 |
| M-T7 | 每类 acc + train-val gap（>0.15 标记）+ PCA EVR 旁报 | 可靠性 |

用户评审冻结：①正负乱序同质化（正样本 replay 恢复段落边界——段内乱序——恢复失败窗口乱序兜底）；
②配对非独立（25 配对同折内 seed 相关）——置换显式保持配对结构；③源效应残余污染声明（M-T5 不显著≠无污染）；
④签名臂 PCA 激进——前几主成分方差报告 + "不能完全排除高阶动态结构"声明；⑤Δ_b 与 Δ_i 同口径（负样本子集+A_center+25 配对）。

三态判定：动态痕迹存在 / 仅静态异质性或构造痕迹 / 无增量。
"""
import sys, json, os, re, time
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path('C:/Users/bai/Desktop/小说系统')
OUT = BASE / 'data' / 'dim_analysis'
PAPER = Path('C:/Users/bai/Desktop/AB系统论文储备')
SRC = BASE / 'data' / 'v090_sources'
sys.path.insert(0, str(BASE / 'stage3'))

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

SEED = 20260819
SEED_FOLDS = SEED + 1
SEED_SHUFFLE = SEED + 4
B_PERM = 500
B_SHUFFLE = 500
N_FOLDS = 5
N_SEEDS = 5
PCA_VAR = 0.95
SIG_LEVEL = 2
SIG_D = 8
SIG_D2 = 16
MIN_SEG = 6
LR_C = 1.0


def seg_texts_of(text):
    from subclause_structure import split_subclauses
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    paras = [l for l in lines if re.search(r'[一-龥]', l) and len(l) >= 30
             and not re.match(r'^第[一二三四五六七八九十百千0-9]+[章节卷回部]', l)]
    segs = []
    for p in paras:
        for c in split_subclauses(p):
            if len(c.strip()) >= 3:
                segs.append(c.strip())
    return segs


def cv_splits(y, n_folds=N_FOLDS, n_seeds=N_SEEDS):
    """25 组共享 (tr,va) 切分——全臂配对前提"""
    from sklearn.model_selection import StratifiedKFold
    splits = []
    for sd in range(n_seeds):
        skf = StratifiedKFold(n_folds, shuffle=True, random_state=SEED_FOLDS + sd)
        for tr, va in skf.split(np.zeros(len(y)), y):
            splits.append((tr, va))
    return splits


def load_corpus():
    d = json.loads((OUT / 'intent_corpus.json').read_text(encoding='utf-8'))
    pos = d['pos']
    neg = d['neg']
    assert len(pos) == 73 and len(neg) == 113
    return pos, neg


def replay_bounds():
    """确定性回放：正样本段落边界（acquire_pos 逻辑）+ 负样本拼接块边界（build_negative 逻辑）"""
    # 正样本段落边界：模拟 acquire_pos（buf += p——每行的字符累积）
    pos_bounds = []
    for fname in ('实践论.txt', '矛盾论.txt'):
        t = (SRC / fname).read_text(encoding='utf-8', errors='replace')
        lines = [l.strip() for l in t.split('\n') if l.strip()]
        paras = [l for l in lines if re.search(r'[一-龥]', l) and len(l) >= 30
                 and not (l.startswith('实践论') or l.startswith('矛盾论') or '一九三七年' in l[:20])]
        buf = ''
        for p in paras:
            buf += p
            if len(buf) >= 400:
                if len(buf) > 600:
                    cut = buf.rfind('。', 300, 600)
                    cut = cut if cut > 300 else 600
                    buf = buf[cut + 1:]
                else:
                    buf = ''
        # 段落归属：每块的组成行（按行累积）
    # 简化：正样本段落边界用"句元级边界"近似——每块的行数
    # 负样本拼接块边界：build_negative 的 rng 确定性回放
    import engine_intent_discriminator as eid
    rng = np.random.default_rng(20260819)
    pos_blocks = []  # 占位——replay 用
    cands = eid.build_negative([], [], rng)  # 同 rng 消耗路径（pos_blocks 参数未被使用——ph_lines 内部重读）
    # build_negative 用 ph_lines() 内部读取——与 v0.91 完全一致
    return cands


def replay_neg_pieces():
    """负样本拼接块边界：与 v0.91 build_negative 相同的 rng 消耗路径——返回每候选的块文本（→句元边界）"""
    import engine_intent_discriminator as eid
    from engine_intent_discriminator import ph_lines
    rng = np.random.default_rng(SEED)  # 与 v0.91 main 相同——main 中 rng 只被 build_negative 消耗
    ph = ph_lines()
    pieces_info = []
    cands = []
    for _ in range(400):
        k = int(rng.choice([2, 3, 4]))
        idx = rng.choice(len(ph), k, replace=False)
        pieces = [ph[i] for i in idx]
        rng.shuffle(pieces)
        out = pieces[0]
        for p_ in pieces[1:]:
            conj = rng.choice(eid.CONJ_WORDS)
            out += '。' + conj + '，' + p_
        if 380 <= len(out) <= 650:
            cands.append(out)
            pieces_info.append(pieces)
    return cands, pieces_info


def bounds_to_seg_idx(pieces, text):
    """块文本 → 句元边界（lo, hi) 索引对——按块字符比例分配句元（用户关键 1 同质化）"""
    segs = seg_texts_of(text)
    n = len(segs)
    if not pieces or n < 2:
        return [(0, n)]
    lens = [len(p) for p in pieces]
    total = sum(lens)
    bounds = []
    start = 0
    for L in lens[:-1]:
        frac = L / total
        end = start + max(1, int(round(frac * n)))
        end = min(end, n - (len(lens) - len(bounds) - 1))
        bounds.append((start, end))
        start = end
    bounds.append((start, n))
    return [(lo, hi) for lo, hi in bounds if hi > lo]


def replay_pos_bounds():
    """正样本段落边界：模拟 acquire_pos 行累积（确定性）——存储文本与组成行拼接精确匹配——恢复段落→句元边界"""
    corpus = json.loads((OUT / 'intent_corpus.json').read_text(encoding='utf-8'))
    stored = [p['text'] for p in corpus['pos']]
    # 与 acquire_pos 相同的行来源（过滤同口径）
    lines = []
    for fname in ('实践论.txt', '矛盾论.txt'):
        t = (SRC / fname).read_text(encoding='utf-8', errors='replace')
        for l in t.split('\n'):
            l = l.strip()
            if len(l) >= 30 and re.search(r'[一-龥]', l) \
                    and not (l.startswith('实践论') or l.startswith('矛盾论') or '一九三七年' in l[:20]):
                lines.append(l)
    joined = ''.join(lines)
    pos_bounds = []
    n_match = 0
    for b in stored:
        pos = joined.find(b[:50])
        if pos >= 0:
            # 定位组成行（字符累积区间重叠）
            sel = []
            acc = 0
            for l in lines:
                if acc < pos + len(b) and acc + len(l) > pos:
                    sel.append(l)
                acc += len(l)
            # 行→句元边界
            seg_bounds = []
            s = 0
            for l in sel:
                ns = len(seg_texts_of(l))
                if ns > 0:
                    seg_bounds.append((s, s + ns))
                    s += ns
            pos_bounds.append(seg_bounds if seg_bounds else [(0, s)])
            n_match += 1
        else:
            pos_bounds.append([(0, len(seg_texts_of(b)))])
    print(f'  正样本 replay 匹配: {n_match}/{len(stored)}——多段块: '
          f'{sum(1 for b in pos_bounds if len(b) > 1)}/{len(stored)}')
    return pos_bounds


def encode_sent_bge(texts):
    """句元级 bge 512（重编码——落盘 intent_sent_bge.npz——存在跳过）"""
    import os
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    f = OUT / 'intent_sent_bge.npz'
    if f.exists():
        z = np.load(f, allow_pickle=True)
        return z['seqs']
    from para_dimensions import load_models
    enc, _ = load_models('cpu')
    seqs = []
    for t in texts:
        segs = seg_texts_of(t)
        if not segs:
            seqs.append(np.zeros((0, 512), np.float32))
            continue
        sv = enc.encode(segs, normalize_embeddings=True, batch_size=16,
                        show_progress_bar=False, device='cpu').astype(np.float32)
        seqs.append(sv)
    arr = np.empty(len(seqs), dtype=object)
    arr[:] = seqs
    np.savez(f, seqs=arr)
    return arr


def traj_stats(seqs, mode='center'):
    """展开 A：8 统计量 × 512（去均值块——用户评审）——NaN 折叠中位插补"""
    n = len(seqs)
    Feat = np.zeros((n, 7 * 512))
    for i, s in enumerate(seqs):
        if len(s) < MIN_SEG:
            Feat[i] = np.nan
            continue
        m = s.mean(0)
        v = s.var(0)
        d = np.diff(s, axis=0)
        lag1 = np.array([np.corrcoef(s[:-1, k], s[1:, k])[0, 1] if s[:, k].std() > 1e-9 else 0
                         for k in range(512)])
        lag2 = np.array([np.corrcoef(s[:-2, k], s[2:, k])[0, 1] if s[:, k].std() > 1e-9 else 0
                         for k in range(512)])
        mad = np.abs(d).mean(0)
        rng_v = s.max(0) - s.min(0)
        # zcross（去均值过零）
        zc = s - m
        zcross = np.mean((zc[:-1] * zc[1:]) < 0, axis=0)
        # trend_R²（时间线性回归解释方差比）
        tt = np.arange(len(s))
        trend = np.polyfit(tt, s, 1)
        fit = np.outer(tt, trend[0]) + trend[1]
        ss_tot = ((s - m) ** 2).sum(0) + 1e-9
        ss_res = ((s - fit) ** 2).sum(0)
        trend_r2 = 1 - ss_res / ss_tot
        Feat[i] = np.concatenate([v, lag1, lag2, mad, rng_v, zcross, trend_r2])
    # NaN 插补（折叠中位）
    col_med = np.nanmedian(Feat, axis=0)
    for k in range(Feat.shape[1]):
        mask = np.isnan(Feat[:, k])
        if mask.sum() / n > 0.10:
            Feat = np.delete(Feat, k, axis=1)
        else:
            Feat[mask, k] = col_med[k]
    return Feat


def projected_dynamics(seqs, n_pc=2):
    """投影动力学块：池化句元 PCA top-2 PC 的时间序列 × 5 统计量"""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    pool = np.vstack([s for s in seqs if len(s) >= MIN_SEG])
    pca = PCA(n_components=n_pc).fit(StandardScaler().fit_transform(pool))
    out = np.zeros((len(seqs), n_pc * 5))
    for i, s in enumerate(seqs):
        if len(s) < MIN_SEG:
            out[i] = np.nan
            continue
        Z = pca.transform(StandardScaler().fit_transform(s))
        for c in range(n_pc):
            z = Z[:, c]
            d = np.diff(z)
            lag1 = np.corrcoef(z[:-1], z[1:])[0, 1] if z.std() > 1e-9 else 0
            td = sum((d[:-1] > 0) & (d[1:] <= 0)) / (len(z) - 2) if len(z) > 2 else 0
            zc = np.mean((z[:-1] * z[1:]) < 0)
            tt = np.arange(len(z))
            trend = np.polyfit(tt, z, 1)
            fit = tt * trend[0] + trend[1]
            tr2 = 1 - ((z - fit) ** 2).sum() / ((z - z.mean()) ** 2).sum() if (z - z.mean()).std() > 1e-9 else 0
            out[i, c * 5:(c + 1) * 5] = [np.abs(d).mean(), lag1, td, zc, tr2]
    cm = np.nanmedian(out, axis=0)
    for k in range(out.shape[1]):
        msk = np.isnan(out[:, k])
        if msk.sum() / len(seqs) > 0.10:
            out = np.delete(out, k, axis=1)
        else:
            out[msk, k] = cm[k]
    return out, pca


def shuffle_sequences(seqs, bounds, kind, rng):
    """乱序：'full' 全乱序 / 'within' 段内乱序（正样本按 replay 段落、负样本按拼接块——同质化）"""
    out = []
    for i, s in enumerate(seqs):
        n = len(s)
        if n < MIN_SEG:
            out.append(s)
            continue
        if kind == 'full':
            idx = rng.permutation(n)
        else:
            b = bounds[i]
            idx = []
            for lo, hi in b:
                lo = max(0, min(lo, n))
                hi = max(lo, min(hi, n))
                if hi <= lo:
                    continue
                seg = list(range(lo, hi))
                rng.shuffle(seg)
                idx += seg
            # 钳制后补齐（边界含 50 字符匹配误差）
            if len(idx) < n:
                rest = [k for k in range(n) if k not in set(idx)]
                rng.shuffle(rest)
                idx += rest
            idx = idx[:n]
        out.append(s[idx])
    return out


def probe_lr(X, y, splits):
    """共享切分 25 evals——acc + 每类 acc + gap"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    accs, accs_pos, accs_neg, gaps = [], [], [], []
    for tr, va in splits:
        sc = StandardScaler().fit(X[tr])
        pca = PCA(n_components=PCA_VAR).fit(sc.transform(X[tr]))
        Xtr = pca.transform(sc.transform(X[tr]))
        Xva = pca.transform(sc.transform(X[va]))
        clf = LogisticRegression(C=LR_C, max_iter=2000).fit(Xtr, y[tr])
        pred = clf.predict(Xva)
        accs.append((pred == y[va]).mean())
        m_pos = y[va] == 1
        accs_pos.append((pred[m_pos] == 1).mean() if m_pos.sum() else np.nan)
        m_neg = y[va] == 0
        accs_neg.append((pred[m_neg] == 0).mean() if m_neg.sum() else np.nan)
        tr_pred = clf.predict(Xtr)
        gaps.append((tr_pred == y[tr]).mean() - (pred == y[va]).mean())
    return {'acc_mean': float(np.mean(accs)), 'acc_pos': float(np.nanmean(accs_pos)),
            'acc_neg': float(np.nanmean(accs_neg)), 'gap': float(np.mean(gaps)),
            'per_eval': accs}


def shuffle_null_paired(X_obs, X_shuf, y, splits, n_perm=B_SHUFFLE, seed=SEED_SHUFFLE):
    """配对保持的乱序置换（用户关键 2）——**预计算每折 PCA/标准化（置换外）——循环内只 LR 拟合
    （优化：25000 次拟合从数小时降到 ~2 分钟——用户提性能）**——置换 p = P(Δ_perm ≥ Δ_obs)"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    rng = np.random.default_rng(seed)
    # 预计算：每折的 (obs 变换, shuf 变换)——tr/va 特征矩阵
    prep = []
    for tr, va in splits:
        row = {}
        for name, X in (('o', X_obs), ('s', X_shuf)):
            sc = StandardScaler().fit(X[tr])
            pca = PCA(n_components=PCA_VAR).fit(sc.transform(X[tr]))
            row[f'{name}_tr'] = pca.transform(sc.transform(X[tr]))
            row[f'{name}_va'] = pca.transform(sc.transform(X[va]))
        prep.append(row)
    ytrs = [y[tr] for tr, _ in splits]
    yvas = [y[va] for _, va in splits]

    def eval_fold(row, ytr, yva, use_obs):
        A = row['o_tr' if use_obs else 's_tr']
        B = row['o_va' if use_obs else 's_va']
        clf = LogisticRegression(C=LR_C, max_iter=2000).fit(A, ytr)
        return (clf.predict(B) == yva).mean()

    obs_delta = float(np.mean([eval_fold(prep[i], ytrs[i], yvas[i], True) -
                               eval_fold(prep[i], ytrs[i], yvas[i], False)
                               for i in range(len(splits))]))
    nulls = []
    for _ in range(n_perm):
        swap = rng.random(len(splits)) < 0.5
        d_accs = [eval_fold(prep[i], ytrs[i], yvas[i], not swap[i]) -
                  eval_fold(prep[i], ytrs[i], yvas[i], swap[i])
                  for i in range(len(splits))]
        nulls.append(np.mean(d_accs))
    nulls = np.array(nulls)
    p = (1 + int(np.sum(nulls >= obs_delta))) / (1 + n_perm)
    return {'obs_delta': obs_delta, 'p': float(p), 'null_mean': float(np.mean(nulls)),
            'null_p95': float(np.percentile(nulls, 95))}


def main():
    print('===== v0.92 轨迹展开预实验（判据预注册——见文件头） =====')
    os.environ.setdefault('HF_HUB_OFFLINE', '1')

    # ===== P0 复现门 + 编码 + replay =====
    print('\nP0 复现门:')
    pos, neg = load_corpus()
    texts = [p['text'] for p in pos] + [n['text'] for n in neg]
    y = np.array([1] * 73 + [0] * 113)
    srcs = [p['src'] for p in pos] + ['拼接'] * 113
    print(f'  corpus 73/113 ✓——y {y.sum()}/{len(y) - y.sum()}')
    # 静态基线（M-I3 协议——random_state=0 单 seed——≈0.650±0.02）
    z = np.load(OUT / 'intent_embeddings.npz')
    X_static = z['X']
    from sklearn.model_selection import StratifiedKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    accs = []
    for tr, va in skf.split(X_static, y):
        sc = StandardScaler().fit(X_static[tr])
        clf = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(X_static[tr]), y[tr])
        accs.append((clf.predict(sc.transform(X_static[va])) == y[va]).mean())
    static_mi3 = float(np.mean(accs))
    assert abs(static_mi3 - 0.650) < 0.02, f'复现门静态探针失败: {static_mi3}'
    print(f'  静态基线（M-I3 协议）: {static_mi3:.4f}（≈0.650 ✓）')

    print('\nP0 句元级编码（~6300 句元——CPU）:')
    t0 = time.time()
    seqs = encode_sent_bge(texts)
    n_sent = sum(len(s) for s in seqs)
    print(f'  编码完成（{time.time() - t0:.0f}s）——句元 {n_sent}——seqs {len(seqs)}')
    # 排除 n_seg<6
    keep = [i for i, s in enumerate(seqs) if len(s) >= MIN_SEG]
    n_excl = len(seqs) - len(keep)
    print(f'  排除 n_seg<{MIN_SEG}: {n_excl} 条')
    keep_arr = np.array(keep)

    print('\nP0 replay 边界:')
    cands, pieces_info = replay_neg_pieces()
    neg_texts = [n['text'] for n in neg]
    cand_map = {}
    for ci, c in enumerate(cands):
        cand_map.setdefault(c, []).append(ci)
    bounds_neg = []
    n_match = 0
    for i, t in enumerate(neg_texts):
        if t in cand_map:
            ci = cand_map[t][0]
            b = bounds_to_seg_idx(pieces_info[ci], t)
            bounds_neg.append(b)
            n_match += 1
        else:
            bounds_neg.append([(0, len(seg_texts_of(t)))])
    print(f'  负样本 replay 匹配: {n_match}/113')
    assert n_match == 113, 'replay 匹配失败'
    bounds_pos = replay_pos_bounds()
    print(f'  正样本段落边界: {sum(1 for b in bounds_pos if len(b) > 1)}/{73} 条多段（段内乱序有效）')

    # ===== P1 展开 A 三阶梯 =====
    print('\nP1 展开 A（8×512 去均值块 + 投影动力学）:')
    t1 = time.time()
    A_center = traj_stats(seqs)
    pdyn, pca_pool = projected_dynamics(seqs)
    A_center = np.hstack([A_center, pdyn])
    A_full = np.hstack([X_static, A_center])  # 含均值（M-T2 增量门用）
    print(f'  A_center {A_center.shape}——A_full {A_full.shape}（{time.time() - t1:.0f}s）')
    # 三阶梯
    splits = cv_splits(y)
    rng_sh = np.random.default_rng(SEED_SHUFFLE)
    # rung1：段内乱序（正样本按 replay 段落、负样本按拼接块——同质化——用户关键 1）
    bounds = bounds_pos + bounds_neg
    seqs_shuf_within = shuffle_sequences(seqs, bounds, 'within', rng_sh)
    A_within = traj_stats(seqs_shuf_within)
    pdyn_w, _ = projected_dynamics(seqs_shuf_within)
    A_within = np.hstack([A_within, pdyn_w])
    seqs_shuf_full = shuffle_sequences(seqs, bounds, 'full', rng_sh)
    A_full_shuf = traj_stats(seqs_shuf_full)
    pdyn_f, _ = projected_dynamics(seqs_shuf_full)
    A_full_shuf = np.hstack([A_full_shuf, pdyn_f])
    print('  三阶梯特征就绪——25 配对探针:')
    yk = y[keep_arr]
    splits_k = [(np.flatnonzero(np.isin(keep_arr, tr)), np.flatnonzero(np.isin(keep_arr, va)))
                for tr, va in splits]
    splits_k = [(tr, va) for tr, va in splits_k if len(tr) > 10 and len(va) > 5]
    r_obs = probe_lr(A_center[keep_arr], yk, splits_k)
    r_within = probe_lr(A_within[keep_arr], yk, splits_k)
    r_static = probe_lr(X_static[keep_arr], yk, splits_k)
    r_full_shuf = probe_lr(A_full_shuf[keep_arr], yk, splits_k)
    splits = splits_k
    print(f'  rung0 静态: {r_static["acc_mean"]:.4f}——rung1 段内乱序: {r_within["acc_mean"]:.4f}——'
          f'rung2 观测序: {r_obs["acc_mean"]:.4f}——全乱序: {r_full_shuf["acc_mean"]:.4f}')
    delta_i = r_obs['acc_mean'] - r_within['acc_mean']
    delta_marg = r_obs['acc_mean'] - r_static['acc_mean']
    print(f'  Δ_i={delta_i:+.4f}（主判）——Δ_marg={delta_marg:+.4f}（实用门 ≥0.05）')

    # 配对置换（用户关键 2——保持配对结构）
    print('\n  配对保持置换（B=500——用户关键 2）:')
    perm = shuffle_null_paired(A_center[keep_arr], A_within[keep_arr], yk, splits_k)
    print(f'  Δ_i 置换: obs={perm["obs_delta"]:.4f} p={perm["p"]:.4f} null mean={perm["null_mean"]:.4f} '
          f'p95={perm["null_p95"]:.4f}')

    # M-T3 同口径边界归因（负样本子集——keep 后位置索引）
    print('\n  M-T3 边界归因（负样本子集——同口径）:')
    neg_idx = np.array([i for i, k in enumerate(keep_arr) if k >= 73])
    new_pos = {int(o): n for n, o in enumerate(neg_idx)}
    splits_neg = [([new_pos[x] for x in tr if x in new_pos],
                   [new_pos[x] for x in va if x in new_pos])
                  for tr, va in splits
                  if sum(1 for x in va if x in new_pos) > 0]
    splits_neg = [(np.array(a), np.array(b)) for a, b in splits_neg]
    if len(splits_neg) < 5:
        print('  负样本子集折过少——Δ_b 不可算——标记')
        delta_i_neg = delta_b = None
    else:
        r_obs_neg = probe_lr(A_center[neg_idx], y[neg_idx], splits_neg)
        r_within_neg = probe_lr(A_within[neg_idx], y[neg_idx], splits_neg)
        r_fullshuf_neg = probe_lr(A_full_shuf[neg_idx], y[neg_idx], splits_neg)
        delta_i_neg = r_obs_neg['acc_mean'] - r_within_neg['acc_mean']
        delta_b = r_within_neg['acc_mean'] - r_fullshuf_neg['acc_mean']
        print(f'  Δ_i_neg={delta_i_neg:+.4f}——Δ_b={delta_b:+.4f}')

    # ===== 判定 =====
    from scipy.stats import wilcoxon
    wi = wilcoxon(np.array(r_obs['per_eval']) - np.array(r_within['per_eval']))
    m_t1 = perm['p'] < 0.05 and wi.pvalue < 0.05
    m_t2 = delta_marg >= 0.05
    m_t3 = (delta_b is not None and delta_b > 0 and delta_i_neg < delta_b)
    if not m_t1 and m_t2 and abs(delta_i) < 0.01:
        overall = '仅静态异质性'
    elif m_t3:
        overall = '构造痕迹（拼接边界）'
    elif not m_t1 and not m_t2:
        overall = '无增量'
    elif m_t1:
        overall = '动态痕迹存在'
    else:
        overall = '仅静态异质性'
    print(f'\n  M-T1: {"PASS" if m_t1 else "FAIL"}（Δ_i={delta_i:+.4f} Wilcoxon p={wi.pvalue:.4f} 置换 p={perm["p"]:.4f}）')
    print(f'  M-T2: {"PASS" if m_t2 else "FAIL"}（Δ_marg={delta_marg:+.4f}）')
    print(f'  M-T3: {"触发" if m_t3 else "未触发"}')
    print(f'  总判定: {overall}')

    # 落盘
    res = {'meta': {'seed': SEED, 'n_pos': 73, 'n_neg': 113},
           'static_mi3': static_mi3,
           'rung': {'static': r_static['acc_mean'], 'within': r_within['acc_mean'],
                    'obs': r_obs['acc_mean'], 'full_shuf': r_full_shuf['acc_mean'],
                    'delta_i': delta_i, 'delta_marg': delta_marg},
           'perm': perm, 'wilcoxon_p': float(wi.pvalue),
           'boundary': {'delta_i_neg': delta_i_neg, 'delta_b': delta_b, 'n_splits_neg': len(splits_neg)},
           'per_class': {'pos': r_obs['acc_pos'], 'neg': r_obs['acc_neg']},
           'gap': r_obs['gap'],
           'verdict': overall}
    (OUT / 'intent_traj_expand.json').write_text(json.dumps(res, ensure_ascii=False, indent=1),
                                                 encoding='utf-8')
    print('落盘 intent_traj_expand.json ✓')


if __name__ == '__main__':
    main()
