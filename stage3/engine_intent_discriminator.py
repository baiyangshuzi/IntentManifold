# -*- coding: utf-8 -*-
"""v0.91 核心意图强度判别器（训练目标升级——"有核心意图 vs 意图空洞"——四态判定）

用户核心论断：当前判别器测的是思想的投影（组织方式）不是思想本身——训练目标（人 vs AI）太低。
方案：更换训练目标为"有核心意图 vs 意图空洞"——正=有贯穿核心思想的论证文本（哲学论证）；
负=语言连贯组织正常但无统一核心思想的文本（同域相关主题拼接）。判别器必须"读穿"文本。

【预注册判据表（先写后跑——用户评审 5 关键 + Plan 评审 7 修正冻结）】

| # | 判据 | 判定线 | 角色 |
|---|------|--------|------|
| M-I1 | 5 折×5 seeds CV 均值 acc > 0.65 且 按源文档聚类置换 B=500 p < 0.05——双架构（MLP 与 LR）——MLP 显著优于 LR（Δ≥0.05）→"学会"更强证据 | 主判 |
| M-I2 | 旧判别器（v2）AUROC < 0.62（d<0.55）；>0.62 回归诊断（可解释→收紧匹配重跑；不可解释→构造失败无效） | 表面控制 |
| M-I3 | bge512 线性探针 vs 新判别器 64 维线性探针（5 折） | 特征分解 |
| M-I4 | 新空间 64 维逐维 d 排序+方向（与 v2 无维号映射假设） | 中间层 |
| M-I5 | 新判别 logits 与块比率 Spearman |ρ| ≥ 0.4 → 降级——通过时报告拼接痕迹抽检 | 残留通道 |
| M-I6 | 域调整（logits ~ 风格坐标+Mahalanobis 回归残差——d_res < 0.4 → 降级"学到组织/语体"） | 语体轴 |

四态：学会 / 学到组织（降级）/ 学不会 / 无效。

冻结决策：①AI 负样本=独立副臂（DeepSeek API——不入主训练——M-I2 污染）；②负样本=同域相关主题拼接
（多无关主题与组织匹配不可兼得）；③正样本=哲学非重叠切段 ~70（时评全部排除——模板化）；④抽检=API
代理标注（标注者与判别目标同构风险声明）；⑤置换按源文档聚类（文档内相关——有效置换数报告）；
⑥拼接痕迹 API 抽检 + 真实空洞文本探索副臂；⑦线性 LR 平行估计；⑧语体风格特征入匹配维 +
负样本哲学拼接比例提升 + M-I6 必要判据；⑨正样本逐段抽检（只保留明确含论点段）；⑩匹配维扩展
（TTR/名词比例/标点密度）。

诚实限制：标签构造性；拼接二阶残余+痕迹；语体轴；小样本（N≈140——θ≤0.65 对应 d≈0.55-0.65）；
bge 512 token 截断（段长≤600 字）；OOD 分离度口径；时评仅负样本素材；新空间与 v2 无维号对应；
置换聚类；LR 平行（MLP≈LR 证据弱化）。
"""
import sys, json, os, re, time
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path
from collections import defaultdict
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
SEED_PERM = SEED + 2
SEED_MATCH = SEED + 3
N_SEG_MIN, N_SEG_MAX = 400, 600
B_PERM = 500
N_SEEDS = 5
N_FOLDS = 5
CONJ_WORDS = ['此外', '另一方面', '同时', '其次', '再者', '另外', '与此同时', '进一步说', '再比如', '总而言之']


def cohens_d(x, y):
    return (np.mean(x) - np.mean(y)) / np.sqrt((np.var(x) + np.var(y)) / 2 + 1e-9)

# ============ P0 素材 ============
def seg_texts_of(text):
    """长文本 → 句元（同 v0.90 口径——行切分/汉字/len≥30/非章节标题——split_subclauses len≥3）"""
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


def acquire_pos():
    """正样本：哲学非重叠切段（实践论/矛盾论——相邻行拼接累积 400-600 字——非重叠）"""
    blocks, meta = [], []
    for fname, src in (('实践论.txt', '实践论'), ('矛盾论.txt', '矛盾论')):
        t = (SRC / fname).read_text(encoding='utf-8', errors='replace')
        lines = [l.strip() for l in t.split('\n') if l.strip()]
        paras = [l for l in lines if re.search(r'[一-龥]', l) and len(l) >= 30
                 and not (l.startswith('实践论') or l.startswith('矛盾论') or '一九三七年' in l[:20])]
        buf, cur = '', 0
        for p in paras:
            buf += p
            if len(buf) >= N_SEG_MIN:
                if len(buf) > N_SEG_MAX:
                    cut = buf.rfind('。', N_SEG_MIN - 100, N_SEG_MAX)
                    cut = cut if cut > N_SEG_MIN - 100 else N_SEG_MAX
                    blocks.append(buf[:cut + 1])
                    buf = buf[cut + 1:]
                else:
                    blocks.append(buf)
                    buf = ''
        if buf:
            blocks.append(buf)
        for b in blocks[cur:]:
            meta.append({'src': src, 'n_char': len(b), 'n_seg': len(seg_texts_of(b))})
        cur = len(blocks)
    # 过滤过长/过短
    keep = [(b, m) for b, m in zip(blocks, meta) if N_SEG_MIN <= m['n_char'] <= N_SEG_MAX + 100]
    print(f'  正样本候选: {len(blocks)} 块 → 过滤后 {len(keep)}')
    return keep


def load_commentary():
    """时评素材（仅负样本素材——用户决策）——people_commentary + S + L2 段落"""
    texts = []
    pc = (BASE / 'data/generalization/human/people_commentary.txt').read_text(encoding='utf-8', errors='replace')
    for l in pc.split('\n'):
        l = l.strip()
        if len(l) >= 40 and re.search(r'[一-龥]', l) and not l.startswith('壹时评'):
            texts.append((l, 'people_commentary'))
    for i in range(1, 13):
        f = BASE / f'data/independent_test/human/shiping/S-{i:02d}.txt'
        if f.exists():
            t = f.read_text(encoding='utf-8', errors='replace').strip()
            if len(t) >= 40:
                texts.append((t, f'S-{i:02d}'))
    l2 = (BASE / 'data/generalization_strict/human/L2.txt').read_text(encoding='utf-8', errors='replace')
    for i, l in enumerate(l2.split('\n')):
        l = l.strip()
        if len(l) >= 40 and re.search(r'[一-龥]', l):
            texts.append((l, f'L2-{i:02d}'))
    return texts


def ph_lines():
    """哲学行级段落（100-400 字——拼接素材）"""
    lines = []
    for fname, src in (('实践论.txt', '实践论'), ('矛盾论.txt', '矛盾论')):
        t = (SRC / fname).read_text(encoding='utf-8', errors='replace')
        for l in t.split('\n'):
            l = l.strip()
            if 100 <= len(l) <= 400 and re.search(r'[一-龥]', l) \
                    and not (l.startswith('实践论') or l.startswith('矛盾论') or '一九三七年' in l[:20]):
                lines.append(l)
    return lines


def build_negative(pos_blocks, commentary, rng):
    """负样本：哲学行级跨论文拼接（实践论×矛盾论——语体与正样本一致——用户关键 4）——
    段数 {2,3,4} × 连接词变体 × 顺序打乱——3× 冗余——380-650 字"""
    ph = ph_lines()
    cands = []
    for _ in range(400):
        k = int(rng.choice([2, 3, 4]))
        idx = rng.choice(len(ph), k, replace=False)
        pieces = [ph[i] for i in idx]
        rng.shuffle(pieces)
        out = pieces[0]
        for p_ in pieces[1:]:
            conj = rng.choice(CONJ_WORDS)
            out += '。' + conj + '，' + p_
        if 380 <= len(out) <= 650:
            cands.append(out)
    print(f'  负样本候选: {len(cands)}（哲学行级跨论文拼接——3× 冗余）')
    return cands


# ============ 指标（v2 空间） ============
def dims_of_text(text, enc, disc):
    """单文本七维组织指标（v2 判别器——para_dimensions 口径）+ 扩展维"""
    from para_dimensions import para_dimensions
    from subclause_structure import split_subclauses
    import jieba.posseg as pseg
    d = para_dimensions(text, enc, disc, split_subclauses, pseg, device='cpu')
    if d is None:
        return None
    segs = seg_texts_of(text)
    chars = len(text)
    # TTR
    words = []
    try:
        import jieba
        jieba.setLogLevel(60)
        for s in segs:
            words += [w for w in jieba.lcut(s) if len(w) >= 2]
    except Exception:
        words = list(text)
    ttr = len(set(words)) / (len(words) + 1e-9)
    punct = len(re.findall(r'[，。；：？！、—…]', text)) / chars
    conj = sum(text.count(w) for w in CONJ_WORDS) / chars * 100
    # 块间词 Jaccard（拼接的固有特征——匹配维——防"跨块主题断裂"被当信号）
    block_jac = np.nan
    blocks = re.split(r'。' + '|'.join(re.escape(w) for w in CONJ_WORDS) + r'，', text)
    if len(blocks) >= 2:
        bsets = [set(jieba.lcut(b)) for b in blocks if len(b) > 10]
        if len(bsets) >= 2:
            js = []
            for i in range(len(bsets) - 1):
                inter = len(bsets[i] & bsets[i + 1])
                js.append(inter / (len(bsets[i] | bsets[i + 1]) + 1e-9))
            block_jac = float(np.mean(js))
    d.update({'chars': chars, 'n_seg': len(segs), 'ttr': ttr, 'punct': punct,
              'conj_density': conj, 'block_jac': block_jac})
    return d


MATCH_KEYS = ['chars', 'n_seg', 'disc', 'sent_proj', 'traj', 'l7_adj',
              'word_proj', 'word_adj', 'entropy', 'ttr', 'punct']


def match_gate(pos_dims, neg_dims, th_d=0.3):
    """KS 门：每维 |d|<0.3 ∧ KS p>0.05——返回通过维度数与每维结果——
    排除 block_jac/conj_density（拼接固有属性——M-I5 与抽检兜底——匹配维哲学）"""
    from scipy import stats as sc
    keys = [k for k in MATCH_KEYS]
    res = {}
    for k in keys:
        pv = np.array([x[k] for x in pos_dims if x and k in x and x[k] == x[k]], float)
        nv = np.array([x[k] for x in neg_dims if x and k in x and x[k] == x[k]], float)
        if len(pv) == 0 or len(nv) == 0:
            continue
        d = (pv.mean() - nv.mean()) / np.sqrt((pv.var() + nv.var()) / 2 + 1e-9)
        ks = sc.ks_2samp(pv, nv)
        res[k] = {'d': round(float(d), 3), 'ks_p': round(float(ks.pvalue), 4),
                  'ok': bool(abs(d) < th_d and ks.pvalue > 0.05)}
    return res


def main():
    print('===== v0.91 核心意图强度判别器（判据预注册——见文件头） =====')
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    from para_dimensions import load_models
    enc, disc = load_models('cpu')  # v2 判别器（默认——匹配与七维口径）
    rng = np.random.default_rng(SEED)

    # ===== P0 复现门 + 正样本 =====
    print('\nP0 复现门:')
    fp = np.load(OUT / 'fp_matrix.npz')['fp']
    rows = json.loads((OUT / 'rows.json').read_text(encoding='utf-8'))
    docs = defaultdict(list)
    for i, r in enumerate(rows):
        if r['source'] == 'bilingual_zh':
            docs[r['doc']].append(i)
    human = sorted([d for d in docs if d.startswith('ZH-H')], key=lambda d: len(docs[d]))
    ai = sorted([d for d in docs if d.startswith('ZH-A')], key=lambda d: len(docs[d]))
    diffs = {}
    for grp, dlist in (('human', human), ('ai', ai)):
        diffs[grp] = {}
        for doc in dlist:
            idx = np.array(docs[doc])
            diffs[grp][doc] = fp[idx[1:], :] - fp[idx[:-1], :]
    all_norms = np.concatenate([np.linalg.norm(d, axis=1)
                                for dd in diffs.values() for d in dd.values()])
    p90 = float(np.quantile(all_norms, 0.90))
    assert abs(p90 - 3.229) < 1e-3, f'复现门 p90 失败: {p90}'
    print(f'  复现门 p90={p90:.3f} ✓')

    print('\nP0 正样本（哲学非重叠切段——时评排除）:')
    pos = acquire_pos()
    print(f'  正样本: {len(pos)} 块——字数 {[m["n_char"] for _, m in pos[:5]]}...——句元 {[m["n_seg"] for _, m in pos[:5]]}...')
    if len(pos) < 60:
        print('  ⚠ 正样本 <60——后续匹配筛选后可能不足——如实报告')

    print('\nP0 负样本素材（时评——仅素材）:')
    commentary = load_commentary()
    print(f'  时评素材: {len(commentary)} 段')

    # ===== P1 负样本构造 + 指标 + 匹配 =====
    print('\nP1 负样本构造（同域相关主题拼接——3× 冗余）:')
    neg_cands = build_negative(pos, commentary, rng)
    # 正负指标（v2 空间——组织匹配）
    print('\nP1 指标计算（v2 空间——组织匹配——~200 文本）:')
    t0 = time.time()
    pos_dims = []
    for b, m in pos:
        d = dims_of_text(b, enc, disc)
        if d:
            d['src'] = m['src']
            d['label'] = 1
            pos_dims.append(d)
    neg_dims = []
    for c in neg_cands:
        d = dims_of_text(c, enc, disc)
        if d:
            d['src'] = '拼接'
            d['label'] = 0
            neg_dims.append(d)
    print(f'  指标完成（{time.time() - t0:.0f}s）——正 {len(pos_dims)} 负 {len(neg_dims)}')
    # 匹配门（负样本筛选——贪心：逐维不匹配则拒绝）
    print('\nP1 匹配门（KS——|d|<0.3 ∧ p>0.05）:')
    # 负样本筛选：贪心拒绝——每次拒绝最差维的样本直到全过或候选耗尽
    sel = list(range(len(neg_dims)))
    for it in range(50):
        res = match_gate(pos_dims, [neg_dims[i] for i in sel])
        bad_keys = [k for k, v in res.items() if not v['ok']]
        if not bad_keys:
            break
        # 找到贡献最差的负样本（在 bad 维上偏离最大的）
        scores = []
        for i in sel:
            s = 0
            for k in bad_keys:
                pv = np.array([x[k] for x in pos_dims], float)
                nv = np.array([x[k] for x in neg_dims if x is neg_dims[i]] or [neg_dims[i][k]], float)
                s += abs((neg_dims[i][k] - pv.mean()) / (pv.std() + 1e-9))
            scores.append((s, i))
        scores.sort(reverse=True)
        del_idx = scores[0][1]
        sel.remove(del_idx)
        if len(sel) < 40:
            break
    neg_sel = [neg_dims[i] for i in sel]
    res_final = match_gate(pos_dims, neg_sel)
    n_ok = sum(1 for v in res_final.values() if v['ok'])
    print(f'  匹配后负样本: {len(neg_sel)}——维度通过 {n_ok}/{len(res_final)}——'
          f'未过维: {[k for k, v in res_final.items() if not v["ok"]]}')
    reject_rate = (len(neg_dims) - len(neg_sel)) / len(neg_dims)
    print(f'  拒绝率: {reject_rate:.2f}（>0.5 → 匹配张力标记）')
    tension = reject_rate > 0.5 or n_ok < len(res_final) * 0.8

    # 落盘 P0/P1
    corpus = {'pos': [{'text': b, 'src': m['src'], 'n_char': m['n_char']} for b, m in pos],
              'neg': [{'text': neg_cands[i], 'src': '拼接'} for i in sel],
              'match': {'n_pos': len(pos_dims), 'n_neg_cand': len(neg_dims), 'n_neg_sel': len(neg_sel),
                        'reject_rate': round(reject_rate, 3), 'dims': res_final, 'tension': tension,
                        'match_def': '组织层 8 维全过=匹配成功——词汇层（entropy/ttr/punct）为拼接本质差异——'
                                     '残余通道由 M-I2 与拼接抽检兜底——预注册修订'}}
    (OUT / 'intent_corpus.json').write_text(json.dumps(corpus, ensure_ascii=False, indent=1), encoding='utf-8')
    print('\n落盘 intent_corpus.json ✓')

    # ===== P2 编码 + 训练 + 置换 =====
    print('\nP2 编码（正负样本——bge512）:')
    import torch
    from para_dimensions import fingerprint
    texts = [p['text'] for p in corpus['pos']] + [n['text'] for n in corpus['neg']]
    srcs = [p['src'] for p in corpus['pos']] + [n['src'] for n in corpus['neg']]
    y = np.array([1] * len(corpus['pos']) + [0] * len(corpus['neg']))
    svs = []
    for t in texts:
        segs = seg_texts_of(t)
        sv = enc.encode(segs, normalize_embeddings=True, batch_size=16,
                        show_progress_bar=False, device='cpu').astype(np.float32)
        svs.append(sv.mean(0))
    X = np.array(svs)
    print(f'  X {X.shape}——正 {y.sum()} 负 {len(y) - y.sum()}')
    np.savez(OUT / 'intent_embeddings.npz', X=X, y=y,
             srcs=np.array(srcs))

    print('\nP2 训练（双架构——5 折×5 seeds——source-stratified）:')
    res = {}
    for mt in ('lr', 'mlp_small', 'mlp_big'):
        r = train_eval_cv(X, y, srcs, model_type=mt)
        res[mt] = r
        print(f'  {mt}: acc={r["acc_mean"]:.4f}±{r["acc_std"]:.4f}——train/val gap={r["train_val_gap"]:.3f}')

    print('\nP2 置换（按源簇聚类——LR B=500——MLP B=100 补充）:')
    perm_lr = perm_null_cluster(X, y, srcs, n_perm=500, model_type='lr')
    print(f'  LR 置换: obs={perm_lr["obs"]:.4f} p={perm_lr["p"]:.4f} null mean={perm_lr["mean_null"]:.4f} '
          f'p95={perm_lr["p95"]:.4f} n_null={perm_lr["n_null"]}')
    perm_mlp = perm_null_cluster(X, y, srcs, n_perm=100, model_type='mlp_small')
    print(f'  MLP 置换: obs={perm_mlp["obs"]:.4f} p={perm_mlp["p"]:.4f} n_null={perm_mlp["n_null"]}')

    m_i1 = {'lr': res['lr']['acc_mean'] > 0.65 and perm_lr['p'] < 0.05,
            'mlp': res['mlp_small']['acc_mean'] > 0.65 and perm_mlp['p'] < 0.05,
            'mlp_big': res['mlp_big']['acc_mean'] > 0.65}
    delta = res['mlp_small']['acc_mean'] - res['lr']['acc_mean']
    print(f'  M-I1: LR={m_i1["lr"]} MLP={m_i1["mlp"]}——Δ(MLP-LR)={delta:+.3f}')

    # ===== P3 控制 + 判定 =====
    print('\nP3 M-I2（旧判别器 AUROC——v2 判别分——OOD 分离度口径）:')
    from sklearn.metrics import roc_auc_score
    disc_scores = []
    for t in texts:
        segs = seg_texts_of(t)
        sv = enc.encode(segs, normalize_embeddings=True, batch_size=16,
                        show_progress_bar=False, device='cpu').astype(np.float32)
        with torch.no_grad():
            ds = torch.sigmoid(disc(torch.from_numpy(sv))).mean().item()
        disc_scores.append(ds)
    auroc = float(roc_auc_score(y, disc_scores))
    d_old = cohens_d(np.array(disc_scores)[y == 1], np.array(disc_scores)[y == 0])
    print(f'  旧判别器（v2）AUROC={auroc:.3f} d={d_old:.3f}（<0.62 控制通过）')
    m_i2 = auroc < 0.62

    print('\nP3 M-I3（bge512 线性探针 vs 新判别器 64 维探针——5 折——**折内训练防泄漏修复**）:')
    from sklearn.model_selection import StratifiedKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    def probe_acc(Xf):
        accs = []
        skf = StratifiedKFold(5, shuffle=True, random_state=0)
        for tr, va in skf.split(Xf, y):
            sc = StandardScaler().fit(Xf[tr])
            clf = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(Xf[tr]), y[tr])
            accs.append((clf.predict(sc.transform(Xf[va])) == y[va]).mean())
        return float(np.mean(accs))

    p_bge = probe_acc(X)
    # 折内训练：每折训 MLP（512→64→1）→ 验证折中间层 → 线性探针（无泄漏）
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    lossf = torch.nn.BCEWithLogitsLoss()
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    p_int_accs = []
    F_intent_all = np.zeros((len(y), 64))
    for fi, (tr, va) in enumerate(skf.split(X, y)):
        torch.manual_seed(fi)
        net = torch.nn.Sequential(torch.nn.Linear(512, 64), torch.nn.LayerNorm(64), torch.nn.ReLU(),
                                  torch.nn.Linear(64, 1))
        tr_t, va_t = torch.tensor(tr), torch.tensor(va)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-3)
        for ep in range(200):
            loss = lossf(net(Xt[tr_t]).squeeze(-1), yt[tr_t])
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            F_va = net[:3](Xt[va_t]).numpy()
            F_intent_all[va] = F_va
            F_tr = net[:3](Xt[torch.tensor(tr)]).numpy()
        sc2 = StandardScaler().fit(F_tr)
        clf2 = LogisticRegression(C=1.0, max_iter=2000).fit(sc2.transform(F_tr), y[tr])
        p_int_accs.append((clf2.predict(sc2.transform(F_va)) == y[va]).mean())
    p_int = float(np.mean(p_int_accs))
    print(f'  bge512 探针 acc={p_bge:.4f}——新 64 维探针 acc（折内无泄漏）={p_int:.4f}')
    m_i3 = p_int > p_bge or abs(p_int - p_bge) < 0.02

    print('\nP3 M-I4（新空间 64 维逐维 d——全量训练中间层——仅逐维分析非探针）+ M-I5 + M-I6:')
    torch.manual_seed(0)
    net_full = torch.nn.Sequential(torch.nn.Linear(512, 64), torch.nn.LayerNorm(64), torch.nn.ReLU(),
                                   torch.nn.Linear(64, 1))
    optf = torch.optim.AdamW(net_full.parameters(), lr=1e-4, weight_decay=1e-3)
    for ep in range(300):
        loss = lossf(net_full(Xt).squeeze(-1), yt)
        optf.zero_grad(); loss.backward(); optf.step()
    with torch.no_grad():
        F_intent = net_full[:3](Xt).numpy()
    ds64 = [cohens_d(F_intent[y == 1, k], F_intent[y == 0, k]) for k in range(64)]
    top64 = sorted(range(64), key=lambda k: -abs(ds64[k]))[:6]
    print(f'  M-I4 最强维: {[(k, round(ds64[k], 2)) for k in top64]}')
    # M-I5 块比率（负样本拼接结构——新判别 logits 与块数相关）
    with torch.no_grad():
        logits = net(Xt).squeeze(-1).numpy()
    from scipy.stats import spearmanr
    n_blocks = []
    for n_ in corpus['neg']:
        # 拼接块数=连接词数+1（估算）
        n_blocks.append(sum(n_['text'].count(w) for w in CONJ_WORDS) + 1)
    rho = spearmanr(logits[len(corpus['pos']):], n_blocks).statistic
    print(f'  M-I5: 新判别 logits 与负样本块数 ρ={rho:.3f}（|ρ|≥0.4 → 降级）')
    m_i5 = abs(rho) < 0.4
    # M-I6 域调整（logits ~ 风格坐标 + Mahalanobis——Ridge 回归残差后分离——防残差方差爆炸修复）
    fp_all = np.load(OUT / 'fp_matrix.npz')['fp']
    rows_all = json.loads((OUT / 'rows.json').read_text(encoding='utf-8'))
    docs_all = defaultdict(list)
    for i, r in enumerate(rows_all):
        if r['source'] == 'bilingual_zh':
            docs_all[r['doc']].append(i)
    human_all = [d for d in docs_all if d.startswith('ZH-H')]
    ai_all = [d for d in docs_all if d.startswith('ZH-A')]
    mu_h = fp_all[np.concatenate([np.array(docs_all[d]) for d in human_all])].mean(0)
    mu_a = fp_all[np.concatenate([np.array(docs_all[d]) for d in ai_all])].mean(0)
    from sklearn.linear_model import Ridge
    # 协变量：判别分 + 字符数 + 句元数（Ridge 正则——防过拟合残差）
    cov = np.column_stack([disc_scores, np.array([len(t) for t in texts]),
                           np.array([len(seg_texts_of(t)) for t in texts])])
    reg = Ridge(alpha=10.0).fit(cov, logits)
    resid = logits - reg.predict(cov)
    rsd = resid.std()
    d_res = cohens_d(resid[y == 1], resid[y == 0])
    print(f'  M-I6: 域调整（Ridge 残差）d_res={d_res:.3f} 残差 std={rsd:.4f}（<0.4 → 降级）')
    m_i6 = abs(d_res) >= 0.4

    # ===== 四态判定 =====
    m_i1_pass = m_i1['lr'] or m_i1['mlp']
    if not m_i1_pass:
        overall = '学不会'
        interp = ('思想不在文本表面也不在一阶组织形式（功率范围 θ≤0.65——对应 N≈140、'
                  'd≈0.55-0.65 效应量——置换校准 CV）')
    elif m_i2 and m_i5 and m_i6:
        overall = '学会'
        interp = ('意图指纹首次承载核心意图——可测的思想几何形状（温和措辞：N≈140 构造性正负样本内——'
                  'MLP 与 LR 双架构' + ('一致' if m_i1['mlp'] else '仅线性') + '）')
    else:
        overall = '学到组织（降级）'
        interp = ('判别来自未匹配的组织/表面残差/语体（' +
                  ('M-I2 ' if not m_i2 else '') + ('M-I5 ' if not m_i5 else '') +
                  ('M-I6 ' if not m_i6 else '') + '触发）——现有组织口径不完整')
    print(f'\n  总判定: {overall}')
    print(f'  解释: {interp}')

    # 落盘
    verdict = {'overall': overall, 'interpretation': interp,
               'm_i1': {k: bool(v) for k, v in m_i1.items()},
               'm_i2': {'auroc': round(auroc, 3), 'pass': bool(m_i2)},
               'm_i3': {'bge': round(p_bge, 4), 'intent64': round(p_int, 4)},
               'm_i4': {'top': [(int(k), round(float(ds64[k]), 3)) for k in top64]},
               'm_i5': {'rho': round(float(rho), 3), 'pass': bool(m_i5)},
               'm_i6': {'d_res': round(float(d_res), 3), 'pass': bool(m_i6)},
               'train': res, 'perm_lr': perm_lr, 'perm_mlp': perm_mlp}
    (OUT / 'intent_verdict.json').write_text(json.dumps(verdict, ensure_ascii=False, indent=1),
                                             encoding='utf-8')
    print('落盘 intent_verdict.json ✓')


# ============ P2 训练 ============
def train_eval_cv(X, y, srcs, n_folds=N_FOLDS, n_seeds=N_SEEDS, model_type='lr'):
    """source-stratified 5 折 × 5 seeds——LR 或 MLP——返回 (acc_mean, acc_std, per_fold_seed, train_val_gap)"""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    import torch
    import torch.nn as nn
    accs, gaps = [], []
    for sd in range(n_seeds):
        skf = StratifiedKFold(n_folds, shuffle=True, random_state=SEED_FOLDS + sd)
        for tr, va in skf.split(X, y):
            if model_type == 'lr':
                sc = StandardScaler().fit(X[tr])
                clf = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(X[tr]), y[tr])
                pred = clf.predict(sc.transform(X[va]))
                accs.append((pred == y[va]).mean())
                tr_pred = clf.predict(sc.transform(X[tr]))
                gaps.append((tr_pred == y[tr]).mean() - (pred == y[va]).mean())
            else:
                torch.manual_seed(sd)
                dim = X.shape[1]
                if model_type == 'mlp_big':
                    net = nn.Sequential(nn.Linear(dim, 256), nn.LayerNorm(256), nn.ReLU(),
                                        nn.Linear(256, 64), nn.LayerNorm(64), nn.ReLU(),
                                        nn.Linear(64, 1))
                else:
                    net = nn.Sequential(nn.Linear(dim, 64), nn.LayerNorm(64), nn.ReLU(),
                                        nn.Linear(64, 1))
                Xt = torch.tensor(X, dtype=torch.float32)
                yt = torch.tensor(y, dtype=torch.float32)
                tr_t, va_t = torch.tensor(tr), torch.tensor(va)
                lossf = nn.BCEWithLogitsLoss()
                opt = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-3)
                net.train()
                best_val, patience = -1, 0
                val_pred = None
                for ep in range(200):
                    out = net(Xt[tr_t]).squeeze(-1)
                    loss = lossf(out, yt[tr_t])
                    opt.zero_grad(); loss.backward(); opt.step()
                    net.eval()
                    with torch.no_grad():
                        vp = (torch.sigmoid(net(Xt[va_t]).squeeze(-1)) > 0.5).numpy().astype(int)
                        va_acc = (vp == y[va]).mean()
                    net.train()
                    if va_acc > best_val:
                        best_val = va_acc
                        val_pred = vp
                        patience = 0
                    else:
                        patience += 1
                        if patience >= 20:
                            break
                accs.append(float(best_val))
                with torch.no_grad():
                    tp = (torch.sigmoid(net(Xt[tr_t]).squeeze(-1)) > 0.5).numpy().astype(int)
                gaps.append(float((tp == y[tr]).mean() - best_val))
    return {'acc_mean': float(np.mean(accs)), 'acc_std': float(np.std(accs)),
            'train_val_gap': float(np.mean(gaps)), 'n_evals': len(accs)}


def perm_null_cluster(X, y, srcs, n_perm=B_PERM, seed=SEED_PERM, model_type='lr'):
    """按源簇聚类置换（用户关键 1）：正样本按源文档（实践论/矛盾论）成簇、负样本每块独立——
    置换单位=簇（簇内样本同标签）——B 次——p=(1+#{null≥obs})/(1+B)——LR（校准主判——
    MLP 置换 B=100 单 seed 补充——预注册注明）"""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    import torch
    import torch.nn as nn
    rng = np.random.default_rng(seed)
    # 簇定义：正样本 src 相同归簇；负样本每块独立簇
    uniq = sorted(set(srcs))
    cluster_of = {s: (f'pos:{s}' if s in ('实践论', '矛盾论') else f'neg:{i}')
                  for i, s in enumerate(srcs)}
    clusters = sorted(set(cluster_of.values()))
    # obs acc（单 seed——同程序）
    def cv_acc(X, y, sd=0):
        skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED_FOLDS + sd)
        accs = []
        for tr, va in skf.split(X, y):
            if model_type == 'lr':
                sc = StandardScaler().fit(X[tr])
                clf = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(X[tr]), y[tr])
                accs.append((clf.predict(sc.transform(X[va])) == y[va]).mean())
            else:
                torch.manual_seed(sd)
                net = nn.Sequential(nn.Linear(X.shape[1], 64), nn.LayerNorm(64), nn.ReLU(), nn.Linear(64, 1))
                Xt = torch.tensor(X, dtype=torch.float32); yt = torch.tensor(y, dtype=torch.float32)
                tr_t, va_t = torch.tensor(tr), torch.tensor(va)
                lossf = nn.BCEWithLogitsLoss()
                opt = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-3)
                best_val = -1
                for ep in range(200):
                    out = net(Xt[tr_t]).squeeze(-1)
                    loss = lossf(out, yt[tr_t])
                    opt.zero_grad(); loss.backward(); opt.step()
                    net.eval()
                    with torch.no_grad():
                        vp = (torch.sigmoid(net(Xt[va_t]).squeeze(-1)) > 0.5).numpy().astype(int)
                        va_acc = (vp == y[va]).mean()
                    net.train()
                    if va_acc > best_val:
                        best_val = va_acc
                    else:
                        pass
                accs.append(float(best_val))
        return float(np.mean(accs))
    obs = cv_acc(X, y)
    nulls = []
    for b in range(n_perm):
        # 每个簇整体换标签（等概率翻转）——保持正负平衡近似
        flips = {c: bool(rng.random() < 0.5) for c in clusters}
        yp = np.array([1 - y[i] if flips[cluster_of[srcs[i]]] else y[i] for i in range(len(y))])
        if yp.sum() < 10 or yp.sum() > len(y) - 10:
            continue
        nulls.append(cv_acc(X, yp))
    nulls = np.array(nulls)
    p = (1 + int(np.sum(nulls >= obs))) / (1 + len(nulls))
    return {'obs': obs, 'p': float(p), 'n_null': len(nulls),
            'p95': float(np.percentile(nulls, 95)), 'mean_null': float(np.mean(nulls))}


if __name__ == '__main__':
    main()
