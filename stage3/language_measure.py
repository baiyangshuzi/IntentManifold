# -*- coding: utf-8 -*-
"""v0.67 共享模块：中文侧段级测量（照搬 build_language_mechanism.main() 内逻辑——提为可 import 函数）

measure(txt, enc, disc, pseg, device) -> dict | None
  一段：指纹量（sent_proj + fp_mean 64 维）+ 6 个可解释语言特征
  （指代密度/连接词密度/句首已知信息率/主题连续性/主题词密度/句长变化——CONJ 23 词表）
cohens_d(h, a) -> float   # pooled sd 双侧效应量
"""
import math
import numpy as np

CONJ = ['虽然', '但是', '但', '因为', '所以', '然而', '于是', '因此', '却', '便', '就', '不过',
        '而且', '并且', '如果', '那么', '只要', '只有', '无论', '即使', '尽管', '可是', '于是乎']


def measure(txt, enc, disc, pseg, device='cpu'):
    """一段：指纹量 + 6 语言特征（句元数 <2 返回 None）"""
    import torch
    from para_dimensions import fingerprint, norm_rows
    from subclause_structure import split_subclauses

    ss = [s for s in split_subclauses(txt) if len(s) >= 3]
    if len(ss) < 2:
        return None
    with torch.no_grad():
        sv = enc.encode(ss, normalize_embeddings=True, batch_size=64,
                        show_progress_bar=False, device=device)
        SV = torch.from_numpy(sv.astype(np.float32)).to(device)
        F = norm_rows(fingerprint(SV, disc)).cpu().numpy()
        core = F.mean(0)
        core = core / (np.linalg.norm(core) + 1e-9)
        proj = (F @ core)  # 每句元投影
        sent_proj = float(proj.mean())
    segs = [list(pseg.cut(s)) for s in ss]
    n = len(segs)
    pron_density = sum(1 for seg in segs for w, f in seg if f == 'r') / n          # 指代密度
    conj_density = sum(1 for seg in segs for w, f in seg if w in CONJ) / n          # 连接词密度
    known_first = sum(1 for seg in segs if seg and seg[0].flag in ('r', 'c')) / n    # 句首已知信息率

    def content_words(seg):
        return {w for w, f in seg if len(w) >= 2 and f[0] in ('n', 'v', 'a')}
    cws = [content_words(s) for s in segs]
    shared = []
    for i in range(len(cws) - 1):
        inter = len(cws[i] & cws[i + 1])
        union = len(cws[i] | cws[i + 1])
        shared.append(inter / union if union else 0)
    topic_cont = float(np.mean(shared)) if shared else 0.0
    from collections import Counter
    all_cw = [w for cw in cws for w in cw]
    wc = Counter(all_cw)
    high = sum(1 for w in all_cw if wc[w] >= 2)
    topic_density = high / len(all_cw) if all_cw else 0.0
    len_std = float(np.std([len(s) for s in ss]))
    return {'sent_proj': sent_proj, 'fp_mean': F.mean(0).tolist(),
            'pron_density': pron_density, 'conj_density': conj_density,
            'known_first': known_first, 'topic_cont': topic_cont,
            'topic_density': topic_density, 'len_std': len_std}


def cohens_d(h, a):
    h = np.asarray(h, float); a = np.asarray(a, float)
    sp = math.sqrt(((len(h) - 1) * h.var(ddof=1) + (len(a) - 1) * a.var(ddof=1)) / (len(h) + len(a) - 2))
    return float((h.mean() - a.mean()) / sp) if sp else 0.0
