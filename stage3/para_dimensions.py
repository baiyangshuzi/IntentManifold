# -*- coding: utf-8 -*-
"""v0.65-5 维度计算公共模块（para_dimensions——风格判别全维度实现）

七维度计算（判别分=模型计算；六维=规则统计计算——可解释非黑盒）：
  disc       判别分    = MLP 输出（模型计算）——sigmoid(disc(bge(段落句元)))
  sent_proj  句元投影  = 句元指纹 · 段核心（mean 核心）余弦——贴塑造核心程度（规则）
  traj       轨迹粘合  = 段内相邻句元投影差绝对值均值——核心漂移（规则）
  l7_adj     相邻句元  = 段内相邻句元指纹余弦——句元联系一致性（规则）
  word_proj  词投影    = 词指纹 · 句元核心余弦均值——用词贴核心程度（规则）
  word_adj   词相邻    = 句元内相邻词指纹余弦均值——词与词连接一致（规则）
  entropy    词熵      = 词频分布熵公式——词汇丰富度（纯规则——不依赖判别器）

指纹（64 维）= 判别器中间层（ParaDiscNN net[0:3]+relu+net[3]+relu——冻结特征提取）
段核心 = 段内句元指纹均值（mean 核心——不是整段 bge 编码）
"""
import torch
import numpy as np


def fingerprint(x, disc):
    """风格指纹（判别器中间层 64 维——net[:4] 不含最后 ReLU 需手动补）"""
    h = disc.net[0](x); h = disc.net[1](h); h = torch.relu(h)
    h = disc.net[2](h); h = disc.net[3](h); h = torch.relu(h)
    return h


def norm_rows(m):
    return m / (m.norm(2, 1, keepdim=True) + 1e-9)


def entropy_of_words(words):
    """词熵（词频分布熵——纯规则——不依赖判别器）
    H = -Σ (c_i/N)·ln(c_i/N)——c_i=词频 N=总词数"""
    wc = {}
    for w in words:
        wc[w] = wc.get(w, 0) + 1
    ws = list(wc.values())
    N = sum(ws)
    if N == 0:
        return 0.0
    return -sum((c / N) * np.log(c / N) for c in ws)


def para_dimensions(p, enc, disc, split_subclauses, pseg, device='cpu'):
    """段落全维度计算（一个入口——七维）

    参数：
      p    - 自然段文本
      enc  - bge 编码器（sentence_transformers——BAAI/bge-small-zh-v1.5）
      disc - ParaDiscNN 判别器（已加载——fingerprint 来源）
      split_subclauses - 句元切分函数（subclause_structure）
      pseg - jieba.posseg（词性标注）
    返回：
      dict：disc/sent_proj/traj/l7_adj/word_proj/word_adj/entropy
    """
    ss = [s for s in split_subclauses(p) if len(s) >= 3]
    if len(ss) < 1:
        return None
    with torch.no_grad():
        sv = enc.encode(ss, normalize_embeddings=True, batch_size=64,
                        show_progress_bar=False, device=device)
        SV = torch.from_numpy(sv.astype(np.float32)).to(device)
        F = fingerprint(SV, disc)                    # 句元指纹 (n,64)——模型特征（冻结）
        Fn = norm_rows(F)
        core = Fn.mean(0)                            # 段核心（mean——句元指纹均值）
        core = core / (core.norm() + 1e-9)
        proj = (Fn * core).sum(1).cpu().numpy()      # 句元投影（余弦——规则）
        disc_s = torch.sigmoid(disc(SV)).cpu().numpy()  # 判别分（MLP——模型）
    # 词级（词 bge → 指纹 → 句元核心投影——规则）
    w_projs, w_adj = [], []
    all_words = []
    for si, s in enumerate(ss):
        ws = [w for w, _ in pseg.cut(s) if len(w) >= 2]
        all_words += ws
        if len(ws) < 2:
            continue
        with torch.no_grad():
            wv = enc.encode(ws, normalize_embeddings=True, batch_size=64,
                            show_progress_bar=False, device=device)
            WV = torch.from_numpy(wv.astype(np.float32)).to(device)
            WF = norm_rows(fingerprint(WV, disc))    # 词指纹（模型特征——冻结）
            s_core = Fn[si]                          # 句元核心（单句元指纹归一化）
            s_core = s_core / (s_core.norm() + 1e-9)
            w_projs.append(float((WF * s_core).sum(1).mean().cpu().item()))   # 词投影
            w_adj.append(float((WF[:-1] * WF[1:]).sum(1).mean().cpu().item()))  # 词相邻
    return {
        'disc': float(np.mean(disc_s)),                       # 模型计算
        'sent_proj': float(np.mean(proj)),                    # 规则：指纹·段核心
        'traj': float(np.mean(np.abs(np.diff(proj)))) if len(proj) > 1 else 0.0,  # 规则：投影差分
        'l7_adj': float(np.mean((Fn[:-1] * Fn[1:]).sum(1).cpu().numpy())) if len(Fn) > 1
                  else float((Fn[0] * core).sum().item()),    # 规则：相邻指纹余弦
        'word_proj': float(np.mean(w_projs)) if w_projs else 0,   # 规则：词贴核心
        'word_adj': float(np.mean(w_adj)) if w_adj else 0,        # 规则：词相邻一致
        'entropy': entropy_of_words(all_words),                   # 纯规则：词频熵
    }


def load_models(device='cuda'):
    """加载 bge 编码器 + 判别器（ParaDiscNN）——工程资产"""
    from sentence_transformers import SentenceTransformer
    from train_para_discriminator_nn import ParaDiscNN
    from pathlib import Path
    import os
    BASE = Path(__file__).resolve().parent.parent
    enc = SentenceTransformer('BAAI/bge-small-zh-v1.5')
    enc.to(device); enc.eval()
    disc = ParaDiscNN(512)
    fname = os.environ.get('PARA_DISC', 'para_discriminator_v2.pt')  # 默认 v2 泛用性
    disc.load_state_dict(torch.load(BASE / "data" / fname, map_location=device))
    disc.to(device); disc.eval()
    return enc, disc


if __name__ == '__main__':
    """示例：人类段 vs AI 段七维计算"""
    import sys, os
    sys.path.insert(0, 'stage3')
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    from subclause_structure import split_subclauses
    import jieba
    import jieba.posseg as pseg
    jieba.setLogLevel(60)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    enc, disc = load_models(DEVICE)
    human_p = '东门城头，风从江面灌过来，水腥气扑面而来。楚休红踏上最后一级石阶时，看见陆经渔背对着他，正望着城下黑沉沉的江面。甲胄在月光下泛着青白的光，薄霜似的覆着。'
    ai_p = '沈彻站在牢房中央，目光扫过四周的石壁。他深吸一口气，感受到空气中弥漫的潮湿气息。第七十三滴水珠从岩缝渗出，缓缓滑落。他伸出手，接住了那滴水，水珠在掌心泛着微光，像是一颗凝结的琥珀。'
    for tag, p in [('人类', human_p), ('AI', ai_p)]:
        r = para_dimensions(p, enc, disc, split_subclauses, pseg, DEVICE)
        print(f'{tag}: ' + ' '.join(f'{k}={v:.3f}' for k, v in r.items()))
