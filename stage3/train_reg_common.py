# -*- coding: utf-8 -*-
"""v0.93 训练期正则共享模块——G0/G1/配对/代理/损失/训练循环共用

锚点口径（与 v0.79-0.92 完全一致——fingerprint 原始空间、句元级）：
- fingerprint(x, disc)：ParaDiscNN net[0..3] + 手动 ReLU——(n,512)→(n,64) 原始指纹
- D_shared：axis_analysis.json axis.D_shared（64 维 unit-norm）
- ratio_of：per-doc ratio_unit（engine_ratio_validate.py:73-79 精确复刻）
- p90 = 3.2286（全局固定跳跃阈值——v0.90 口径）
"""
import sys, json
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path
import re
import numpy as np
import torch
import torch.nn as nn

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / 'data' / 'dim_analysis'
DISC_PATH = BASE / 'data' / 'para_discriminator_v2.pt'
P90 = 3.2286
TARGET = {'alpha_mean_abs': 1.4746, 'alpha_ai': 1.2989, 'alpha_mid': 1.3,
          'ratio_human': 0.0863, 'ratio_ai': 0.1048, 'ratio_mid': 0.095,
          'density_human': 12.08, 'density_ai': 7.74, 'density_mid': 9.0}


class ParaDiscNN(nn.Module):
    """精确复刻 train_para_discriminator_nn.py:37-47"""

    def __init__(self, dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 64), nn.LayerNorm(64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_causal_lm(model_id, dtype, device):
    """版本自适应加载（transformers 5.x: dtype=；4.46.x: torch_dtype=）"""
    import transformers
    try:
        m = transformers.AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    except TypeError:
        m = transformers.AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    return m.to(device)


def load_disc(device='cpu'):
    m = ParaDiscNN(512)
    m.load_state_dict(torch.load(DISC_PATH, map_location=device))
    m.to(device).eval()
    return m


def get_D_shared():
    d = json.loads((OUT / 'axis_analysis.json').read_text(encoding='utf-8'))
    D = np.asarray(d['axis']['D_shared'], float)
    D = D / (np.linalg.norm(D) + 1e-12)
    assert D.shape == (64,) and abs(np.linalg.norm(D) - 1.0) < 1e-3
    return D


def fingerprint(x, disc):
    """x: (n,512) tensor——返回 (n,64) 原始指纹"""
    with torch.no_grad():
        h = disc.net[0](x)
        h = disc.net[1](h)
        h = torch.relu(h)
        h = disc.net[2](h)
        h = disc.net[3](h)
        h = torch.relu(h)
    return h.cpu().numpy()


def fp_from_bge(seq, disc):
    """seq: (n,512) numpy——bge 归一化嵌入——指纹"""
    if len(seq) == 0:
        return np.zeros((0, 64))
    return fingerprint(torch.as_tensor(np.asarray(seq, dtype=np.float32)), disc)


def ratio_of(D_fp, D_axis):
    """per-doc ratio_unit——engine_ratio_validate.py:73-79 精确复刻——D_fp:(n,64) 差分"""
    if len(D_fp) == 0:
        return 0.0, 0.0, 0.0
    n = np.linalg.norm(D_fp, axis=1)
    u = D_fp / (n[:, None] + 1e-9)
    jppu = float(np.mean(np.abs(u - (u @ D_axis)[:, None] * D_axis)))
    jpu = float(np.mean(np.abs(u @ D_axis)))
    return jppu / (jpu + 1e-9), jpu, jppu


def alpha_metrics(fp, D):
    """fp:(n,64)——α = Δ·D——返回 mean_abs, mean_signed, n"""
    if len(fp) < 2:
        return 0.0, 0.0, 0
    Df = fp[1:] - fp[:-1]
    a = Df @ D
    return float(np.mean(np.abs(a))), float(np.mean(a)), len(a)


def density_hard(fp, p90=P90):
    """跳跃密度 100·|{‖Δ‖>p90}|/(n−1)"""
    if len(fp) < 2:
        return 0.0
    Df = fp[1:] - fp[:-1]
    return 100.0 * np.mean(np.linalg.norm(Df, axis=1) > p90)


def density_soft(fp, k=5.0, p90=P90):
    """软计数密度（可微——sigmoid 软阈值 k=5 预注册）"""
    if len(fp) < 2:
        return 0.0
    Df = fp[1:] - fp[:-1]
    return 100.0 * float(torch.sigmoid(k * (torch.norm(Df, dim=1) - p90)).mean())


def seg_texts_of(text):
    """长文本 → 句元（同 engine_trajectory_expand.py:55-67 口径）"""
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


def chunk_paragraphs(text, lo=400, hi=800):
    """行式段落 → 400-800 字符连续块（段落边界对齐——LM 流语料单元）"""
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    chunks, cur, cur_src = [], '', None
    for l in lines:
        if not re.search(r'[一-龥]', l) or re.match(r'^第[一二三四五六七八九十百千0-9]+[章节卷回部]', l):
            continue
        if len(l) < 10:  # 短行（编号/落款）并入前块
            continue
        if len(cur) + len(l) > hi and len(cur) >= lo:
            chunks.append(cur)
            cur = ''
        cur += l
    if len(cur) >= lo:
        chunks.append(cur)
    return chunks
