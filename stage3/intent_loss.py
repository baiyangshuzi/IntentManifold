# -*- coding: utf-8 -*-
"""v0.93 意图损失模块（评审 4 吸收——平方 hinge + 软计数 k=5——全部 torch 可微）

pred_fp: (n,64) torch 张量——代理预测的原始指纹轨迹（一个文档）
D: (64,) torch——D_shared（unit-norm 固定）
p90 = 3.2286 固定——k=5 软计数（评审 4：k=10 过陡梯度饱和）

损失（预注册——平方 hinge——kink 处梯度连续）：
- L_α   = (max(0, 1.3 − mean|α̂|))²        α̂ = Δ̂·D——mean_abs 向 ≥1.3 推（中点 1.2989/1.4746）
- L_ratio = (max(0, ratiô − 0.095))²       ratiô 向 ≤0.095 压（0.0863/0.1048 中点）
- L_jump = (max(0, 9.0 − density_soft))²   density_soft = 100·mean(sigmoid(5·(‖Δ̂‖−3.2286)))——向 ≥9 推
- 返回 (l_alpha, l_ratio, l_jump)——批内 = mean(三项和)
"""
import torch

P90 = 3.2286
EPS = 1e-9


def intent_losses(pred_fp, D, k=5.0, p90=P90):
    """pred_fp:(n,64) 原始指纹轨迹——返回 (l_alpha, l_ratio, l_jump) 标量张量"""
    D = D.to(pred_fp.dtype)
    Df = pred_fp[1:] - pred_fp[:-1] if len(pred_fp) > 1 else pred_fp[:0]
    if len(Df) < 1:
        return (torch.zeros((), device=pred_fp.device),
                torch.zeros((), device=pred_fp.device),
                torch.zeros((), device=pred_fp.device))
    # α
    alpha = Df @ D
    mean_abs = alpha.abs().mean()
    l_alpha = torch.clamp(1.3 - mean_abs, min=0.0) ** 2
    # ratio（单位化垂直/平行比——engine_ratio_validate 同口径）
    n = torch.norm(Df, dim=1)
    u = Df / (n[:, None] + EPS)
    jppu = (u - (u @ D)[:, None] * D).abs().mean()
    jpu = (u @ D).abs().mean()
    ratio = jppu / (jpu + EPS)
    l_ratio = torch.clamp(ratio - 0.095, min=0.0) ** 2
    # 软计数密度
    density = 100.0 * torch.sigmoid(k * (torch.norm(Df, dim=1) - p90)).mean()
    l_jump = torch.clamp(9.0 - density, min=0.0) ** 2
    return l_alpha, l_ratio, l_jump
