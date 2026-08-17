# -*- coding: utf-8 -*-
"""意图损失模块数值验证（torch vs numpy 参考——多工况）"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
import torch
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
from intent_loss import intent_losses
from train_reg_common import get_D_shared, ratio_of, density_hard, alpha_metrics

D = get_D_shared()
Dt = torch.as_tensor(D, dtype=torch.float32)


def check(fp, tag):
    fp = np.asarray(fp, np.float32)
    Df = fp[1:] - fp[:-1]
    a_abs, _, _ = alpha_metrics(fp, D)
    r, _, _ = ratio_of(Df, D)
    n_ = np.linalg.norm(Df, axis=1)
    den_soft = 100.0 * np.mean(1 / (1 + np.exp(-5.0 * (n_ - 3.2286))))
    la_ref = max(0, 1.3 - a_abs) ** 2
    lr_ref = max(0, r - 0.095) ** 2
    lj_ref = max(0, 9.0 - den_soft) ** 2
    pf = torch.tensor(fp, dtype=torch.float32, requires_grad=True)
    la, lr, lj = intent_losses(pf, Dt)
    (la + lr + lj).backward()
    g = pf.grad.norm().item()
    ok = (abs(la.item() - la_ref) < 1e-4 and abs(lr.item() - lr_ref) < 1e-4
          and abs(lj.item() - lj_ref) < 1e-4 and np.isfinite(g))
    print(f'{tag}: La={la.item():.4f}(ref {la_ref:.4f}) Lr={lr.item():.4f}({lr_ref:.4f}) '
          f'Lj={lj.item():.4f}({lj_ref:.4f}) grad={g:.3f} {"OK" if ok else "FAIL"}')
    return ok


rng = np.random.default_rng(7)
ok1 = check(rng.normal(0, 1, (30, 64)).astype(np.float32) * 2.0, '随机(alpha达标)')
ok2 = check(rng.normal(0, 0.3, (30, 64)).astype(np.float32), '小尺度(损失全激活)')
base = rng.normal(0, 2, (64,))
t = np.linspace(0, 2, 30)
ok3 = check((np.sin(t)[:, None] * base[None, :]).astype(np.float32), '平滑轨迹(密度低)')
ok4 = check((base[None, :] + rng.normal(0, 1, (30, 64)) * 3).astype(np.float32), '高弥散(ratio高)')
print('全部通过 ✓' if all([ok1, ok2, ok3, ok4]) else '存在失败!')
