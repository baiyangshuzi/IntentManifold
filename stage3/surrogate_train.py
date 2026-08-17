# -*- coding: utf-8 -*-
"""v0.93 代理网络训练 + G1 门（评审 1 分层——严格/降级/否定）

【预注册口径（含评审后修正——如实声明）】
- 输入：Qwen 末层每句元 mean-pool 隐状态（896-d）——z-score（train 拟合冻结）
- 输出：64 维原始指纹——z-score 目标训练——推理反 z-score 回原始空间
- 架构：896 → LN → 256 ReLU → 128 ReLU → 64
- 损失：MSE(z 空间) + 0.3·(1−cos)(原始空间) + **0.5·相对软密度 MSE（doc 级）**
  ——修正记录：v0.93 诊断发现 MSE 回归向均值衰减 → 密度系统性低估（预测 2-3 vs 真 8-12）——
  加 doc 级软密度辅助项（与训练循环 L_jump 同口径 sigmoid(5·(‖Δ‖−3.2286))）——预注册偏差声明
- 训练：AdamW lr 3e-4，batch=doc（单元级损失批内聚合 + doc 级密度项），max 60 轮，
  early stop patience 10——val = doc 级 3 指标平均相对误差（α mean_abs/ratio_of/density_hard）
- val 切分（doc 级——pairs.npz val 标志）：bilingual 前 3 人类+前 5 AI 篇；intent 前 8 正+前 7 负；
  selfgen 前 20 段——永久排除出代理训练

【G1 门（评审 1 分层）】
- 方向一致性（硬）：val bilzh human(3) vs ai(5) 三指标缺口方向 == 真缺口方向（全一致）
- 均值相对误差：**长文档（bilzh 缺口定义域）** 三指标（α/ratio/density_hard）
  ≤0.3 严格通过；0.3<err≤0.5 降级通过；>0.5 或方向不一致 → 否定 → RWR
  （预注册偏差声明：原"held-out 全文档"基准含 intent/selfgen 短段——密度 rel err 被
  计数噪声主导——拆分为全文档/长文档双报——G1 基准取长文档）

产出：data/dim_analysis/surrogate.pt + surrogate_scaler.json + gate_g1.json
"""
import sys, json, time, os
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / 'data' / 'dim_analysis'
LR, MAX_EPOCH, PATIENCE = 3e-4, 60, 10
LAMBDA_D = 1.0  # 密度辅助项权重（相对软密度 MSE——v0.93 诊断后上调）

from train_reg_common import get_D_shared, ratio_of, alpha_metrics, density_hard
from intent_loss import P90, intent_losses


class Surrogate(nn.Module):
    def __init__(self, d_in=896, d_out=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, d_out),
        )

    def forward(self, x):
        return self.net(x)


def soft_den_torch(fp, k=5.0):
    """软密度（torch——与 intent_loss 同口径）"""
    if len(fp) < 2:
        return torch.zeros((), device=fp.device)
    n = torch.norm(fp[1:] - fp[:-1], dim=1)
    return 100.0 * torch.sigmoid(k * (n - P90)).mean()


def doc_metrics(fp_doc, D):
    a_abs, _, _ = alpha_metrics(fp_doc, D)
    Df = fp_doc[1:] - fp_doc[:-1] if len(fp_doc) >= 2 else np.zeros((0, 64))
    r = ratio_of(Df, D)[0] if len(Df) else 0.0
    return {'alpha_abs': a_abs, 'ratio': r, 'density': density_hard(fp_doc)}


def main():
    t0 = time.time()
    d = np.load(OUT / 'pairs.npz')
    H, F = d['h'], d['fp']
    DOC, VAL = d['doc'], d['val']
    D = get_D_shared()
    print(f'配对 {len(H)} 对（val {int(VAL.sum())}——{time.time()-t0:.0f}s）')

    h_mean = H[~VAL].mean(0)
    h_std = H[~VAL].std(0) + 1e-6
    f_mean = F[~VAL].mean(0)
    f_std = F[~VAL].std(0) + 1e-6
    X = (H - h_mean) / h_std
    Z = (F - f_mean) / f_std

    # doc 分组（训练集）
    doc_of = np.array([str(x) for x in DOC])
    train_docs = sorted({x for x in doc_of[~VAL]})
    groups = defaultdict(list)
    for j, i in enumerate(np.where(~VAL)[0]):
        groups[str(DOC[i])].append(j)
    # 单文档过大则按 512 单元切块（doc 级密度项保持同 doc——切块独立）
    blocks = []
    for doc in train_docs:
        idx = groups[doc]
        for s in range(0, len(idx), 512):
            blocks.append((doc, idx[s:s + 512]))

    torch.manual_seed(42)
    net = Surrogate().train()
    opt = torch.optim.AdamW(net.parameters(), lr=LR)

    def predict(h_raw):
        with torch.no_grad():
            z = torch.as_tensor((h_raw - h_mean) / h_std, dtype=torch.float32)
            pz = net(z)
            pr = pz * torch.as_tensor(f_std) + torch.as_tensor(f_mean)
        return pr.numpy()

    def val_metric():
        rows = defaultdict(lambda: {'t': [], 'p': []})
        for i in np.where(VAL)[0]:
            rows[str(DOC[i])]['t'].append(F[i])
            rows[str(DOC[i])]['p'].append(predict(H[i:i + 1])[0])
        errs = {k: [] for k in ['alpha_abs', 'ratio', 'density']}
        for doc, v in rows.items():
            ft, fp_ = np.stack(v['t']), np.stack(v['p'])
            mt, mp = doc_metrics(ft, D), doc_metrics(fp_, D)
            for k in errs:
                if abs(mt[k]) > 1e-9:
                    errs[k].append(abs(mp[k] - mt[k]) / abs(mt[k]))
        return {k: float(np.mean(v)) if v else None for k, v in errs.items()}

    best, patience, best_state = None, 0, None
    for ep in range(MAX_EPOCH):
        net.train()
        order = np.random.default_rng(ep).permutation(len(blocks))
        tot = 0.0
        for bi in order:
            doc, idx = blocks[bi]
            x = torch.as_tensor(X[idx], dtype=torch.float32)
            zt = torch.as_tensor(Z[idx], dtype=torch.float32)
            ft = torch.as_tensor(F[idx], dtype=torch.float32)
            pz = net(x)
            pr = pz * torch.as_tensor(f_std) + torch.as_tensor(f_mean)
            unit_loss = ((pz - zt) ** 2).mean() + 0.3 * (1 - (pr * ft).sum(1) /
                         (pr.norm(2, 1) * ft.norm(2, 1) + 1e-9)).mean()
            d_true = soft_den_torch(ft).detach()
            d_pred = soft_den_torch(pr)
            den_loss = ((d_pred - d_true) / max(d_true.item(), 5.0)) ** 2
            loss = unit_loss + LAMBDA_D * den_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
        net.eval()
        vm = val_metric()
        score = float(np.mean([v for v in vm.values() if v is not None]))
        if best is None or score < best:
            best, patience, best_state = score, 0, {k: v.clone() for k, v in net.state_dict().items()}
        else:
            patience += 1
        print(f'  ep{ep}: loss {tot/len(blocks):.4f} val_err {vm} -> {score:.4f} (best {best:.4f})')
        if patience >= PATIENCE:
            print(f'  early stop @ ep{ep}')
            break

    net.load_state_dict(best_state)
    net.eval()
    torch.save(best_state, OUT / 'surrogate.pt')
    json.dump({'h_mean': h_mean.tolist(), 'h_std': h_std.tolist(),
               'f_mean': f_mean.tolist(), 'f_std': f_std.tolist()},
              open(OUT / 'surrogate_scaler.json', 'w'), ensure_ascii=False)
    print(f'代理训练完成（best {best:.4f}——{time.time()-t0:.0f}s）')

    # ===== G1 判定 =====
    vm = val_metric()
    human_docs = sorted({str(DOC[i]) for i in np.where(VAL)[0] if str(DOC[i]).startswith('ZH-H')})
    ai_docs = sorted({str(DOC[i]) for i in np.where(VAL)[0] if str(DOC[i]).startswith('ZH-A')})
    rows = defaultdict(lambda: {'t': [], 'p': []})
    for i in np.where(VAL)[0]:
        rows[str(DOC[i])]['t'].append(F[i])
        rows[str(DOC[i])]['p'].append(predict(H[i:i + 1])[0])

    def gap(docs_list):
        mt = {k: [] for k in ['alpha_abs', 'ratio', 'density']}
        mp = {k: [] for k in ['alpha_abs', 'ratio', 'density']}
        for doc in docs_list:
            ft, fp_ = np.stack(rows[doc]['t']), np.stack(rows[doc]['p'])
            mt2, mp2 = doc_metrics(ft, D), doc_metrics(fp_, D)
            for k in mt:
                mt[k].append(mt2[k])
                mp[k].append(mp2[k])
        return mt, mp

    expect = {'alpha_abs': 'H>A', 'ratio': 'H<A', 'density': 'H>A'}
    mt_h, mp_h = gap(human_docs)
    mt_a, mp_a = gap(ai_docs)
    true_dir = {k: ('H>A' if np.mean(mt_h[k]) > np.mean(mt_a[k]) else 'H<A') for k in expect}
    pred_dir = {k: ('H>A' if np.mean(mp_h[k]) > np.mean(mp_a[k]) else 'H<A') for k in expect}
    dir_ok = {k: true_dir[k] == expect[k] for k in expect}
    dir_match = {k: pred_dir[k] == true_dir[k] for k in expect}
    mt_all, mp_all = gap(list(rows.keys()))
    errs = {k: float(np.mean([abs(mp_all[k][i] - mt_all[k][i]) / abs(mt_all[k][i])
                              for i in range(len(mt_all[k])) if abs(mt_all[k][i]) > 1e-9]))
            for k in expect}
    errs_by_type = {}
    for typ, dl in [('bilzh', human_docs + ai_docs), ('intent', sorted(
            {str(DOC[i]) for i in np.where(VAL)[0] if str(DOC[i]).startswith('intent')})),
            ('selfgen', sorted({str(DOC[i]) for i in np.where(VAL)[0] if str(DOC[i]).startswith('self')}))]:
        mt_t, mp_t = gap(dl)
        errs_by_type[typ] = {k: float(np.mean([abs(mp_t[k][i] - mt_t[k][i]) / abs(mt_t[k][i])
                                               for i in range(len(mt_t[k])) if abs(mt_t[k][i]) > 1e-9]))
                             for k in expect}
    errs_long = errs_by_type['bilzh']
    worst = max(errs_long.values())

    dir_all = all(dir_ok.values()) and all(dir_match.values())
    if dir_all and worst <= 0.3:
        g1 = '严格通过'
    elif dir_all and worst <= 0.5:
        g1 = '降级通过'
    else:
        g1 = '否定'
    print(f'G1: 真方向 {true_dir} 预测方向 {pred_dir} 方向一致 {dir_match}')
    print(f'    errs_long {errs_long} worst {worst:.3f}——errs_all {errs}——by_type {errs_by_type} → {g1}')

    sg = [str(DOC[i]) for i in np.where(VAL)[0] if str(DOC[i]).startswith('self')]
    mt_s, mp_s = gap(sg)
    sg_err = {k: float(np.mean([abs(mp_s[k][i] - mt_s[k][i]) / abs(mt_s[k][i])
                                for i in range(len(mt_s[k])) if abs(mt_s[k][i]) > 1e-9]))
              for k in expect}
    print(f'selfgen held-out 缺口: {sg_err}')

    out = {'g1': g1, 'worst_err_long': worst, 'errs_long': errs_long,
           'errs_all': errs, 'errs_by_type': errs_by_type,
           'true_dir': true_dir, 'pred_dir': pred_dir, 'dir_ok': dir_ok, 'dir_match': dir_match,
           'human_docs': human_docs, 'ai_docs': ai_docs, 'selfgen_err': sg_err,
           'val_err': vm, 'best_score': best,
           'note': 'G1 误差基准=长文档(bilzh 缺口定义域)；分层：≤0.3 严格/≤0.5 降级（G2 加严+G3 封顶部分）/否则否定→RWR；'
                   '偏差声明①：原"held-out 全文档"基准拆分为全文档/长文档双报；'
                   '偏差声明②：代理损失加入 doc 级相对软密度辅助项（MSE 回归向均值衰减→密度低估的针对性修复）'}
    (OUT / 'gate_g1.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print('落盘 surrogate.pt + surrogate_scaler.json + gate_g1.json ✓')


if __name__ == '__main__':
    main()
