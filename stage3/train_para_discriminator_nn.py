# -*- coding: utf-8 -*-
"""v0.60 判别器升级为模型（用户定案：判别器需要训练为模型——选择合适的模型下载——bge-small-zh）

模型：bge-small-zh-v1.5（512 维中文专用——段落级判别）→ MLP（512→256→64→1）
训练：AI 段落（data/ai_clean_manifest——26477 段）vs 人类段落（**龙族为主——用户定案：
  龙族情绪化/人味明显方便学习——龙族 70% + 诛仙 30%**）
验收：5 折准确率/AUROC（对比手工特征 85.5%/0.934）+ 阈值权衡（人类误抓 vs AI 检出）
产出：data/para_discriminator_nn.pt（段落判别模型）+ 阈值报告
"""
import sys, os, json, re
sys.path.insert(0, 'stage3')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from pathlib import Path
import time

BASE = Path(__file__).resolve().parent.parent
DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"
import torch
import torch.nn as nn


def paras_of(path, kind):
    enc = 'gbk' if kind == 'zx' else 'utf-8'
    t = Path(path).read_text(encoding=enc, errors='ignore')
    if kind == 'zx':
        i = t.find('用户上传之内容开始'); t = t[i + 9:] if i >= 0 else t
    elif kind == 'lz':
        i = t.find('========正文========'); t = t[i + 9:] if i >= 0 else t
    return [l.strip() for l in t.split('\n') if len(l.strip()) >= 15 and not l.startswith('#')]


META_RE = re.compile(r'SYSTEM|USER|=====|角色|提示词|生成|细纲|决策|锚点|情绪弧线|主线|禁区|写作细则|记忆|指令|协议')


class ParaDiscNN(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 64), nn.LayerNorm(64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def main():
    # ===== 数据 =====
    manifest = json.loads((BASE / "data" / "ai_clean_manifest.json").read_text(encoding="utf-8"))
    ai_ps = []
    for rel in manifest:
        t = (BASE / "data" / "generations" / rel).read_text(encoding="utf-8", errors="ignore")
        for l in t.split('\n'):
            l = l.strip()
            if len(l) >= 15 and not l.startswith('#') and not META_RE.search(l):
                ai_ps.append(l)
    lz_ps = paras_of(BASE / "《龙族》【爱上阅读_www.isyd.net】.txt", 'lz')
    zx_ps = paras_of(BASE / "诛仙.txt", 'zx')
    # 人类以龙族为主（用户定案）——龙族 70% + 诛仙 30%
    import random
    rng = random.Random(0)
    n_hu = 6000
    n_lz = int(n_hu * 0.7)
    n_zx = n_hu - n_lz
    hu_ps = rng.sample(lz_ps, min(n_lz, len(lz_ps))) + rng.sample(zx_ps, min(n_zx, len(zx_ps)))
    ai_samp = rng.sample(ai_ps, n_hu)
    print(f'AI {len(ai_samp)} 段 | 人类 {len(hu_ps)} 段（龙族 {n_lz} + 诛仙 {n_zx}——用户定案龙族为主）')

    # ===== 段落向量（bge-small-zh——512 维——整段编码）=====
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer('BAAI/bge-small-zh-v1.5')
    enc.to(DEVICE); enc.eval()
    print('编码中（bge-small-zh——512 维）...')
    t0 = time.time()
    with torch.no_grad():
        X_ai = enc.encode(ai_samp, normalize_embeddings=True, batch_size=128, show_progress_bar=True,
                          device=DEVICE)
        X_hu = enc.encode(hu_ps, normalize_embeddings=True, batch_size=128, show_progress_bar=True,
                          device=DEVICE)
    X = np.concatenate([X_hu, X_ai]).astype(np.float32)
    y = np.concatenate([np.zeros(len(X_hu)), np.ones(len(X_ai))])
    print(f'向量: {X.shape}（{(time.time()-t0)/60:.1f} 分钟）')

    # ===== 训练（5 折——过拟合检查）=====
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    accs, aucs = [], []
    Xt = torch.from_numpy(X).to(DEVICE)
    yt = torch.from_numpy(y).to(DEVICE)
    for fold, (tr, va) in enumerate(cv.split(X, y)):
        model = ParaDiscNN(512).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        lossf = nn.BCEWithLogitsLoss()
        tr_t = torch.from_numpy(tr).to(DEVICE)
        va_t = torch.from_numpy(va).to(DEVICE)
        model.train()
        for ep in range(15):
            p = torch.randperm(len(tr_t), device=DEVICE)
            for i in range(0, len(tr_t), 1024):
                bi = tr_t[p[i:i + 1024]]
                logits = model(Xt[bi])
                loss = lossf(logits, yt[bi])
                opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            logits = model(Xt[va_t])
            probs = torch.sigmoid(logits).cpu().numpy()
        acc = ((probs > 0.5) == y[va]).mean()
        auc = roc_auc_score(y[va], probs)
        accs.append(acc); aucs.append(auc)
        print(f'fold{fold+1}: acc {acc*100:.1f}% | AUROC {auc:.3f}')
    print(f'\n5 折: acc {np.mean(accs)*100:.1f}%±{np.std(accs)*100:.1f} | AUROC {np.mean(aucs):.3f}'
          f'（对比手工特征 85.5%/0.934）')

    # ===== 全量重训 + 阈值权衡 =====
    model = ParaDiscNN(512).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    model.train()
    for ep in range(20):
        p = torch.randperm(len(X), device=DEVICE)
        for i in range(0, len(X), 1024):
            bi = p[i:i + 1024]
            loss = lossf(model(Xt[bi]), yt[bi])
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(Xt)).cpu().numpy()
    ph, pa = probs[:len(X_hu)], probs[len(X_hu):]
    print(f'\n=== 阈值权衡（判别人类段落误抓）===')
    for thr in [0.5, 0.6, 0.7, 0.8]:
        print(f'  阈值 {thr}: 人类误抓 {(ph>thr).mean()*100:.0f}% | AI 检出 {(pa>thr).mean()*100:.0f}%')

    # ===== 落盘 =====
    torch.save(model.state_dict(), BASE / "data" / "para_discriminator_nn.pt")
    json.dump({'acc5': float(np.mean(accs)), 'auc5': float(np.mean(aucs)),
               'thresholds': {str(t): {'误抓': float((ph > t).mean()), '检出': float((pa > t).mean())}
                              for t in [0.5, 0.6, 0.7, 0.8]}},
              open(BASE / "data" / "para_discriminator_nn_report.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print('\n落盘 data/para_discriminator_nn.pt + para_discriminator_nn_report.json')


if __name__ == '__main__':
    main()
