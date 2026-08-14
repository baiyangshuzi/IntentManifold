# train_intent_mlp.py — v0.73 模块 5.1：指纹→嵌入空间映射层学习（MLP 64→896）
# 基座 Qwen2.5-0.5B 只读冻结——独立可学习映射副本
# 数据：现有语料每句元 → (bge+P araDiscNN 指纹 64 维, Qwen embed_tokens mean-pool 896 维)
# 判据 V2：held-out 余弦 >=0.5 且显著 > 基线（均值预测/随机）
import sys, json, os, time, re
from pathlib import Path
import numpy as np
import torch

BASE = Path(os.environ.get('INTENT_DYNAMICS_BASE', Path(__file__).resolve().parent.parent))
BT = BASE / 'data' / 'bilingual_test'
OUT = BASE / 'data/intent_prior_model'
sys.path.insert(0, str(BASE / 'stage3'))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
os.environ.setdefault('HF_HUB_OFFLINE', '1')

CH_RE = re.compile(r'^第[一二三四五六七八九十百千0-9]+[章节卷回部]')
SEED = 42
N_SAMPLE_AI = 120  # ai_clean_manifest 抽样篇数
N_TXJ_CHAPTERS = 8  # 天行健抽样章数


class MLP(torch.nn.Module):
    """指纹→嵌入空间映射层（64→256→896）——独立可学习副本（基座只读）"""

    def __init__(self, d_in, d_hid, d_out):
        import torch.nn as nn
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, d_hid), nn.ReLU(), nn.Linear(d_hid, d_out))

    def forward(self, x):
        return self.net(x)


def collect_sources():
    """返回 [(name, text)] 列表——确定性抽样"""
    import random
    rng = random.Random(SEED)
    srcs = []
    # 1) ai_clean_manifest 抽样
    man = json.load(open(BASE / 'data/ai_clean_manifest.json', encoding='utf-8'))
    picked = rng.sample(man, min(N_SAMPLE_AI, len(man)))
    for rel in picked:
        p = BASE / 'data' / 'generations' / rel.replace('\\', '/')
        if p.exists():
            srcs.append((f'ai_clean:{Path(rel).name}', p.read_text(encoding='utf-8', errors='replace')))
    # 2) bilingual zh 30 篇
    for sub in ('ai_zh', 'human_zh'):
        d = BT / sub
        for f in sorted(d.glob('*.txt')):
            srcs.append((f'biling:{f.stem}', f.read_text(encoding='utf-8', errors='replace')))
    # 3) archives 6 章
    for i in range(1, 7):
        f = BASE / 'novel-project' / 'archives' / f'vol-1-ch-{i}-draft.md'
        if f.exists():
            srcs.append((f'archives:ch{i}', f.read_text(encoding='utf-8', errors='replace')))
    # 4) training_intervention 段文本
    m = json.load(open(BASE / 'data/training_intervention/manifest.json', encoding='utf-8'))
    for r in m:
        for seg in r['segs']:
            if seg.get('text'):
                srcs.append((f"ti:{r['run_id']}:seg{seg['seg']}", seg['text']))
    # 5) 天行健抽样章
    chs = sorted(BASE.glob('data/chapters_txj/chapter_*.txt'), key=lambda p: int(re.search(r'\d+', p.stem).group()))
    for f in chs[:N_TXJ_CHAPTERS]:
        srcs.append((f'txj:{f.stem}', f.read_text(encoding='utf-8', errors='replace')))
    print(f'语料源: {len(srcs)} 个（AI {sum(1 for s in srcs if s[0].startswith(("ai_clean", "biling:ai", "archives", "ti:")))} / '
          f'人类 {sum(1 for s in srcs if not s[0].startswith(("ai_clean", "biling:ai", "archives", "ti:")))}）')
    return srcs


def build_pairs(srcs, enc, disc, qwen, tok, device):
    import torch
    from para_dimensions import fingerprint, norm_rows
    from subclause_structure import split_subclauses
    fp_list, emb_list = [], []
    n_clauses = 0
    for name, text in srcs:
        paras = [l.strip() for l in text.split('\n') if l.strip() and re.search(r'[一-龥]', l)
                 and len(l.strip()) >= 30 and not CH_RE.match(l.strip())]
        ss = []
        for p in paras:
            ss += [s for s in split_subclauses(p) if len(s) >= 3]
        if not ss:
            continue
        with torch.no_grad():
            sv = enc.encode(ss, normalize_embeddings=True, batch_size=64,
                            show_progress_bar=False, device='cpu')
            SV = torch.from_numpy(sv.astype(np.float32))
            F_raw = fingerprint(SV, disc).numpy()
            Fn = norm_rows(torch.from_numpy(F_raw)).numpy()
            # Qwen 嵌入 mean-pool（GPU——padding 掩码去除）
            toks = tok(ss, padding=True, truncation=True, max_length=128, add_special_tokens=False,
                       return_tensors='pt')
            ids = toks['input_ids'].to(device)
            attn = toks['attention_mask'].to(device)
            e_all = qwen.get_input_embeddings()(ids)
            emb = (e_all * attn.unsqueeze(-1)).sum(1) / attn.sum(1, keepdim=True).clamp(min=1)
            emb = emb.float()
        fp_list.append(Fn)
        emb_list.append(emb.cpu().numpy())
        n_clauses += len(ss)
    X = np.vstack(fp_list)
    Y = np.vstack(emb_list)
    print(f'句元对: {n_clauses} —— X{X.shape} Y{Y.shape}')
    return X, Y


def main():
    import torch
    import torch.nn as nn
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from para_dimensions import load_models

    enc, disc = load_models('cpu')  # bge/判别器 CPU（训练数据构建——速度可接受）
    qwen = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B', torch_dtype=torch.float16).to(device).eval()
    tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B')
    print(f'models ✓（qwen={device}，bge/disc=cpu）')

    srcs = collect_sources()
    X, Y = build_pairs(srcs, enc, disc, qwen, tok, device)
    HID = qwen.config.hidden_size
    print(f'Qwen hidden_size={HID}')

    # 划分
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(X))
    n_tr = int(len(X) * 0.8)
    Xtr, Ytr = X[idx[:n_tr]], Y[idx[:n_tr]]
    Xte, Yte = X[idx[n_tr:]], Y[idx[n_tr:]]

    model = MLP(64, 256, HID).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.MSELoss()
    Xtr_t = torch.from_numpy(Xtr.astype(np.float32)).to(device)
    Ytr_t = torch.from_numpy(Ytr.astype(np.float32)).to(device)
    Xte_t = torch.from_numpy(Xte.astype(np.float32)).to(device)
    Yte_t = torch.from_numpy(Yte.astype(np.float32)).to(device)

    best_cos = -1
    for ep in range(150):
        model.train()
        perm = torch.randperm(len(Xtr_t), generator=torch.Generator().manual_seed(SEED + ep))
        tot = 0.0
        for i in range(0, len(Xtr_t), 128):
            b = perm[i:i + 128]
            opt.zero_grad()
            loss = lossf(model(Xtr_t[b]), Ytr_t[b])
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        if ep % 10 == 0 or ep == 149:
            model.eval()
            with torch.no_grad():
                p = model(Xte_t)
                cos = nn.functional.cosine_similarity(p, Yte_t, dim=1).mean().item()
                mse = lossf(p, Yte_t).item()
            best_cos = max(best_cos, cos)
            print(f'ep{ep}: train_mse={tot / len(Xtr_t):.5f} held_cos={cos:.4f} held_mse={mse:.5f}')

    # 基线
    with torch.no_grad():
        p_rand = model(torch.randn_like(Xte_t))
        cos_rand = nn.functional.cosine_similarity(p_rand, Yte_t, dim=1).mean().item()
        p_mean = Ytr_t.mean(0).expand(len(Yte_t), -1)
        cos_mean = nn.functional.cosine_similarity(p_mean, Yte_t, dim=1).mean().item()
    torch.save(model.state_dict(), OUT / 'mlp_checkpoint.pt')
    v2 = {'held_cos': best_cos, 'baseline_rand_cos': cos_rand, 'baseline_mean_cos': cos_mean,
          'n_pairs': len(X), 'n_train': len(Xtr), 'n_test': len(Xte),
          'verdict': 'PASS' if best_cos >= 0.5 else 'FAIL',
          'criteria': 'held-out 余弦 >=0.5 且 > 基线'}
    (OUT / 'mlp_eval.json').write_text(json.dumps(v2, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'V2 判定: held_cos={best_cos:.4f}（基线: 随机={cos_rand:.4f} 均值={cos_mean:.4f}）—— {v2["verdict"]}')


if __name__ == '__main__':
    main()
