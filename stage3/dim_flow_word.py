# -*- coding: utf-8 -*-
"""v0.74-4 词级意图流动 + 11 维全维度流动图（用户：测词级流动/其他维度流动）

词级：代表篇（人类 ZH-H01 vs AI ZH-A-DS-05）逐词 bge+判别器 → dim10/dim48 词级轨迹 + 跳跃/转折/转移熵
11 维：句元级 11 维激活热力图（人类 vs AI 代表篇）+ 11 维跳跃度全景（区分力表）
产出（中文名——论文储备）：词级流动图.png / 十一维流动热力图.png
"""
import sys, json, os, re
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats as sc

BASE = Path('C:/Users/bai/Desktop/小说系统')
OUT = BASE / 'data' / 'dim_analysis'
PAPER = Path('C:/Users/bai/Desktop/AB系统论文储备')
sys.path.insert(0, str(BASE / 'stage3'))
import dim_flow as DF  # hurst/paras_of
import dim_flow_sent as DFS  # transfer_entropy/turn_points

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
TARGET = [5, 10, 11, 22, 26, 34, 43, 46, 48, 52, 59]


def word_level(enc, disc, text, device='cpu'):
    """逐词 bge+判别器 → 64 维词指纹（词级轨迹）"""
    from para_dimensions import fingerprint
    import jieba
    words = [w for w in jieba.cut(text) if len(w) >= 2]
    with torch.no_grad():
        sv = enc.encode(words, normalize_embeddings=True, batch_size=64,
                        show_progress_bar=False, device=device)
        SV = torch.from_numpy(sv.astype(np.float32)).to(device)
        F = fingerprint(SV, disc).cpu().numpy()
    return words, F


def main():
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    from para_dimensions import load_models
    enc, disc = load_models('cpu')

    # ===== 词级流动（代表篇）=====
    d = np.load(OUT / 'fp_matrix.npz')
    fp = d['fp']
    rows = json.loads((OUT / 'rows.json').read_text(encoding='utf-8'))
    from collections import defaultdict
    doc_clauses = defaultdict(list)
    for i, r in enumerate(rows):
        if r['source'] == 'bilingual_zh':
            doc_clauses[r['doc']].append((r['para'], i))
    docs = {}
    for doc, items in doc_clauses.items():
        items.sort(key=lambda x: (x[0], x[1]))
        docs[doc] = [i for _, i in items]

    # 代表篇文本（从 bilingual txt 读——按句元顺序拼接段落）
    def doc_text(doc):
        from subclause_structure import split_subclauses
        f = BASE / 'data/bilingual_test' / ('human_zh' if doc.startswith('ZH-H') else 'ai_zh') / f'{doc}.txt'
        txt = f.read_text(encoding='utf-8', errors='replace')
        paras = DF.paras_of(txt)
        return '\n'.join(paras)

    reps = {'人类': 'ZH-H01', 'AI': 'ZH-A-DS-05'}
    word_stats = {}
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))
    for ax, (grp, doc) in zip(axes, reps.items()):
        words, Fw = word_level(enc, disc, doc_text(doc))
        w10, w48 = Fw[:, 10], Fw[:, 48]
        word_stats[grp] = {'n_words': len(words),
                           'dim10_jump': float(np.mean(np.abs(np.diff(w10)))),
                           'dim48_jump': float(np.mean(np.abs(np.diff(w48)))),
                           'te_10_to_48': DFS.transfer_entropy(w10, w48),
                           'turn10_steep': DFS.turn_points(w10)['peak_steep_mean'],
                           'turn48_steep': DFS.turn_points(w48)['peak_steep_mean']}
        n_show = min(400, len(w10))
        ax.plot(w10[:n_show], label='dim10 主题', color='#1f6fb2', lw=0.7)
        ax.plot(w48[:n_show], label='dim48 指代', color='#e67e22', lw=0.7)
        ax.set_title(f'{grp} 词级轨迹（{doc}——前 {n_show} 词）')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        print(f'词级 {grp}（{doc}）: {len(words)} 词——dim10跳跃={word_stats[grp]["dim10_jump"]:.4f} '
              f'dim48跳跃={word_stats[grp]["dim48_jump"]:.4f} TE={word_stats[grp]["te_10_to_48"]:.4f} '
              f'转折陡度10={word_stats[grp]["turn10_steep"]:.3f}/48={word_stats[grp]["turn48_steep"]:.3f}')
    plt.tight_layout()
    plt.savefig(PAPER / '词级流动图.png', dpi=150)
    plt.close()
    print('词级流动图 ✓')

    # ===== 11 维句元级流动热力图 =====
    fig2, axes2 = plt.subplots(2, 1, figsize=(15, 7))
    for ax, (grp, doc) in zip(axes2, reps.items()):
        idx = np.array(docs[doc])[:300]  # 前 300 句元
        Z = fp[idx][:, TARGET].T  # (11, n)
        im = ax.imshow(Z, aspect='auto', cmap='viridis', interpolation='nearest')
        ax.set_yticks(range(len(TARGET)))
        ax.set_yticklabels([f'dim{d}' for d in TARGET], fontsize=8)
        ax.set_xlabel('句元序号')
        ax.set_title(f'{grp}（{doc}）——11 维句元级流动热力图（前 300 句元）', fontsize=10)
        plt.colorbar(im, ax=ax, shrink=0.6)
    plt.tight_layout()
    plt.savefig(PAPER / '十一维流动热力图.png', dpi=150)
    plt.close()
    print('十一维流动热力图 ✓')

    # ===== 11 维跳跃度全景（哪些维度有区分力）=====
    jumps = {}
    for j in TARGET:
        h = [np.mean(np.abs(np.diff(fp[np.array(docs[x]), j]))) for x in
             sorted([x for x in docs if x.startswith('ZH-H')])]
        a = [np.mean(np.abs(np.diff(fp[np.array(docs[x]), j]))) for x in
             sorted([x for x in docs if x.startswith('ZH-A')])]
        u = sc.mannwhitneyu(h, a)
        d_ = (np.mean(h) - np.mean(a)) / np.sqrt((np.var(h) + np.var(a)) / 2 + 1e-9)
        jumps[j] = {'human': round(float(np.mean(h)), 4), 'ai': round(float(np.mean(a)), 4),
                    'd': round(float(d_), 3), 'p': round(float(u.pvalue), 4)}
    print('=== 11 维句元级跳跃度全景（人类 vs AI）===')
    for j in TARGET:
        v = jumps[j]
        print(f'  dim{j}: 人类 {v["human"]} vs AI {v["ai"]}——d={v["d"]:+.2f} p={v["p"]:.3f}')

    out = {'word': word_stats, 'dim11_jumps': jumps}
    (OUT / 'flow_word_analysis.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print('落盘 flow_word_analysis.json ✓')


if __name__ == '__main__':
    main()
