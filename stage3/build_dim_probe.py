# -*- coding: utf-8 -*-
"""v0.66-6 指纹维度探针分析（用户：剩下 24 维为什么没法分析——探针）

步骤：
  1. 扩展特征集（原 6 → +12 新特征）：词性分布（名/动/形/副/代比例）/句长均值/标点密度/数字密度/
     情绪词密度/引语密度/词汇多样性/句元数/叹问号/破折号/四字格/虚词比例
  2. 64 维 × 18 特征 Spearman 相关——看原 24 个未解释维度有多少被新特征解释
  3. 对仍无法解释的维度：极值句元对比（维度值 top/bottom 句元——打印样本观察）
产出：data/independent_test/dim_probe.json + 报告
"""
import sys, json, math, time
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path
import numpy as np

BASE = Path('C:/Users/bai/Desktop/小说系统')
OUT = BASE / 'data' / 'independent_test'
sys.path.insert(0, str(BASE / 'stage3'))

EMOTION = ['开心', '难过', '愤怒', '害怕', '紧张', '激动', '委屈', '幸福', '痛苦', '温暖',
           '孤独', '焦虑', '恐惧', '喜悦', '悲伤', '惊喜', '失望', '希望', '爱', '恨']
EMOTION += ['开心', '伤心', '生气', '担心', '高兴', '难过']


def main():
    t0 = time.time()
    import os
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    import torch
    from para_dimensions import load_models, fingerprint, norm_rows
    from subclause_structure import split_subclauses
    import jieba
    import jieba.posseg as pseg
    jieba.setLogLevel(60)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    enc, disc = load_models(DEVICE)
    print(f'v2 加载 ✓（{DEVICE}）')
    from scipy import stats as sc

    def measure(txt):
        """一段：64 维平均指纹 + 18 个语言特征"""
        ss = [s for s in split_subclauses(txt) if len(s) >= 3]
        if len(ss) < 2:
            return None
        with torch.no_grad():
            sv = enc.encode(ss, normalize_embeddings=True, batch_size=64,
                            show_progress_bar=False, device=DEVICE)
            SV = torch.from_numpy(sv.astype(np.float32)).to(DEVICE)
            F = norm_rows(fingerprint(SV, disc)).cpu().numpy()
        segs = [list(pseg.cut(s)) for s in ss]
        n = len(segs)
        def pos_ratio(flags):
            return sum(1 for seg in segs for w, f in seg if f[0] in flags) / max(n, 1)
        from collections import Counter
        all_words = [w for seg in segs for w, f in seg if len(w) >= 2]
        wc = Counter(all_words)
        text = txt
        feats = {
            # 词性分布
            'n_ratio': pos_ratio('n'), 'v_ratio': pos_ratio('v'), 'a_ratio': pos_ratio('a'),
            'd_ratio': pos_ratio('d'), 'r_ratio': pos_ratio('r'),
            # 句法/表层
            'sent_len_mean': float(np.mean([len(s) for s in ss])),
            'punct_density': sum(1 for c in text if c in '，。！？；：、') / max(len(text), 1),
            'digit_density': sum(1 for c in text if c.isdigit()) / max(len(text), 1),
            'emotion_density': sum(1 for w in all_words if w in EMOTION) / max(n, 1),
            'quote_density': text.count('"') + text.count('“') + text.count('”'),
            'ttr': len(set(all_words)) / max(len(all_words), 1),
            'n_sent': n,
            'exclaim_q': text.count('！') + text.count('？'),
            'dash': text.count('——') + text.count('—'),
            'fourchar': sum(1 for w in all_words if len(w) == 4) / max(n, 1),
            'func_ratio': pos_ratio('u') + pos_ratio('c') + pos_ratio('p'),
            'word_count': len(all_words),
            'conj_density': sum(1 for seg in segs for w, f in seg if f[0] == 'c') / max(n, 1),
        }
        return {'fp': F.mean(0), 'feats': feats}

    # ===== 数据（独立集白话——人类侧——与之前同源）=====
    ri = json.loads((OUT / 'manifest.json').read_text(encoding='utf-8'))
    samples = []
    for p in ri['pairs']:
        if p['domain'] == 'baihua':
            m = measure(p['human'])
            if m:
                samples.append(m)
    print(f'独立集白话人类段 {len(samples)} 段')

    FEATS = list(samples[0]['feats'].keys())
    fp_mat = np.array([s['fp'] for s in samples])
    feat_vals = {f: np.array([s['feats'][f] for s in samples]) for f in FEATS}

    # ===== 1. 64 维 × 18 特征相关 =====
    print('\n=== 探针 1：64 维 × 18 特征（新增 12 个）===')
    dim_feats = {}   # dim -> [(feat, rho)]
    for dim in range(64):
        fv = fp_mat[:, dim]
        sig = []
        for f in FEATS:
            rho, pv = sc.spearmanr(fv, feat_vals[f])
            if pv < 0.05 and abs(rho) > 0.3:
                sig.append((f, round(float(rho), 2)))
        dim_feats[dim] = sig
    explained = {dim for dim, sig in dim_feats.items() if sig}
    print(f'新特征集下可解释维度：{len(explained)}/64')
    remaining = [dim for dim in range(64) if dim not in explained]
    print(f'仍无法解释的维度：{len(remaining)} 个——{remaining}')
    # 原 24 个（旧 6 特征）有多少被新特征解释
    print('\n=== 原 24 个未解释维度的探针结果 ===')
    old6 = ['pron_density', 'conj_density', 'known_first', 'topic_cont', 'topic_density', 'len_std']
    old_explained = set()
    for dim in range(64):
        for f in old6:
            if f in [x[0] for x in dim_feats[dim]]:
                old_explained.add(dim)
    orig_24 = [d for d in range(64) if d not in old_explained]
    newly = [d for d in orig_24 if d in explained]
    still = [d for d in orig_24 if d not in explained]
    print(f'原 24 个：新特征解释 {len(newly)} 个——{newly}')
    print(f'仍未解释：{len(still)} 个——{still}')
    for d in newly:
        print(f'  维度 {d}: {dim_feats[d]}')

    # ===== 2. 极值句元对比（仍无法解释的维度）=====
    print('\n=== 探针 2：极值句元对比（仍未解释维度——top/bottom 句元）===')
    obs = {}
    if still:
        # 收集句元级数据（该维度值最高的句元 vs 最低）
        for d in still[:5]:
            sent_vals = []
            for p in ri['pairs']:
                if p['domain'] != 'baihua':
                    continue
                ss = [s for s in split_subclauses(p['human']) if len(s) >= 3]
                if len(ss) < 2:
                    continue
                with torch.no_grad():
                    sv = enc.encode(ss, normalize_embeddings=True, batch_size=64,
                                    show_progress_bar=False, device=DEVICE)
                    SV = torch.from_numpy(sv.astype(np.float32)).to(DEVICE)
                    F = norm_rows(fingerprint(SV, disc)).cpu().numpy()
                for si, s in enumerate(ss):
                    sent_vals.append((float(F[si, d]), s))
            sent_vals.sort(key=lambda x: x[0])
            tops = [s for _, s in sent_vals[-5:]]
            bots = [s for _, s in sent_vals[:5]]
            obs[d] = {'top': tops, 'bottom': bots}
            print(f'  维度 {d} 高值句元: {tops[0][:50]} …')
            print(f'  维度 {d} 低值句元: {bots[0][:50]} …')

    results = {'explained_dims': sorted(explained), 'remaining_dims': still,
               'newly_explained': newly, 'dim_feats': {str(d): v for d, v in dim_feats.items()},
               'extreme_obs': {str(d): v for d, v in obs.items()},
               'feat_list': FEATS}
    (OUT / 'dim_probe.json').write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'\n落盘 dim_probe.json（{int(time.time()-t0)}s）')


if __name__ == '__main__':
    main()
