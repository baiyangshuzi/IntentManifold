# -*- coding: utf-8 -*-
"""v0.74-5 64 维全维度全景分析（用户：把 64 个维度都分析一遍数据）

每维（0-63）：①活性（方差/死维度/seed 稳定性）②跨域差异（白话 vs 时评人类 d）
③人机差异（bilingual zh human vs AI d）④双模型方向一致（DS/Qwen）
⑤句元级跳跃度（人类 vs AI d）⑥可解释性（18 特征 Spearman 显著特征数）
产出：64 维全景表（json）+ 排序图（中文名——论文储备）
"""
import sys, json, os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path
import numpy as np
from scipy import stats as sc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path(os.environ.get('INTENT_DYNAMICS_BASE', Path(__file__).resolve().parent.parent))
OUT = BASE / 'data' / 'dim_analysis'
PAPER = Path('C:/Users/bai/Desktop/AB系统论文储备')
sys.path.insert(0, str(BASE / 'stage3'))

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def cohens_d(a, b):
    return (np.mean(a) - np.mean(b)) / np.sqrt((np.var(a) + np.var(b)) / 2 + 1e-9)


def main():
    d = np.load(OUT / 'fp_matrix.npz')
    fp = d['fp']
    rows = json.loads((OUT / 'rows.json').read_text(encoding='utf-8'))
    # 已解释维度（两口径并集——53 维）
    probe = json.load(open(BASE / 'data/independent_test/dim_probe.json', encoding='utf-8'))
    explained_18 = set(probe.get('explained_dims', []))
    mech = json.load(open(BASE / 'data/independent_test/language_mechanism.json', encoding='utf-8'))
    explained_6 = set()
    for f, dims in (mech.get('dim_groups') or {}).items():
        for x in dims:
            explained_6.add(x[0] if isinstance(x, (list, tuple)) else x)
    explained = explained_18 | explained_6
    target11 = [5, 10, 11, 22, 26, 34, 43, 46, 48, 52, 59]

    # 索引分组
    from collections import defaultdict
    doc_clauses = defaultdict(list)
    for i, r in enumerate(rows):
        if r['source'] == 'bilingual_zh':
            doc_clauses[r['doc']].append((r['para'], i))
    docs = {}
    for doc, items in doc_clauses.items():
        items.sort(key=lambda x: (x[0], x[1]))
        docs[doc] = [i for _, i in items]
    hu = [i for i, r in enumerate(rows) if r['source'] == 'bilingual_zh' and r['side'] == 'human']
    ai = [i for i, r in enumerate(rows) if r['source'] == 'bilingual_zh' and r['side'] == 'ai']
    bai = [i for i, r in enumerate(rows) if r['source'] == 'independent_test' and r['side'] == 'human' and 'B-' in r['doc']]
    ship = [i for i, r in enumerate(rows) if r['source'] == 'independent_test' and r['side'] == 'human' and 'S-' in r['doc']]
    ds = [i for i, r in enumerate(rows) if r['source'] == 'independent_test' and r['side'] == 'ai' and 'B-' in r['doc']]
    qw = [i for i, r in enumerate(rows) if r['source'] == 'independent_test' and r['side'] == 'qwen' and 'B-' in r['doc']]

    # seed 稳定性（none/b03/b05 同 prompt 3 seed——每维跨 seed 方差）
    seed_groups = defaultdict(list)
    for i, r in enumerate(rows):
        if r['source'] == 'training_intervention' and r['side'] in ('none', 'b03', 'b05'):
            seed_groups[(r['side'], r['prompt'], r['seg'])].append((r['seed'], i))
    within_var = np.zeros(64)
    n_groups = 0
    for key, items in seed_groups.items():
        by_seed = defaultdict(list)
        for s, i in items:
            by_seed[s].append(i)
        if len(by_seed) < 2:
            continue
        n_groups += 1
        means = [fp[idx].mean(0) for idx in by_seed.values()]
        within_var += np.array(means).var(0)
    within_var = within_var / max(n_groups, 1)

    v_act = fp.var(0)
    seed_ratio = within_var / (v_act + 1e-9)
    p5_var = np.percentile(v_act, 5)
    med_ratio = np.median(seed_ratio)

    # 句元级跳跃（每篇）
    def jump_d(dim):
        h = [np.mean(np.abs(np.diff(fp[np.array(docs[x]), dim]))) for x in
             sorted([x for x in docs if x.startswith('ZH-H')])]
        a = [np.mean(np.abs(np.diff(fp[np.array(docs[x]), dim]))) for x in
             sorted([x for x in docs if x.startswith('ZH-A')])]
        return cohens_d(h, a), sc.mannwhitneyu(h, a).pvalue

    # 18 特征探针（每维显著特征数——独立集白话人类段）
    import jieba, jieba.posseg as pseg
    from subclause_structure import split_subclauses as _ss
    jieba.setLogLevel(60)
    man = json.load(open(BASE / 'data/independent_test/manifest.json', encoding='utf-8'))
    texts = {}
    for pair in man.get('pairs', []):
        if 'B-' in pair['pair_id'] and pair.get('human'):
            texts[pair['pair_id']] = pair['human']
    from collections import Counter
    EMOTION = set(['开心', '难过', '愤怒', '害怕', '紧张', '激动', '委屈', '幸福', '痛苦', '温暖',
                   '孤独', '焦虑', '恐惧', '喜悦', '悲伤', '惊喜', '失望', '希望', '爱', '恨',
                   '伤心', '生气', '担心', '高兴'])
    def mf(txt):
        ss = [s for s in _ss(txt) if len(s) >= 3]
        if len(ss) < 2:
            return None
        segs = [list(pseg.cut(s)) for s in ss]
        n = len(segs)
        def pr(fs):
            return sum(1 for seg in segs for w, f in seg if f[0] in fs) / max(n, 1)
        aw = [w for seg in segs for w, f in seg if len(w) >= 2]
        return {'n_ratio': pr('n'), 'v_ratio': pr('v'), 'a_ratio': pr('a'), 'd_ratio': pr('d'),
                'r_ratio': pr('r'), 'sent_len_mean': float(np.mean([len(s) for s in ss])),
                'punct_density': sum(1 for c in txt if c in '，。！？；：、') / max(len(txt), 1),
                'digit_density': sum(1 for c in txt if c.isdigit()) / max(len(txt), 1),
                'emotion_density': sum(1 for w in aw if w in EMOTION) / max(n, 1),
                'quote_density': txt.count('"') + txt.count('“') + txt.count('”'),
                'ttr': len(set(aw)) / max(len(aw), 1), 'n_sent': n,
                'exclaim_q': txt.count('！') + txt.count('？'), 'dash': txt.count('——') + txt.count('—'),
                'fourchar': sum(1 for w in aw if len(w) == 4) / max(n, 1),
                'func_ratio': pr('u') + pr('c') + pr('p'), 'word_count': len(aw),
                'conj_density': sum(1 for seg in segs for w, f in seg if f[0] == 'c') / max(n, 1)}
    feats_map = {}
    for doc in texts:
        m = mf(texts[doc])
        if m:
            feats_map[doc] = m
    # 段级 fp（与特征对齐）
    seg_fp = defaultdict(list)
    for i, r in enumerate(rows):
        if r['source'] == 'independent_test' and r['side'] == 'human' and 'B-' in r['doc']:
            seg_fp[r['doc']].append(i)
    docs_ok = [x for x in feats_map if x in seg_fp]
    Fp_ok = np.array([fp[seg_fp[x]].mean(0) for x in docs_ok])
    F18 = np.array([[feats_map[x][k] for k in feats_map[x]] for x in docs_ok])
    n_feat = F18.shape[1]
    n_sig_feat = np.zeros(64, int)
    for j in range(64):
        for b in range(n_feat):
            rho, p = sc.spearmanr(Fp_ok[:, j], F18[:, b])
            if p < 0.05 and abs(rho) > 0.3:
                n_sig_feat[j] += 1

    # ===== 64 维全景 =====
    rows64 = []
    for j in range(64):
        dead = bool(v_act[j] < p5_var)
        noisy = bool(seed_ratio[j] > med_ratio * 3)
        d_dom = cohens_d(fp[bai, j], fp[ship, j])
        d_hum = cohens_d(fp[hu, j], fp[ai, j])
        _, p_hum = sc.ttest_ind(fp[hu, j], fp[ai, j], equal_var=False)
        d_ds = cohens_d(fp[ds, j], fp[hu, j])
        d_qw = cohens_d(fp[qw, j], fp[hu, j])
        d_jump, p_jump = jump_d(j)
        status = '死' if dead else ('噪声' if noisy else '活性')
        expl = '已解释' if j in explained else ('未解释11' if j in target11 else '未解释')
        rows64.append({'dim': j, 'var': round(float(v_act[j]), 5), 'seed_ratio': round(float(seed_ratio[j]), 3),
                       'status': status, 'domain_d': round(float(d_dom), 3),
                       'human_ai_d': round(float(d_hum), 3), 'human_ai_p': round(float(p_hum), 4),
                       'ds_d': round(float(d_ds), 3), 'qwen_d': round(float(d_qw), 3),
                       'dual_consistent': bool((d_ds > 0) == (d_qw > 0)),
                       'jump_d': round(float(d_jump), 3), 'jump_p': round(float(p_jump), 4),
                       'n_sig_feat': int(n_sig_feat[j]), 'explained': expl})

    # 排序图：人机 d（64 维）
    rows_sorted = sorted(rows64, key=lambda x: -abs(x['human_ai_d']))
    fig, ax = plt.subplots(figsize=(12, 9))
    dims = [r['dim'] for r in rows_sorted]
    ds_ = [r['human_ai_d'] for r in rows_sorted]
    colors = []
    for r in rows_sorted:
        if r['dim'] in target11:
            colors.append('#e67e22')  # 未解释 11 维——橙
        elif r['status'] != '活性':
            colors.append('#999')     # 死/噪声——灰
        else:
            colors.append('#4da6ff')  # 已解释/活性——蓝
    ax.barh(range(64), ds_, color=colors, edgecolor='#555', lw=0.3)
    ax.set_yticks(range(64))
    ax.set_yticklabels([f'dim{d}' for d in dims], fontsize=6.5)
    ax.axvline(0, color='#333', lw=1)
    ax.set_xlabel('人机差异 d（bilingual zh——人类 vs AI——句元级）')
    ax.set_title('64 维全维度人机差异全景（橙=未解释 11 维——蓝=已解释/活性——灰=死/噪声）')
    ax.grid(axis='x', alpha=0.25)
    plt.tight_layout()
    plt.savefig(PAPER / '六十四维全景图.png', dpi=150)
    plt.close()

    (OUT / 'dim64_full.json').write_text(json.dumps(
        {'rows': rows64, 'explained_n': len(explained), 'target11': target11,
         'n_dead': sum(1 for r in rows64 if r['status'] == '死'),
         'n_noisy': sum(1 for r in rows64 if r['status'] == '噪声')},
        ensure_ascii=False, indent=1), encoding='utf-8')

    # ===== 汇总 =====
    print('=== 64 维全景汇总 ===')
    print(f'状态: 活性 {sum(1 for r in rows64 if r["status"]=="活性")}——死 {sum(1 for r in rows64 if r["status"]=="死")}——噪声 {sum(1 for r in rows64 if r["status"]=="噪声")}')
    print(f'解释状态: 已解释 {sum(1 for r in rows64 if r["explained"]=="已解释")}——未解释11 {sum(1 for r in rows64 if r["explained"]=="未解释11")}——其他未解释 {sum(1 for r in rows64 if r["explained"]=="未解释")}')
    print(f'双模型方向一致: {sum(1 for r in rows64 if r["dual_consistent"])}/64')
    sig_d = [r for r in rows64 if abs(r['human_ai_d']) > 0.3 and r['human_ai_p'] < 0.05]
    print(f'人机差异显著（|d|>0.3 且 p<0.05）: {len(sig_d)}/64')
    print('人机 d 最强 10 维:')
    for r in rows_sorted[:10]:
        print(f'  dim{r["dim"]}: d={r["human_ai_d"]:+.2f}（{r["explained"]}——跳跃d={r["jump_d"]:+.2f}——特征{r["n_sig_feat"]}）')
    print('人机 d 最弱 10 维:')
    for r in rows_sorted[-10:]:
        print(f'  dim{r["dim"]}: d={r["human_ai_d"]:+.2f}（{r["explained"]}——{r["status"]}）')
    # 未解释 11 维全景
    print('未解释 11 维详细:')
    for r in rows64:
        if r['dim'] in target11:
            print(f'  dim{r["dim"]}: 人机d={r["human_ai_d"]:+.2f} 跳跃d={r["jump_d"]:+.2f} '
                  f'跨域d={r["domain_d"]:+.2f} 双模型{"一致" if r["dual_consistent"] else "✗"} 特征{r["n_sig_feat"]}')
    print('落盘 dim64_full.json + 六十四维全景图.png ✓')


if __name__ == '__main__':
    main()
