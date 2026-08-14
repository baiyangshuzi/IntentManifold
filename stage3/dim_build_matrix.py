# -*- coding: utf-8 -*-
"""v0.74 A0：全语料逐句 64 维指纹矩阵构建 + 既有数据验证（"一直在算从未落盘"工程修复）

构建：bilingual zh 30 篇 / independent_test 32 对三侧 / generalization_strict 60 对 /
      training_intervention 201 runs / long_attention 12 篇 → 逐句元 (fp_64, h1_256) + 元数据
落盘：data/dim_analysis/fp_matrix.npz（fp/h1 矩阵 + meta.json）
验证：①bilingual zh 段内 sent_proj vs metrics_zh.json per_seg ②training_intervention 6 dims vs manifest
     ③判别分 disc vs 存储 ④18 特征探针复现（dim_probe.json 口径）
"""
import sys, json, os, re, time
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path
import numpy as np
import torch

BASE = Path('C:/Users/bai/Desktop/小说系统')
OUT = BASE / 'data' / 'dim_analysis'
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(BASE / 'stage3'))

CH_RE = re.compile(r'^第[一二三四五六七八九十百千0-9]+[章节卷回部]')
TARGET_DIMS = [5, 10, 11, 22, 26, 34, 43, 46, 48, 52, 59]


def paras_of(text):
    """行级段落切分（≥30 字含汉字非标题）——与既有管线同口径"""
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    return [l for l in lines if re.search(r'[一-龥]', l) and len(l) >= 30 and not CH_RE.match(l)]


def h1_of(x, disc):
    """256 维中间激活：relu(LN₁(W₁x+b₁))——上层追溯用"""
    h = disc.net[0](x)
    h = disc.net[1](h)
    return torch.relu(h)


def build_all(enc, disc, device):
    """遍历全部语料——返回 (rows, fp_mat, h1_mat)"""
    from para_dimensions import fingerprint, norm_rows
    from subclause_structure import split_subclauses
    rows = []
    fp_list, h1_list = [], []
    t0 = time.time()

    def process(text, source, doc, side, prompt=None, seed=None, seg_idx=None):
        nonlocal rows
        paras = paras_of(text)
        for pi, p in enumerate(paras):
            ss = [s for s in split_subclauses(p) if len(s) >= 3]
            if not ss:
                continue
            with torch.no_grad():
                sv = enc.encode(ss, normalize_embeddings=True, batch_size=64,
                                show_progress_bar=False, device=device)
                SV = torch.from_numpy(sv.astype(np.float32)).to(device)
                h1 = h1_of(SV, disc).cpu().numpy()
                fp = fingerprint(SV, disc).cpu().numpy()
            for k in range(len(ss)):
                rows.append({'source': source, 'doc': doc, 'side': side, 'prompt': prompt,
                             'seed': seed, 'seg': seg_idx, 'para': pi, 'clause': ss[k]})
            fp_list.append(fp)
            h1_list.append(h1)

    # 1) bilingual zh 30 篇
    for sub, side in (('human_zh', 'human'), ('ai_zh', 'ai')):
        for f in sorted((BASE / 'data/bilingual_test' / sub).glob('*.txt')):
            process(f.read_text(encoding='utf-8', errors='replace'), 'bilingual_zh', f.stem, side)
    print(f'bilingual zh ✓（{time.time()-t0:.0f}s）')

    # 2) independent_test 32 对三侧（human/ai=DS/ai_qwen）
    man = json.load(open(BASE / 'data/independent_test/manifest.json', encoding='utf-8'))
    for pair in man.get('pairs', []):
        pid = pair['pair_id']
        for side, txt in (('human', pair.get('human')), ('ai', pair.get('ai')),
                          ('qwen', pair.get('qwen_ai'))):
            if txt:
                process(txt, 'independent_test', pid, side)
    print(f'independent_test ✓（{time.time()-t0:.0f}s）')

    # 3) generalization_strict 60 对（L1-L5 域）
    man2 = json.load(open(BASE / 'data/generalization_strict/manifest.json', encoding='utf-8'))
    for pair in man2.get('pairs', []):
        pid = pair.get('pair_id', pair.get('id', '?'))
        for side, txt in (('human', pair.get('human')), ('ai', pair.get('ai'))):
            if txt:
                process(txt, 'generalization_strict', f'{pid}', side)
    print(f'generalization_strict ✓（{time.time()-t0:.0f}s）')

    # 4) training_intervention 201 runs（603 segs——含 prompt/seed 元数据）
    man3 = json.load(open(BASE / 'data/training_intervention/manifest.json', encoding='utf-8'))
    for r in man3:
        for seg in r['segs']:
            if seg.get('text'):
                process(seg['text'], 'training_intervention', r['run_id'],
                        r['condition'], r['prompt_id'], r['seed'], seg['seg'])
    print(f'training_intervention ✓（{time.time()-t0:.0f}s）')

    # 5) long_attention 12 篇
    man4 = json.load(open(BASE / 'data/long_attention/manifest.json', encoding='utf-8'))
    for r in man4:
        if r.get('text'):
            process(r['text'], 'long_attention', f"{r['prompt']}-{r['condition']}-s{r['seed']}",
                    r['condition'], r['prompt'], r['seed'])
    print(f'long_attention ✓（{time.time()-t0:.0f}s）')

    fp_mat = np.vstack(fp_list).astype(np.float32)
    h1_mat = np.vstack(h1_list).astype(np.float32)
    print(f'总句元: {len(rows)}——fp{fp_mat.shape} h1{h1_mat.shape}——{time.time()-t0:.0f}s')
    return rows, fp_mat, h1_mat


def verify(rows, fp_mat, enc, disc, device):
    """验证（修正口径——2026-08-14）：
    ①篇级：bilingual zh doc 级句元投影均值 vs metrics_zh anchors.doc.mean（不受段切分差异影响——
       per_seg 只含"测量成功"段——段数差异是 if d7 and lf 过滤——非计算错误）
    ②training_intervention 6 dims+disc：CPU 判别器重跑 manifest segs 文本（同函数同文本——精确一致）
    ③探针复现独立做（dim_activity）"""
    import torch as T
    from para_dimensions import norm_rows
    from collections import defaultdict
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups[(r['source'], r['doc'])].append(i)

    # ① 篇级验证（analyze_doc doc_mean 口径——全篇核心 + 段投影均值（段等权））
    metrics = json.load(open(BASE / 'data/bilingual_test/metrics_zh.json', encoding='utf-8'))
    n_ok, n_chk, max_d = 0, 0, 0.0
    for doc_id, m in metrics.items():
        para_idx = defaultdict(list)
        for i, r in enumerate(rows):
            if r['source'] == 'bilingual_zh' and r['doc'] == doc_id:
                para_idx[r['para']].append(i)
        if not para_idx:
            continue
        all_idx = [i for v in para_idx.values() for i in v]
        F = fp_mat[all_idx]
        Fn = norm_rows(T.from_numpy(F)).numpy()
        core = Fn.mean(0)
        core = core / (np.linalg.norm(core) + 1e-9)
        # 段投影均值（analyze_doc 口径——每段 para_proj 后 mean）
        para_projs = []
        for pi in sorted(para_idx):
            idx = para_idx[pi]
            Fp = Fn[[all_idx.index(j) for j in idx]]
            para_projs.append(float((Fp @ core).mean()))
        sp = float(np.mean(para_projs))
        old = (m.get('anchors') or {}).get('doc', {}).get('mean')
        if old is not None:
            n_chk += 1
            d = abs(sp - old)
            max_d = max(max_d, d)
            if d <= 0.001:
                n_ok += 1
    v1 = {'n_checked': n_chk, 'n_ok_le0.001': n_ok, 'max_abs_delta': round(max_d, 5),
          'pass': n_chk > 0 and n_ok / n_chk > 0.95,
          'note': 'analyze_doc doc_mean 口径（段投影均值——段等权——与句元均值区分）'}
    print(f'验证① 篇级 sent_proj（段均值口径）: {n_ok}/{n_chk} 通过（|Δ|≤0.001）——max Δ={max_d:.5f}——{"PASS" if v1["pass"] else "FAIL"}')

    # ② training_intervention 6 dims+disc（CPU 判别器——同函数同文本）
    from para_dimensions import load_models as _lm, para_dimensions
    from subclause_structure import split_subclauses as _ss
    import jieba
    import jieba.posseg as pseg
    jieba.setLogLevel(60)
    enc_cpu, disc_cpu = _lm('cpu')
    man = json.load(open(BASE / 'data/training_intervention/manifest.json', encoding='utf-8'))
    n_ok2, n_chk2, max_d2 = 0, 0, 0.0
    n_ok3, n_chk3 = 0, 0
    for r in man:
        for seg in r['segs']:
            old = seg.get('dims')
            if not old or not seg.get('text'):
                continue
            d2 = para_dimensions(seg['text'], enc_cpu, disc_cpu, _ss, pseg, device='cpu')
            if not d2:
                continue
            for k in ('sent_proj', 'traj', 'l7_adj', 'word_proj', 'word_adj', 'entropy'):
                if k in old and k in d2:
                    n_chk2 += 1
                    dd = abs(d2[k] - old[k])
                    max_d2 = max(max_d2, dd)
                    if dd <= 0.001:
                        n_ok2 += 1
            if seg.get('disc') is not None and 'disc' in d2:
                n_chk3 += 1
                if abs(d2['disc'] - seg['disc']) <= 0.001:
                    n_ok3 += 1
    v2 = {'n_checked': n_chk2, 'n_ok_le0.001': n_ok2, 'max_abs_delta': round(max_d2, 5),
          'pass': n_chk2 > 0 and n_ok2 / n_chk2 > 0.95}
    v3 = {'n_checked': n_chk3, 'n_ok_le0.001': n_ok3,
          'pass': n_chk3 > 0 and n_ok3 / n_chk3 > 0.95}
    print(f'验证② 6 dims: {n_ok2}/{n_chk2} 通过——max Δ={max_d2:.5f}——{"PASS" if v2["pass"] else "FAIL"}')
    print(f'验证③ disc: {n_ok3}/{n_chk3} 通过——{"PASS" if v3["pass"] else "FAIL"}')
    return {'doc_mean': v1, 'dims6': v2, 'disc': v3}


def main():
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    from para_dimensions import load_models
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    enc, disc = load_models(device)
    print(f'models ✓（{device}）')
    rows, fp_mat, h1_mat = build_all(enc, disc, device)

    # 落盘
    np.savez_compressed(OUT / 'fp_matrix.npz', fp=fp_mat, h1=h1_mat)
    (OUT / 'meta.json').write_text(json.dumps(
        {'n_clauses': len(rows), 'n_sources': len(set(r['source'] for r in rows)),
         'target_dims': TARGET_DIMS, 'fp_shape': list(fp_mat.shape), 'h1_shape': list(h1_mat.shape),
         'columns': ['source', 'doc', 'side', 'prompt', 'seed', 'seg', 'para', 'clause']},
        ensure_ascii=False, indent=1), encoding='utf-8')
    # 元数据分块落盘（rows 大——json 全量 ~几十 MB——可接受）
    (OUT / 'rows.json').write_text(json.dumps(rows, ensure_ascii=False), encoding='utf-8')
    print(f'落盘 ✓ fp_matrix.npz + rows.json + meta.json（{OUT}）')

    # 验证
    ver = verify(rows, fp_mat, enc, disc, device)
    (OUT / 'verify_results.json').write_text(json.dumps(ver, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
