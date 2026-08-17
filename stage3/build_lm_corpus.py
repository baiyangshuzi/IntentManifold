# -*- coding: utf-8 -*-
"""v0.93 LM 流语料构建（用户决策①——毛选扩充——train_lm_corpus.json）

来源（人类论说文/哲学——与目标测量域对齐）：
1. bilingual_zh human_zh 10 篇（chunk_paragraphs——400-800 字符块）
2. intent_corpus.pos 73 段（现成 400-800 字哲学切段）
3. 毛选扩充 3 卷（006-反对本本主义 / 026-论持久战 / 043-新民主主义论——GitHub 同源——
   chunk_paragraphs——单卷解析失败跳过不阻塞）

输出：data/dim_analysis/train_lm_corpus.json {segments:[{text,src}], stats}
LM 批 = 4×512 token 块（训练循环内按字符窗切 token——上限 512）。
"""
import sys, json, time
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path
import numpy as np
from train_reg_common import chunk_paragraphs

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / 'data' / 'dim_analysis'
V090 = BASE / 'data' / 'v090_sources'
BIL = BASE / 'data' / 'bilingual_test' / 'human_zh'
MAO_FILES = ['006-反对本本主义.txt', '026-论持久战.txt', '043-新民主主义论.txt']

t0 = time.time()
segs = []

# 1. bilingual human_zh
if BIL.is_dir():
    for f in sorted(BIL.glob('*.txt')):
        t = f.read_text(encoding='utf-8', errors='replace')
        for c in chunk_paragraphs(t):
            segs.append({'text': c, 'src': f'bilzh:{f.stem}'})
    print(f'bilingual human_zh: {sum(1 for s in segs if s["src"].startswith("bilzh:"))} 块')

# 2. intent pos
corpus = json.loads((OUT / 'intent_corpus.json').read_text(encoding='utf-8'))
for i, p in enumerate(corpus['pos']):
    segs.append({'text': p['text'], 'src': f"intent_pos:{i}"})
print(f'intent pos: {len(corpus["pos"])} 段')

# 3. 毛选扩充
for fname in MAO_FILES:
    fp = V090 / fname
    if not fp.exists():
        print(f'  跳过（不存在）: {fname}')
        continue
    try:
        t = fp.read_text(encoding='utf-8', errors='replace')
        cs = chunk_paragraphs(t)
        for c in cs:
            segs.append({'text': c, 'src': f'mao:{fname[:3]}'})
        print(f'  毛选 {fname}: {len(cs)} 块')
    except Exception as e:
        print(f'  跳过（解析失败）: {fname} — {e}')

# 去重（近重复全文）
seen, uniq = set(), []
for s in segs:
    k = s['text'][:80]
    if k in seen:
        continue
    seen.add(k)
    uniq.append(s)
segs = uniq

n_chars = sum(len(s['text']) for s in segs)
print(f'总计 {len(segs)} 段 / {n_chars} 字符（{time.time()-t0:.0f}s）')
out = {'segments': segs, 'stats': {'n_segments': len(segs), 'n_chars': n_chars,
                                   'src_counts': {}},
       'note': 'LM 流语料——人类论说文/哲学——与目标测量域对齐；毛选 3 卷 GitHub 同源（sources.md 延续）'}
for s in segs:
    out['stats']['src_counts'][s['src']] = out['stats']['src_counts'].get(s['src'], 0) + 1
(OUT / 'train_lm_corpus.json').write_text(json.dumps(out, ensure_ascii=False), encoding='utf-8')
print('落盘 train_lm_corpus.json ✓')
