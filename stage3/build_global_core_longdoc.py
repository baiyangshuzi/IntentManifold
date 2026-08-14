# -*- coding: utf-8 -*-
"""v0.66-5 长文档篇核心曲线（最优构造 A——全篇句元 mean 核心）

数据：
  AI 3 篇：archives 6 章（8587 字）/doc2（4298）/doc3（4493）
  人类对照：天行健连续段（同量级）+ **天行健前 6 章（章节对照——用户补充）**
测量：篇核心 = 全篇句元指纹均值（归一化）——各段句元对篇核心投影——曲线
章节级：archives 6 章 vs 天行健前 6 章——每章均值/首末章差
产出：data/long_docs/global_core_anchor.json
"""
import sys, json, re, math
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path
import numpy as np

BASE = Path('C:/Users/bai/Desktop/小说系统')
OUT = BASE / 'data' / 'long_docs'
sys.path.insert(0, str(BASE / 'stage3'))


def main():
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

    def all_fp(paras):
        """全部句元指纹（未归一）"""
        Fs = []
        for p in paras:
            ss = [s for s in split_subclauses(p) if len(s) >= 3]
            if not ss:
                continue
            with torch.no_grad():
                sv = enc.encode(ss, normalize_embeddings=True, batch_size=64,
                                show_progress_bar=False, device=DEVICE)
                SV = torch.from_numpy(sv.astype(np.float32)).to(DEVICE)
                Fs.append(fingerprint(SV, disc).cpu().numpy())
        return np.vstack(Fs) if Fs else None

    def para_proj(p, core):
        ss = [s for s in split_subclauses(p) if len(s) >= 3]
        if not ss:
            return None
        with torch.no_grad():
            sv = enc.encode(ss, normalize_embeddings=True, batch_size=64,
                            show_progress_bar=False, device=DEVICE)
            SV = torch.from_numpy(sv.astype(np.float32)).to(DEVICE)
            F = norm_rows(fingerprint(SV, disc)).cpu().numpy()
        cn = core / (np.linalg.norm(core) + 1e-9)
        return float((F @ cn).mean())

    def analyze_doc(name, paras):
        """篇核心（全句元 mean）——各段投影"""
        F_all = all_fp(paras)
        if F_all is None or len(F_all) < 5:
            return None
        Fn = F_all / (np.linalg.norm(F_all, axis=1, keepdims=True) + 1e-9)
        core = Fn.mean(0)
        projs = [para_proj(p, core) for p in paras]
        projs = [x for x in projs if x is not None]
        return {'name': name, 'n_para': len(projs), 'n_sent': len(F_all),
                'mean': round(float(np.mean(projs)), 3),
                'first3': round(float(np.mean(projs[:3])), 3),
                'last3': round(float(np.mean(projs[-3:])), 3),
                'drop': round(float(np.mean(projs[:3]) - np.mean(projs[-3:])), 3),
                'slope': round(float(np.polyfit(np.arange(len(projs)), projs, 1)[0]), 5),
                'series': [round(float(x), 3) for x in projs]}

    # ===== 数据 =====
    docs = []
    arch = []
    for i in range(1, 7):
        t = (BASE / f'novel-project/archives/vol-1-ch-{i}-draft.md').read_text(encoding='utf-8', errors='ignore')
        arch += [l.strip() for l in t.split('\n') if l.strip() and re.search(r'[一-龥]', l)
                 and len(l.strip()) >= 30 and not l.strip().startswith('#')]
    docs.append(('AI-archives 6章', arch))
    for name in ('doc2', 'doc3'):
        t = (OUT / f'{name}.txt').read_text(encoding='utf-8')
        paras = [l.strip() for l in t.split('\n') if l.strip() and re.search(r'[一-龥]', l) and len(l.strip()) >= 30]
        docs.append((f'AI-{name}', paras))

    # 天行健连续段（同量级）+ 前 6 章
    raw = (BASE / '天行健.txt').read_bytes().decode('gbk', errors='ignore')
    lines = [l.strip() for l in raw.split('\n') if l.strip()]
    CH_RE = re.compile(r'^第[一二三四五六七八九十百千0-9]+[章节卷回部]')
    txj_all = [l for l in lines if re.search(r'[一-龥]', l) and len(l) >= 30]
    docs.append(('人类-天行健连续段', txj_all[:len(arch) + 40]))
    # 天行健前 6 章（按章节标记切）
    ch_idx = [i for i, l in enumerate(lines) if CH_RE.match(l)]
    ch6 = ch_idx[1:7]  # 第 1-6 章起点（跳过"第 1 章"前的第一部题头？ch_idx[0] 是第 1 章标记行）
    ch6_end = ch_idx[7]
    txj_ch6 = [l for l in lines[ch_idx[0]:ch_idx[7]] if re.search(r'[一-龥]', l)
               and len(l) >= 30 and not CH_RE.match(l)]
    docs.append(('人类-天行健前6章', txj_ch6))

    results = []
    print('===== 篇核心投影（全句元 mean 核心——各段）=====')
    for name, paras in docs:
        r = analyze_doc(name, paras)
        if r:
            results.append(r)
            print(f'{name}: {r["n_para"]} 段/{r["n_sent"]} 句元——均值 {r["mean"]}——首3 {r["first3"]}/末3 {r["last3"]}——首末差 {r["drop"]:+.3f}——斜率 {r["slope"]:+.5f}')

    # ===== 章节级对比（archives 6 章 vs 天行健前 6 章）=====
    print('\n===== 章节级对比（archives 6 章 vs 天行健前 6 章——各章对篇核心投影）=====')
    ch_compare = {'archives': [], 'tianxingjian': []}
    # archives 各章
    for i in range(1, 7):
        t = (BASE / f'novel-project/archives/vol-1-ch-{i}-draft.md').read_text(encoding='utf-8', errors='ignore')
        paras = [l.strip() for l in t.split('\n') if l.strip() and re.search(r'[一-龥]', l)
                 and len(l.strip()) >= 30 and not l.strip().startswith('#')]
        ch_compare['archives'].append(paras)
    # 天行健各章（前 6 章）
    for k in range(6):
        s, e = ch_idx[k], ch_idx[k + 1]
        paras = [l for l in lines[s:e] if re.search(r'[一-龥]', l) and len(l) >= 30 and not CH_RE.match(l)]
        ch_compare['tianxingjian'].append(paras)
    # 篇核心（各自全篇）→ 每章均值
    for side, chs in list(ch_compare.items()):
        allp = [p for ch in chs for p in ch]
        F_all = all_fp(allp)
        Fn = F_all / (np.linalg.norm(F_all, axis=1, keepdims=True) + 1e-9)
        core = Fn.mean(0)
        ch_means = []
        for ch in chs:
            vals = [para_proj(p, core) for p in ch]
            vals = [x for x in vals if x is not None]
            ch_means.append(round(float(np.mean(vals)), 3) if vals else None)
        ch_compare[side + '_means'] = ch_means
        print(f'{side}: 章均值 {ch_means}——首章 {ch_means[0]} / 末章 {ch_means[-1]}——差 {ch_means[0]-ch_means[-1]:+.3f}')

    out = {'docs': results, 'chapter_compare': {
        'archives_means': ch_compare['archives_means'],
        'tianxingjian_means': ch_compare['tianxingjian_means'],
        'note': 'archives=AI 6 章 draft——天行健前 6 章=人类对照（同题材——章级对齐）'}}
    (OUT / 'global_core_anchor.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print('\n落盘 global_core_anchor.json')


if __name__ == '__main__':
    main()
