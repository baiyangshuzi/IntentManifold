# -*- coding: utf-8 -*-
"""v0.74 阶段 C：因果干预——维度扰动（替换 μ_h/缩放/加噪对照）→ vt 注入 → 表层指标变化

6 维代表（跨域差异分层）：dim10(-0.49) dim48(-0.41) dim46(-0.38) dim22(+0.44) dim43(+0.37) dim34(-0.36)
——含 dim48（B2 唯一复合维度候选）
每维 4 条件（replace/scale1.5/scale0.5/noise 对照）× 3 prompt × 2 seed = 24 runs——6 维 = 144 runs
测量：sent_proj/traj/l7_adj + 主题词密度/指代密度/连接词密度（language_measure）
判定：扰动 vs 加噪对照配对差异（同 prompt/seed）
"""
import sys, json, os, time
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path
import numpy as np

BASE = Path('C:/Users/bai/Desktop/小说系统')
OUT = BASE / 'data' / 'dim_analysis'
sys.path.insert(0, str(BASE / 'stage3'))

DIMS = [10, 48, 46, 22, 43, 34]  # 跨域差异分层代表（含 dim48 复合维度候选）
MU_H = {}  # 独立集白话人类段的人类均值（阶段 C 启动时算）


def main():
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    import torch
    from gen_theme_guidance import load_gen_model, load_monitor, build_vocab_cache, run_one
    from language_measure import measure as lang_measure
    enc, disc, pseg = load_monitor('cpu')
    model, tok, device = load_gen_model('cuda')
    vocab_cache = build_vocab_cache(tok, vocab_size=model.config.vocab_size)

    # 人类均值 μ_h（独立集白话人类段 fp——从 fp_matrix 取）
    d = np.load(OUT / 'fp_matrix.npz')
    fp = d['fp']
    rows = json.loads((OUT / 'rows.json').read_text(encoding='utf-8'))
    hu_idx = [i for i, r in enumerate(rows) if r['source'] == 'independent_test'
              and r['side'] == 'human' and 'B-' in r['doc']]
    hu_fp = fp[hu_idx]
    for j in DIMS:
        MU_H[j] = float(np.mean(hu_fp[:, j]))
    print(f'人类均值 μ_h: {MU_H}')

    # manifest
    f_man = OUT / 'intervene_manifest.json'
    results = json.loads(f_man.read_text(encoding='utf-8')) if f_man.exists() else []
    done = {(r['dim'], r['mode'], r['val'], r['prompt'], r['seed']) for r in results}

    rng_noise = np.random.default_rng(42)
    noise_vals = {j: rng_noise.normal(0, np.std(hu_fp[:, j]), 1)[0] for j in DIMS}

    total = len(DIMS) * 4 * 3 * 2
    n_done = 0
    for dim in DIMS:
        mu = MU_H[dim]
        sig = float(np.std(hu_fp[:, dim]))
        conditions = [('replace', mu), ('scale', 1.5), ('scale', 0.5), ('noise', noise_vals[dim])]
        for mode, val in conditions:
            for prompt_id, prompt in (('P1', None), ('P2', None), ('P3', None)):
                for seed in (0, 1):
                    key = (dim, mode, round(float(val), 5), prompt_id, seed)
                    if key in done:
                        n_done += 1
                        continue
                    prompt = next(p for p in __import__('gen_theme_guidance').PROMPTS if p[0] == prompt_id)
                    cfg = {'temperature': 0.9, 'top_k': 50, 'top_p': 0.9, 'K': 5,
                           'dim_perturb': (dim, mode, val)}
                    r = run_one('vt_ext', prompt, seed, enc, disc, pseg, model, tok, cfg,
                                device, vocab_cache)
                    # 测量：窗口指标 + 语言特征
                    texts = [s['text'] for s in r['segs']]
                    win = np.mean([s['dims']['sent_proj'] for s in r['segs'][1:] if s.get('dims')])
                    lm = {k: float(np.mean([lang_measure(t, enc, disc, pseg, 'cpu')[k]
                                            for t in texts[1:] if lang_measure(t, enc, disc, pseg, 'cpu')]))
                          for k in ('topic_density', 'pron_density', 'conj_density')}
                    rec = {'dim': dim, 'mode': mode, 'val': round(float(val), 5), 'prompt': prompt_id,
                           'seed': seed, 'win_sent_proj': round(float(win), 4),
                           'seg_projs': [round(float(s['dims']['sent_proj']), 4) for s in r['segs'] if s.get('dims')],
                           'l7': [round(float(s['dims']['l7_adj']), 4) for s in r['segs'] if s.get('dims')],
                           **{f'lm_{k}': round(v, 4) for k, v in lm.items()},
                           'noise_sigma': round(sig, 5)}
                    results = [x for x in results if not (x['dim'] == dim and x['mode'] == mode
                                                          and x['val'] == round(float(val), 5)
                                                          and x['prompt'] == prompt_id and x['seed'] == seed)] + [rec]
                    f_man.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding='utf-8')
                    n_done += 1
                    print(f'[{n_done}/{total}] dim{dim}-{mode}={val:.3f}-{prompt_id}-s{seed}: win={win:.4f}')
    print(f'完成 {n_done}/{total}')


if __name__ == '__main__':
    main()
