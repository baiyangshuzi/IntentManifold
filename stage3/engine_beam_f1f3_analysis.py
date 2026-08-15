# -*- coding: utf-8 -*-
"""v0.85-6 主线分析：F1/F3 融合 beam 选优 vs 纯 sent_proj（对照 beam5）

对比：sent_proj（不退化判据）+ F1/F3 生成文本重算（方向判据——F1 高/F3 低=人类侧）
"""
import sys, json
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path
import numpy as np

BASE = Path('C:/Users/bai/Desktop/小说系统')
OUT = BASE / 'data' / 'dim_analysis'
TI = BASE / 'data' / 'training_intervention'
sys.path.insert(0, str(BASE / 'stage3'))


def main():
    import os
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    from para_dimensions import load_models, fingerprint, norm_rows
    from subclause_structure import split_subclauses
    from engine_ratio_validate import ratio_of, get_D_shared
    import torch
    enc, disc = load_models('cpu')
    D = get_D_shared()
    D = D / (np.linalg.norm(D) + 1e-9)

    manifest = json.loads((TI / 'manifest.json').read_text(encoding='utf-8'))

    def run_f1f3(run):
        """从 manifest 文本重算 F1/F3（句元级——v0.84 口径）"""
        Fs = []
        for seg in run.get('segs', []):
            for s in split_subclauses(seg.get('text', '')):
                if len(s) < 3:
                    continue
                sv = enc.encode([s], normalize_embeddings=True, batch_size=1,
                                show_progress_bar=False, device='cpu')
                SV = torch.from_numpy(sv.astype(np.float32))
                with torch.no_grad():
                    Fs.append(fingerprint(SV, disc).detach().cpu().numpy()[0])
        if len(Fs) < 8:
            return None
        F = np.array(Fs)
        Df = F[1:] - F[:-1]
        alpha = Df @ D
        f1 = float(np.mean(np.abs(alpha)))
        k = int(len(Df) * 0.6)
        f3, _, _ = ratio_of(Df[k:], D)
        # sent_proj（段级——manifest dims 已有）
        sps = [seg['dims']['sent_proj'] for seg in run.get('segs', []) if seg.get('dims')]
        return {'f1': f1, 'f3': f3, 'sent_proj': float(np.mean(sps)) if sps else None,
                'n_clause': len(Fs)}

    print('===== v0.85-6 主线：beam5 vs beam5_f1f3 =====')
    res = {'beam5': [], 'beam5_f1f3': []}
    for cond in ('beam5', 'beam5_f1f3'):
        runs = [r for r in manifest if r.get('condition') == cond and r.get('status') == 'done'
                and r.get('prompt_id') and r.get('seed') in (0, 1)]
        for r in runs:
            f = run_f1f3(r)
            if f:
                f['run_id'] = r['run_id']
                res[cond].append(f)
        print(f'  {cond}: n={len(res[cond])}——sent_proj {np.mean([f["sent_proj"] for f in res[cond]]):.4f}——'
              f'F1 {np.mean([f["f1"] for f in res[cond]]):.3f}——F3 {np.mean([f["f3"] for f in res[cond]]):.4f}')

    # 配对对比（同 prompt×seed）
    print('\n配对（同 prompt×seed）:')
    pairs = []
    for r5 in res['beam5']:
        rid = r5['run_id'].replace('beam5', 'beam5_f1f3')
        r_f = next((f for f in res['beam5_f1f3'] if f['run_id'] == rid), None)
        if r_f:
            pairs.append((r5, r_f))
    print(f'  配对 n={len(pairs)}')
    for k, lbl in (('sent_proj', 'sent_proj'), ('f1', 'F1'), ('f3', 'F3')):
        d5 = [p[0][k] for p in pairs]
        df = [p[1][k] for p in pairs]
        if d5 and df:
            diff = np.mean(df) - np.mean(d5)
            print(f'  {lbl}: beam5 {np.mean(d5):.4f} → f1f3 {np.mean(df):.4f}（Δ={diff:+.4f}）')

    # 判据（预注册）
    sp_ok = np.mean([p[1]['sent_proj'] for p in pairs]) >= np.mean([p[0]['sent_proj'] for p in pairs]) - 0.01
    f1_ok = np.mean([p[1]['f1'] for p in pairs]) > np.mean([p[0]['f1'] for p in pairs])
    f3_ok = np.mean([p[1]['f3'] for p in pairs]) < np.mean([p[0]['f3'] for p in pairs])
    print(f'\n判据: sent_proj 不退化 {"PASS" if sp_ok else "FAIL"}——F1 提升 {"PASS" if f1_ok else "FAIL"}——'
          f'F3 下降（人类侧）{"PASS" if f3_ok else "FAIL"}')
    overall = '成功（F1/F3 融合 beam 选优有效）' if (sp_ok and (f1_ok or f3_ok)) else '否定'
    print(f'总判定: {overall}')

    out = {'beam5': res['beam5'], 'beam5_f1f3': res['beam5_f1f3'],
           'paired': [{'run': p[0]['run_id'], 'sp5': p[0]['sent_proj'], 'spf': p[1]['sent_proj'],
                       'f1_5': p[0]['f1'], 'f1_f': p[1]['f1'], 'f3_5': p[0]['f3'], 'f3_f': p[1]['f3']}
                      for p in pairs],
           'verdict': overall}
    (OUT / 'beam_f1f3_analysis.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print('落盘 beam_f1f3_analysis.json ✓')


if __name__ == '__main__':
    main()
