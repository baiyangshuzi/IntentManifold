# -*- coding: utf-8 -*-
"""v0.93 AutoDL 上传打包——scripts + data → autodl_upload.zip（~120MB）

【清单】
脚本：train_reg_common / intent_loss / surrogate_train / lora_train / control_sft / eval_g3 /
      prep_pairs（GPU 复核用）/ subclause_structure / gate_g0（复核用）
数据：fp_matrix.npz / rows.json / axis_analysis.json / intent_sent_bge.npz / intent_corpus.json /
      train_lm_corpus.json / prompts.json / pairs.npz（本地产出——GPU 复核）/
      surrogate.pt + surrogate_scaler.json + gate_g0.json + gate_g1.json（本地产出——复核）/
      para_discriminator_v2.pt / bilingual_test human_zh+ai_zh / v090_sources（毛选扩充）
要求：requirements.txt（AutoDL 锁版本——torch cu121 / transformers 4.46.3 / peft 0.13.2 /
      sentence-transformers 3.2.1 / numpy 1.26.4 / scipy / scikit-learn / accelerate / jieba）
"""
import sys, zipfile, os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / 'data' / 'dim_analysis'
DST = BASE / 'autodl_upload.zip'

SCRIPTS = ['train_reg_common.py', 'intent_loss.py', 'surrogate_train.py', 'lora_train.py',
           'control_sft.py', 'eval_g3.py', 'prep_pairs.py', 'subclause_structure.py', 'gate_g0.py']
DATA = [OUT / 'fp_matrix.npz', OUT / 'rows.json', OUT / 'axis_analysis.json',
        OUT / 'intent_sent_bge.npz', OUT / 'intent_corpus.json', OUT / 'train_lm_corpus.json',
        OUT / 'pairs.npz', OUT / 'surrogate.pt', OUT / 'surrogate_scaler.json',
        OUT / 'gate_g0.json', OUT / 'gate_g1.json',
        BASE / 'data' / 'para_discriminator_v2.pt',
        BASE / 'data' / 'bilingual_test' / 'prompts.json']
BILZH = sorted((BASE / 'data' / 'bilingual_test' / 'human_zh').glob('*.txt')) + \
        sorted((BASE / 'data' / 'bilingual_test' / 'ai_zh').glob('*.txt'))
V090 = sorted((BASE / 'data' / 'v090_sources').glob('*.txt'))
REQ = """torch==2.1.0
transformers==4.46.3
peft==0.13.2
sentence-transformers==3.2.1
accelerate
numpy==1.26.4
scipy
scikit-learn
jieba
"""

REQS_TXT = BASE / 'requirements_autodl.txt'
REQS_TXT.write_text(REQ, encoding='utf-8')

items = [(BASE / 'stage3' / s, f'stage3/{s}') for s in SCRIPTS]
items += [(p, f'data/dim_analysis/{p.name}') for p in DATA if p.exists()]
items += [(p, f'data/bilingual_test/human_zh/{p.name}') for p in BILZH]
items += [(p, f'data/bilingual_test/ai_zh/{p.name}') for p in BILZH]
items += [(p, f'data/v090_sources/{p.name}') for p in V090]
items += [(REQS_TXT, 'requirements_autodl.txt')]

missing = [str(p) for p, _ in items if not p.exists()]
if missing:
    print('缺失文件（跳过）:'); [print('  ', m) for m in missing]

with zipfile.ZipFile(DST, 'w', zipfile.ZIP_DEFLATED) as z:
    for p, arc in items:
        if p.exists():
            z.write(p, arc)
    z.write(BASE / 'stage3' / 'README_v093_AUTODL.md', 'README_v093_AUTODL.md') if (
        BASE / 'stage3' / 'README_v093_AUTODL.md').exists() else None
print(f'打包完成: {DST}（{os.path.getsize(DST)/1e6:.1f} MB——{len(items)} 文件）')
