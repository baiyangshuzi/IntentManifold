# v0.93 AutoDL 执行指南（训练期正则——LoRA Qwen2.5-0.5B）

镜像：**PyTorch 2.1.0 / Python 3.10 (ubuntu22.04) / CUDA 12.1**（机型 RTX 3080 10GB）
解锁后操作（SSH 或 JupyterLab 终端）：

## 0. 上传
把 `autodl_upload.zip` 传到实例（JupyterLab 拖拽或 `scp -P <端口> autodl_upload.zip root@<地址>:/root/autodl-tmp/`），解压：
```bash
cd /root/autodl-tmp && unzip -q autodl_upload.zip
```

## 1. 环境（锁版本——评审风险隔离）
```bash
cd /root/autodl-tmp
python -m venv venv && source venv/bin/activate
pip install torch==2.1.0 transformers==4.46.3 peft==0.13.2 sentence-transformers==3.2.1 \
  accelerate numpy==1.26.4 scipy scikit-learn jieba -i https://pypi.tuna.tsinghua.edu.cn/simple
export HF_ENDPOINT=https://hf-mirror.com
python - <<'EOF'
import os
from huggingface_hub import snapshot_download
os.environ['HF_ENDPOINT']='https://hf-mirror.com'
snapshot_download('Qwen/Qwen2.5-0.5B')
snapshot_download('BAAI/bge-small-zh-v1.5')
EOF
```
（模型从 hf-mirror 下载 ~1.1GB；若慢可改用 modelscope：`pip install modelscope -q && modelscope download --model Qwen/Qwen2.5-0.5B --local_dir ~/models/qwen25-05b` 后把脚本里 `QWEN` 常量改本地路径。）

## 2. GPU 复核（配对 + 代理 + G1——~10-30min）
```bash
python stage3/prep_pairs.py          # GPU 上重新生成 pairs.npz（确定性——与本地一致）
python stage3/surrogate_train.py     # 代理重训 + G1 复核——gate_g1.json 须与本地同判
```

## 3. 主臂训练（nohup——心跳/日志）
```bash
nohup python stage3/lora_train.py --mode main > train_main.log 2>&1 &
tail -f train_main.log              # 每循环一行 JSON + 心跳 lora_run/heartbeat.json
```
16 循环 ≈ 1.5-2h。检查点：`lora_run/adapter_main_step192/...` 每 192 步；最终 `adapter_final_main`。
停止码：3=G2 未过 / 9=NaN——检查 train_main.log 尾部。

## 4. 对照臂（同 768 步同配置——唯一差异无意图损失）
```bash
nohup python stage3/control_sft.py > train_control.log 2>&1 &
```

## 5. G3 评估（主臂/对照/基线同报）
```bash
python stage3/eval_g3.py > eval_g3.log 2>&1
cat data/dim_analysis/eval_g3.json
```

## 6. 下载回本地
```bash
# 只取小文件：adapter（几 MB）+ json 日志
scp -P <端口> -r root@<地址>:/root/autodl-tmp/data/dim_analysis/lora_run ./lora_run
scp -P <端口> root@<地址>:/root/autodl-tmp/data/dim_analysis/eval_g3.json .
scp -P <端口> root@<地址>:/root/autodl-tmp/data/dim_analysis/cycle_log_*.jsonl .
```
本地 F 阶段只读这些 JSON——无需 peft。

## 7. 停机
训练+评估完成后在控制台**关机**（按分钟计费——别让实例闲置）。
