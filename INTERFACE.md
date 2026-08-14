# 接口文档（IntentDynamics）

## 一、数据接口

### 1. data/dim_analysis/fp_matrix.npz

**逐句元 64 维指纹矩阵 + 256 维中间激活**（v0.74 核心资产——26,734 句元）：

| 键 | 形状 | 含义 |
|---|---|---|
| `fp` | (26734, 64) float32 | 逐句元判别器指纹（net[0:3]+relu+net[3]+relu——跳过 LN(64)） |
| `h1` | (26734, 256) float32 | 逐句元 256 维中间激活（relu(LN₁(W₁x+b₁))——上层追溯用） |

### 2. data/dim_analysis/rows.json

**句元元数据**（与 fp/h1 行对齐——JSON 数组 26,734 条）：

```json
{"source": "bilingual_zh|independent_test|generalization_strict|training_intervention|long_attention",
 "doc": "文档标识", "side": "human|ai|qwen|条件名", "prompt": "P1|P2|P3|null",
 "seed": 0, "seg": 段序号, "para": 段落序号, "clause": "句元文本"}
```

### 3. data/dim_analysis/ 分析结果（JSON schema）

| 文件 | 内容 | 关键字段 |
|---|---|---|
| activity.json | 64 维活性 | rows[d].{var, seed_ratio, status, domain_d, ...}——s_active |
| probe_results.json | 探针+追溯 | b1[j].mlp_r2 / b2[j].{top8_256dims, n_sig_pairs, verdict} / b2_rand_baseline |
| cross_validate.json | 双模型 | bilingual_zh[j].d / dual_model[j].dir_consistent |
| intervene_manifest.json | 因果干预 144 runs | [{dim, mode, val, prompt, seed, win_sent_proj, lm_*}] |
| flow_analysis.json | 段级流转 | stats.human/ai[doc].{dim10_jump, hurst, ...} |
| flow_sent_analysis.json | 句元级+TE | metrics.{dim10_jump, te_10_to_48, turn10_peak_steep_mean, ...} |
| flow_word_analysis.json | 词级+全景 | word.{人类, AI} / dim11_jumps[j].{d, p} |
| dim64_full.json | 64 维全景 | rows[d].{human_ai_d, jump_d, dual_consistent, explained, ...} |

### 4. data/training_intervention/manifest.json

**干预实验 201 runs**：`[{run_id, condition, prompt_id, seed, segs[3], time_s?}]`——每 seg：`{seg, text, n_steps, dims{sent_proj,traj,l7_adj,word_proj,word_adj,entropy}, disc, strategy?, ctrl_trace?, time_s?}`。

### 5. 度量口径（跨全部数据统一）

- **sent_proj**：段内句元指纹对段内均值核心的余弦投影均值（无量纲 0-1）
- **窗口口径**：seg2+3 均值（段 1 基线不参与）
- **oracle%**：Δsent_proj ÷ 0.09 × 100（α=0.3 偏置修正模拟上限参照）
- **跳跃度**：相邻句元 |Δ激活| 均值
- **转折陡度/深度**：局部峰/谷的幅度
- **Hurst**：RS 分析（长程自相关）
- **转移熵**：符号化 3-bin 条件熵（TE(X→Y)）
- **耦合**：滑动窗口（3 单元）内 dimX×dimY 相关

## 二、模型接口

### 1. 判别器 ParaDiscNN（data/models/para_discriminator_v2.pt）

```python
from para_dimensions import load_models, fingerprint, norm_rows
enc, disc = load_models('cpu')          # enc=bge-small-zh——disc=ParaDiscNN
sv = enc.encode(句元列表, normalize_embeddings=True)
F = fingerprint(torch.from_numpy(sv), disc)   # (n,64)——指纹
Fn = norm_rows(F)                             # 行归一化
# 段内均值核心：core = Fn.mean(0) → normalize——投影 = Fn @ core
```

- ParaDiscNN：`net = Sequential(Linear(512,256), LayerNorm(256), ReLU, Linear(256,64), LayerNorm(64), ReLU, Linear(64,1))`
- **指纹路径**：net[0]→net[1]→relu→net[3]→relu（**跳过 net[4] LayerNorm(64)**）
- W₁=net[0].weight (256×512)——W₂=net[3].weight (64×256)

### 2. 映射层 MLP（64→896——train_intent_mlp.py）

```python
from train_intent_mlp import MLP
mlp = MLP(64, 256, 896)   # Qwen2.5-0.5B hidden_size
mlp.load_state_dict(torch.load('data/dim_analysis/mlp_checkpoint.pt'))
# 指纹 64 维 → 嵌入空间（注入虚拟 token 前需范数校准 ×0.46）
```

### 3. Kalman2D（kalman_pid.py）

```python
kf = Kalman2D(q=0.001, r=0.0025, x0=0.85)   # 状态 [intent, drift]
xp = kf.predict()   # 段开始——推进后验→预测本段先验（v0.73-2 修复——S2 r 0.147→0.389）
x_est, x_pred = kf.step(z)   # 段结束更新（predict+update 合并——兼容）
```

### 4. 生成干预（gen_theme_guidance.py）

```python
from gen_theme_guidance import load_gen_model, load_monitor, run_one, PROMPTS
enc, disc, pseg = load_monitor('cpu')
model, tok, device = load_gen_model('cuda')   # Qwen2.5-0.5B fp16
r = run_one(condition, PROMPTS[0], seed, enc, disc, pseg, model, tok, cfg, device, vocab_cache)
```

- **条件**（22 个）：none/b03/b05/lg05/lg10/lg10u/lg_placebo/beam5/beam5_ctl/t2_prompt/t3_prompt/t3_human/t3_self/pid_kalman/pid_kalman_ext/p_kalman_strategy/sentence_seed_beam/vt_oracle/vt_ext/vt_kalman/vt_seed_beam/vt_seed/vt_kalman_seed/vt_kalman_gate/vt_gate_beam/seed_only
- **虚拟 token 注入**：cfg 加 `dim_perturb=(dim, mode, val)` 可做维度扰动（replace/scale/noise——因果干预）
- **采样**：sample_next（手动循环——top-k/top-p/复读防御——vtok_emb/vtok_pos 注入）

## 三、脚本接口

| 脚本 | 用法 | 输入 | 输出 |
|---|---|---|---|
| dim_build_matrix.py | `python dim_build_matrix.py` | 语料路径（脚本内配置） | fp_matrix.npz + rows.json + 三项验证 |
| dim_activity.py | 同上 | fp_matrix.npz | activity.json（维度活性） |
| dim_probe_deep.py | 同上 | fp_matrix + 判别器权重 | probe_results.json（探针+追溯） |
| dim_intervene.py | 同上 | fp_matrix + Qwen 模型 | intervene_manifest.json（144 runs） |
| dim_cross_validate.py | 同上 | fp_matrix | cross_validate.json |
| dim_flow.py | 同上 | fp_matrix + 本地素材（archives/天行健） | flow_analysis.json + 图 |
| dim_flow_sent.py | 同上 | fp_matrix | flow_sent_analysis.json + 图 |
| dim_flow_word.py | 同上 | fp_matrix + 素材 | flow_word_analysis.json + 图 |
| dim_64_full.py | 同上 | fp_matrix | dim64_full.json + 全景图 |
| gen_theme_guidance.py | `--conditions a,b --seeds 0,1` | prompt/条件 | manifest + texts/ |
| train_intent_mlp.py | 同上 | 语料 | mlp_checkpoint.pt |
| md_to_docx.py | `python md_to_docx.py 论文.md` | md | docx（排版控制） |

## 四、复现流水线（端到端）

```
1. 指纹矩阵：dim_build_matrix（验证 PASS 后）
2. 维度分析：dim_activity → dim_probe_deep → dim_cross_validate → dim_64_full
3. 意图流转：dim_flow → dim_flow_sent → dim_flow_word
4. 因果干预：dim_intervene（GPU）
5. 干预谱系：gen_theme_guidance（GPU）
6. 论文：md → docx（md_to_docx）
```

## 五、常见问题

- **判别器设备不匹配**：para_dimensions 的 fingerprint 要求 SV 与 disc 同设备——CPU 分析用 `load_models('cpu')`
- **HF_HUB_OFFLINE**：本地模型缓存环境——首次需在线下载（bge/Qwen）
- **vt 注入 dtype**：虚拟 token 嵌入需转模型 dtype（fp16）——范数校准 0.46
- **复现确定性**：三重 seed（torch/numpy/rng）固定——相同输入相同输出（确定性是证据非噪声）
