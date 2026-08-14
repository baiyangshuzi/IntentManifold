# IntentDynamics (Machine Intent Dynamics)

> A multi-level semantic fingerprint system with the prototype of **"Machine Intent Dynamics"** — for the first time transforming "root intent" from an abstract cognitive concept into a **computable geometric gravity field**, revealing how human thought-flow differs from AI's probabilistic gliding (waves, turns, coupling), and providing an empirical path beyond the autoregressive paradigm with a three-layer engine architecture.

## I. Core Proposition

Autoregressive models have a single driver: **local probability maximization** — no persistent state of "why write this sentence". This system proposes:

> **A defining feature of intelligence is a measurable, intervenable, trainable "intent state" during text generation — it does not decide every word directly, but provides the gravitational field within which all words are chosen.**

## II. Key Findings (all empirically verified — reproducible)

### 1. Flow-of-Thought Capture (Human vs AI Thought Dynamics Portrait)

| Metric (clause-level) | Human | AI | d / p |
|---|---|---|---|
| Dimensional jump (waves vs flat) | 0.310 | 0.257 | **+2.16 / <0.001** |
| Turn steepness (topic opening/closing) | 0.468 | 0.381 | **+2.09 / <0.001** |
| Long-range memory Hurst | 0.621 | 0.457 | Tianxingjian vs archives |
| Transfer entropy (information-flow freedom) | 0.0124 | 0.0194 | -0.79 (AI mechanical) |
| Topic×reference coupling | +0.517 | +0.432 | directional |

**In one sentence**: Humans = **free fluctuation within a root-intent gravity field** (memorable loops, deep turns, free flow) — AI = **field-less flat gliding** (no waves, goldfish memory, mechanical flow).

### 2. Intent-State Dissection (64-Dimension Atlas)

- **Interpretability × discrimination inversely related**: unexplained-11 dimensions median human-AI d=0.53 (vs 0.32 for explained-42) — deep signals = discriminative signal proper
- **dim48 = reference-chain/cohesion dimension** (triple-evidenced: composite traceback + dual-model agreement + causal control of reference density -0.42)
- **dim10 = topic-organization dimension** (perturbation ↑ topic density +0.12~+0.23)
- **Polarity grouping**: Human-organization group (dim10/11/34/46/48/59) vs AI-feature group (dim22/26/43/52/5)

### 3. Structural/Expressive-Layer Intervention (Inference-Time Spectrum — 201 runs / 22 conditions — NOT the flow-of-thought layer)

**Positioning (consistent with paper §5.0)**: this spectrum is **based on structural/expressive-layer metrics** (the discriminator's static projection sent_proj — "core-clinging", topic-word retention, logits bias, beam selection, virtual-token injection) — the intervention target is **static organizational consistency** (clinging to the core) — **predating flow-of-thought measurement (v0.74) — the flow metrics (jump/turn/Hurst/TE/coupling) were NOT used at all** (those are §2 measurement/monitoring-layer observational evidence, not the objective of existing interventions). **Using flow metrics as intervention targets (e.g., "generated trajectory's turn steepness approaching human levels") is the Intent Dynamics Engine's conception (§4 — not implemented)**.

Spectrum: probability 17-25% → logits 40% → beam 51% → seed+beam **82%** → virtual-token channel → **vt_seed_beam 117%** → **vt_gate_beam 103% (67% cost reduction — recommended)** — minimal-intervention principle (seed+beam = minimal effective combination) — "when to inject" matters more than "what to inject".

### 4. Intent Dynamics Engine (Three-Layer Architecture)

**Root Intent Field** (anchor — gravity-field minimum — pull-back only on deviation) → **Intent Trajectory Planner** (learns human fluctuation patterns — target-state distributions) → **Intent Steering Executor** (virtual-token injection + gating) — from "token continuation" to "goal-driven reasoning around root intent".

## III. Repository Structure

```
IntentDynamics/
├── README.md / README_EN.md      # Documentation (ZH / EN)
├── INTERFACE.md                  # Interface spec (data/model/scripts)
├── LICENSE                       # MIT
├── 论文/ (Papers/)
│   ├── 意图动力学：从意图流转捕捉到模拟智能的架构构想.md/.docx   # Current mainline (ZH)
│   ├── Intent-Dynamics-EN.md/.docx                                # English version
│   └── 多层级语义指纹系统…md/.docx                                # Full empirical archive (ZH)
├── stage3/                       # 18 core scripts (analysis & modeling)
├── data/
│   ├── dim_analysis/             # Fingerprint matrix (26,734 clauses) + all analysis JSON + REPORT
│   ├── independent_test/         # Probe/corpus manifest (for 11-dim list reproduction)
│   ├── intent_prior_model/       # mlp_checkpoint.pt (64→896 mapping)
│   ├── training_intervention/    # Intervention manifest + all 201 run texts
│   └── para_discriminator_v2.pt  # Discriminator weights
└── figs/                         # Core figures (9 — Chinese-named)
```

## IV. Environment & Dependencies

- **Python 3.13** (tested on Windows — Linux/macOS compatible)
- torch / transformers (≥4.40 — for Qwen2.5-0.5B generation)
- sentence-transformers (bge-small-zh-v1.5 — discriminator encoder)
- scipy / numpy / matplotlib / jieba / scikit-learn
- **Local models** (HF_HUB_OFFLINE=1 cache): `Qwen/Qwen2.5-0.5B` (generation) — `BAAI/bge-small-zh-v1.5` (encoding)
- Generation experiments need NVIDIA GPU (MX570 2GB verified, fp16) — analysis is pure CPU

## V. Quick Start

```bash
# 1. Dimension analysis (offline — minutes)
python stage3/dim_activity.py          # activity (11-dim list)
python stage3/dim_probe_deep.py        # probe + upper-layer traceback
python stage3/dim_cross_validate.py    # dual-model consistency (11/11)
python stage3/dim_64_full.py           # 64-dim panorama
# 2. Flow-of-thought (offline)
python stage3/dim_flow.py              # segment-level + Hurst (skips missing corpus)
python stage3/dim_flow_sent.py         # clause-level + transfer entropy
python stage3/dim_flow_word.py         # word-level + 11-dim panorama
# 3. Generation interventions (GPU)
python stage3/gen_theme_guidance.py --conditions vt_ext --seeds 0,1
# 4. Paper
python stage3/md_to_docx.py "论文/意图动力学：从意图流转捕捉到模拟智能的架构构想.md"
```

## VI. Reproduction Verification (2026-08-14 — all PASSED)

| Check | Script | Result |
|---|---|---|
| Dimension activity | dim_activity.py | ✓ (11-dim list reproduced) |
| 64-dim panorama | dim_64_full.py | ✓ (matches paper) |
| Clause-level flow + TE | dim_flow_sent.py | ✓ (d=+2.16 / TE 0.0124 vs 0.0194 — matches paper) |
| Probe + traceback | dim_probe_deep.py | ✓ (dim48: 21 sig-pairs/8-8 — matches paper) |
| Generation pipeline | gen_theme_guidance (vt_ext smoke) | ✓ (window 0.9602 — matches paper) |
| Paper conversion | md_to_docx.py | ✓ |

**Reproducibility engineering** (v0.75-3): ①13 scripts use configurable BASE (`INTENT_DYNAMICS_BASE` env — default = script parent — works in repo & original project); ②dependency scripts/data completed (verified by actually running); ③dim_flow corpus-missing tolerance; ④fixed missing `import os` in 2 scripts.

**Environment note**: offline analysis = pure CPU (minutes); generation = GPU + local model cache; copyrighted corpus (books) not bundled — the fingerprint matrix (all results derivable) IS bundled.

## VII. Papers

- **《意图动力学：从意图流转捕捉到模拟智能的架构构想》(Intent Dynamics: From Flow-of-Thought Capture to Simulated Intelligence)** — current mainline: measurement → dissection → intervention → conception — includes the "Machine Intent Dynamics" positioning statement
- **《多层级语义指纹系统》(Multi-level Semantic Fingerprint System)** — full empirical archive (~140K chars: spectrum/discrimination/bilingual/reasoning-decoupling/64-dim atlas)

## VIII. License

MIT (code & data) — corpus copyrights reserved to original authors (not bundled) — Author: Baitao Wang (王柏涛)
