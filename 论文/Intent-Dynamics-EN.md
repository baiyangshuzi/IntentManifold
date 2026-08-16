# Intent Dynamics: From Flow-of-Thought Capture to Simulated Intelligence
### Geometric measurement and dynamical modeling of human–AI thought differences

> **Author: Baitao Wang | August 2026**
>
> **Current flagship result (v0.75)** — complete empirical chain: measurement (flow-of-thought capture) → dissection (64-dimension atlas) → intervention (inference-time spectrum) → **conception (Intent Dynamics Engine)**. Full empirical archive: *Multi-level Semantic Fingerprint System* (historical comprehensive version, in Chinese).

> **【Machine Intent Dynamics — positioning】** The multi-level semantic fingerprint system built in this study embodies the prototype of "Machine Intent Dynamics." For the first time, it transforms "root intent" from an abstract cognitive concept into a computable geometric gravity field, and reveals the dynamical features (waves, turns, coupling) that distinguish human thought-flow from AI's probabilistic gliding. Although current validation is limited to small models and the textual domain, this framework provides an empirical path beyond the autoregressive paradigm: using an explicit intent-state space to guide models from "local-probability token continuation" toward "goal-driven reasoning around a core intent." If validated at larger parameter scales and in multimodal settings, this mechanism could lay the cognitive and engineering foundation for a new generation of AI architectures with intrinsic self-drive, interpretable thought trajectories, and extremely low inference energy.

## Abstract

Autoregressive models have a single driver — local probability maximization — and no persistent state of "why write this sentence." Starting from engineering practice, this study builds a **geometric representation of intent state** from the discriminator's middle-layer 64-dimensional fingerprint, and captures systematic differences between human and AI thought-flow at clause granularity: **Humans = breathing waves** (inter-clause dimensional jump d=+2.16, turn peak/valley steepness d=+2.09, long-range memory Hurst 0.62, topic–reference coupling 0.52, low-determinism information flow), **AI = flat lines** (small jumps, shallow turns, goldfish memory Hurst 0.46, mechanical dictionary-like flow, higher transfer entropy). The 64-dimension dissection further shows that the deep signals learned by the discriminator (the unexplained-11 dimensions) are precisely the strongest human–AI difference group (median d=0.53 vs 0.32 for explained dimensions) — among them dim48 is triple-evidenced (composite traceback + dual-model agreement + causal control of reference density −0.42) as the **reference-chain/cohesion dimension**, and dim10 as the **topic-organization dimension**. The inference-time intervention spectrum (201 runs / 22 conditions) proves intent state is intervenable (peak 117% oracle; recommended form vt_gate_beam 103% with 67% cost reduction — virtual-token channel + Kalman gating — minimal-intervention principle). **From these phenomena, this paper derives the three-layer Intent Dynamics Engine** (Root Intent Field / Intent Trajectory Planner / Intent Steering Executor) — five empirical laws — moving from "predicting the next word" to "free thinking around a root intent": AI's first possession of an internal "why."

## 1. Introduction: From "Resembling Humans" to "How Thought Flows"

### 1.1 Origin

This research began with an engineering pain point in a narrative-writing system: AI-generated text remained machine-identifiable even after prompt engineering and local lexical substitution. The first-generation research (v0.60–v0.66) answered: **the difference lies in the organization of local clauses** — humans organize language around an intent core (clause projection 0.96 vs AI 0.79–0.84), AI is local-optimum stitching. This is static "resemblance."

The v0.74 flow-of-thought capture advanced the question: **the deepest human–AI difference is not "whether a moment resembles" but "how thought flows"** — human thought trajectories are waves fluctuating freely around a core (with undulation, memory, loops); AI trajectories are flat lines (no undulation, goldfish memory, mechanical flow). This paper's core contribution turns "intent" from philosophical metaphor into a **measurable dynamical system**, and derives an architecture from the measurements.

### 1.2 Structure

§2 Capture tools → §3 **Flow-of-thought capture (core)** → §4 Intent-state dissection → §5 Intent intervenability → §6 **Intent Dynamics Engine** → §7 Boundaries & decoupling → §8 Conclusion.

## 2. Capture Tools: Geometric Representation of Intent State

### 2.1 Discriminator and Fingerprint

ParaDiscNN (bge-small-zh-v1.5 encoding → 512→256→64→1 MLP, trained to discriminate AI/human paragraphs). The middle-layer 64-dim activation (net[0:3]+relu+net[3]+relu — skipping LN(64)) is the **style fingerprint**:

```
f(x) = relu(W₂·relu(LN₁(W₁x+b₁)) + b₂)    # f ∈ R^64 — W₁=net[0].weight (256×512) — W₂=net[3].weight (64×256)
```

Geometric property (v0.66): discriminative power comes from **projection onto the core** (in-segment mean fingerprint direction — human 0.972 vs AI 0.807), not pairwise cosine (AUC 0.55 meaningless) — "the fingerprint points at the shaping core."

### 2.2 Trajectory Metric System (v0.74 — the capture toolbox)

Slice text at a granularity (segment/clause/word) → extract per-slice dimensional activations → connect into a trajectory — **this line is the intent-flow trajectory**.

| Metric | Definition | Thought property captured |
|---|---|---|
| Jump | mean \|Δactivation\| between neighbors | **Undulation rhythm** (wave-like progression?) |
| Turn type | local peak/valley steepness/depth | **Topic opening/closing sharpness** |
| Hurst | R/S analysis (long-range autocorrelation) | **Memory** (does earlier text influence later? looping ability) |
| Transfer entropy | symbolic conditional entropy (X past→Y future) | **Information-flow freedom** (mechanical vs free) |
| Coupling | sliding-window dimX×dimY correlation | **Inter-dimension tension** (topic–reference bonded?) |
| Dimension activation | per-dim value/variance/sign | Coordinates of intent state |

**Why clause-level is the best observation granularity** (v0.74-4): word-level trajectories have the weakest discriminative power (dim10 jump human 0.309 vs AI 0.295 — word fingerprints are highly continuous); segment-level loses undulation (dim10 jump d=+1.24); **clause-level (509–1029 points per human text) carries full undulation with the strongest discrimination (d=+2.16)**.

## 3. Flow-of-Thought Capture: Human vs AI Thought-Dynamics Portrait (Core)

### 3.1 Data and Procedure

- **Corpus**: bilingual zh 30 texts (10 human, 96–160 segments / 509–1029 clauses each — Dragon Clan/Zhu Xian/Tianxingjian continuous segments; 20 AI, 23–98 segments — DS/Qwen continuous generation) + same-theme ultra-long (Tianxingjian ch.1–6, 899 segments human vs archives, 146 segments AI imitation)
- **Fingerprint matrix**: 26,734 clauses × 64-dim fingerprints archived (fp_matrix.npz — **three regression validations ALL PASS**: document-level sent_proj 30/30 Δ≤0.0005; 6 dims 3618/3618 Δ≤0.0001; disc 603/603 — fingerprints verified authentic)
- **Analysis**: per-text clause-level dim10 (topic)/dim48 (reference) activations → trajectory → five metrics (jump/turn/Hurst/TE/coupling) → human vs AI group tests (Mann-Whitney U + Cohen's d)

### 3.2 Finding 1: Humans are Waves, AI is a Flat Line (Jump)

**Data** (clause-level):

| Metric | Human (median) | AI (median) | d | p |
|---|---|---|---|---|
| dim10 jump | 0.310 | 0.257 | **+2.16** | <0.001 |
| dim48 jump | 0.368 | 0.339 | **+1.07** | 0.009 |
| (segment-level dim10) | 0.155 | 0.115 | +1.24 | 0.010 |

**Derivation**: human thought trajectories **fluctuate freely** around the core — dim10 (topic) rises when opening a new topic, holds during development, falls during transitions — forming waves. AI trajectories are significantly flatter (d=+2.16) — **AI is not "randomly jagged" but "wave-less"** — lacking topic opening/closing structure. **For humans, undulation is the breathing of thought; for AI, flatness is the standstill of thought.**

![Intent flow trajectory (human waves vs AI flat)](意图流转轨迹图.png)

![Clause-level intent trajectory (human steep vs AI flat)](句元级意图轨迹图.png)

### 3.3 Finding 2: Deeper Human Turns, Shallow AI Turns (Turn Type)

**Data** (turn "type" not "count"):

| Metric | Human | AI | d | p |
|---|---|---|---|---|
| dim10 peak steepness | 0.468 | 0.381 | **+2.09** | <0.001 |
| dim10 valley depth | 0.468 | 0.391 | **+2.14** | <0.001 |
| dim48 peak steepness | 0.556 | 0.520 | **+0.90** | 0.033 |
| dim48 valley depth | 0.550 | 0.523 | **+0.93** | 0.026 |

(Control: turn **count** / break **density** show no difference — 0.175 vs 0.170 — "count" metrics are ineffective; "type" metrics are effective.)

**Derivation**: turn **steepness/depth** (not count) separates humans and AI — human topic opening/closing is distinct (sharp peaks/valleys — the "cadence" of thought), AI turns are shallow (smooth gliding). **Counts equal, amplitudes differ sharply** — both have similar numbers of turns, but human turns are "meaningful pivots," AI turns are "directionless slides."

### 3.4 Finding 3: Human Long-Range Memory, AI Goldfish Memory (Hurst)

**Data** (same-theme ultra-long trajectories):

| Trajectory | Segments | dim10 Hurst | dim48 Hurst |
|---|---|---|---|
| Tianxingjian ch.1–6 (human) | 899 | **0.621** (>0.5 — long-range memory) | 0.638 |
| archives (AI imitation) | 146 | **0.457** (<0.5 — anti-persistent / near random walk) | 0.592 |

**Derivation**: human writing has **memory** — earlier content influences later organization across distance (loops, echoes, foreshadowing — Hurst>0.5) — the ability for "the original intent to periodically resurface." AI's dim10 Hurst below 0.5 (anti-persistent) — **inter-segment topic-memory breaks — goldfish memory** — AI cannot "remember its original intent." This is the dynamical explanation of "AI long-text digression" (not semantic forgetting — organizational-memory fracture; consistent with long-range semantic retention 0.94+ showing no difference: **semantics remembered, organization forgotten**).

### 3.5 Finding 4: AI's Information Flow is Mechanical, Human's is Free (Transfer Entropy)

**Data** (clause-level):

| Direction | Human | AI | d | p |
|---|---|---|---|---|
| TE(dim10→dim48) | 0.0124 | **0.0194** | -0.79 | 0.075 (marginal) |
| TE(dim48→dim10) | 0.0123 | 0.0160 | -0.55 | 0.244 |

**Derivation**: transfer entropy measures the predictability gain of Y's future given X's past — **AI's higher topic→reference flow = more mechanically determined "topic-word→reference-pronoun" transitions** (dictionary-like association: a topic word is predictably followed by a reference — predictable) — **human's lower flow = more complex, less predictable topic–reference relations (free organization)** — an information-theoretic quantification of "human freedom vs AI mechanism."

![Transfer entropy (AI mechanical flow higher)](转移熵对比图.png)

### 3.6 Finding 5: Human Topic–Reference "Bonded," AI "Disconnected" (Coupling)

**Data** (3-segment sliding window dim10×dim48 correlation): human coupling mean **+0.517** vs AI **+0.432** (p=0.118 — directional).

**Derivation**: in human writing, reference chains naturally follow topics (mention topic → "this/it/that" follows — co-fluctuation) — **high coupling**. AI's lower coupling — a segment may repeat topic words (dim10 surging) while using no references (dim48 low) — **the mathematical explanation of "circular repetition"**: dictionary repetition replaces logical reference — the two threads of thought (topic and cohesion) are "disconnected" in AI.

### 3.7 Synthesis: Human vs AI Thought-Dynamics Portrait

| Dimension | Human | AI |
|---|---|---|
| Undulation | **Waves** (jump d=+2.16) | **Flat** (no waves) |
| Turns | **Deep** (steepness d=+2.09) | **Shallow** |
| Memory | **Long-range** (Hurst 0.62) | **Goldfish** (Hurst 0.46) |
| Information flow | **Free** (low TE) | **Mechanical** (high TE) |
| Coupling | **Bonded** (+0.52) | **Disconnected** (+0.43) |

**In one sentence**: human writing is **free fluctuation within a root-intent gravity field** (memorable loops, deep turns, free flow); AI writing is **field-less flat gliding** (no undulation, no memory, mechanical flow) — **"thought with a gravity field" vs "thought without a field"** — the observational basis of the Intent Dynamics Engine.

![Thought manifold (human concentrated vs AI dispersed)](思想流形图.png)

## 4. Intent-State Dissection: 64-Dimension Atlas (v0.74)

### 4.1 Dimension Function Localization (Causal Intervention — 144 runs)

Dimension perturbation (replace μ_h / scale 1.5 / 0.5 / noise control) → virtual-token channel injection → language-feature changes in generated text:

| Dimension | Perturbation effect | Function |
|---|---|---|
| **dim48** | **reference density −0.42** / conjunction −0.20 | **reference-chain/cohesion dimension** |
| **dim10** | topic density +0.12~+0.23 / reference density +0.13~+0.15 | **topic-organization dimension** |
| dim46 | sent_proj scale0.5 significant (+0.021 p=0.04) | projection-adjustment candidate |
| dim34/22/43 | weak directional | discriminative signal |

(Note: sent_proj-level effects are weak — 2/18 significant — attenuated by the MLP 64→896 mapping — **dimension control must be measured at the language-feature level**)

### 4.2 Interpretability × Discrimination Inverse Relation (64-Dimension Panorama)

| Group | Median \|human-AI d\| | Significant |
|---|---|---|
| Explained (42 — surface features) | 0.32 | 23 |
| Other unexplained (11) | 0.35 | 7 |
| **Unexplained-11** | **0.53** | **9/11** |

**Derivation**: **the deep signals the discriminator learned are precisely the dimensions not linearly explainable by surface features** — unexplained-11 = discriminative-signal proper (top-10 strongest includes 5 unexplained: dim58 +0.69/dim55 +0.67/dim59 +0.65/dim34/dim46) — **"unexplainable" ≠ "useless" — it is the most concentrated discriminative information** — the core coordinates of intent state lie on these deep dimensions.

![64-dim human-AI difference panorama (orange=unexplained-11)](六十四维全景图.png)

### 4.3 Polarity Grouping (basis of Law 3)

Unexplained-11 splits into two groups (consistent with internal correlation 38/55): **Human-organization group** (dim10/11/34/46/48/59 — positive human-AI d + positive jump — human-high-activation + human-waves) vs **AI-feature group** (dim22/26/43/52/5 — negative + negative — AI-high-activation + AI-fluctuation) — **human waves appear on human-high-activation dimensions; AI fluctuations on AI-high-activation dimensions** — intervention must distinguish polarity (Law 3).

## 5. Intent Intervenability: Inference-Time Spectrum and Causal Control

### 5.0 Positioning Statement: Existing Interventions Are Structural/Expressive-Layer — NOT Flow-Of-Thought Layer (Important Distinction)

**Layer relations that must be clarified** (three periods of this research):

| Layer | Period | Metrics/Mechanisms | Status |
|---|---|---|---|
| **Structural/expressive-layer intervention** | v0.68–v0.73 (earlier) | sent_proj projection (core-clinging)/topic-word retention/logits bias/beam selection/virtual-token injection — **all based on the discriminator's static projection and word-level structural metrics** | **Implemented (this section)** |
| **Measurement/monitoring layer (flow-of-thought)** | v0.74 (later) | jump/turn steepness/Hurst/transfer entropy/coupling — capturing human–AI **flow differences** (§3) | Implemented (measurement) |
| **Flow-of-thought intervention** | v0.75 conception (future) | trajectory planner uses wave patterns (§3 target distributions) as **generation targets** | **Not implemented (§6 engine)** |

**Explicit statement**: the §5 intervention spectrum (probability→logits→beam→seed→virtual token→Kalman gating) is **entirely based on structural/expressive-layer metrics** — the intervention target is "generated text clings to the core" (static projection consistency, sent_proj) — **these experiments (v0.68–v0.73) predate flow-of-thought measurement (v0.74) and did NOT use the §3 flow metrics (jump/turn/Hurst/TE/coupling) at all**. In this study, flow-of-thought metrics serve the **measurement and monitoring role** (revealing human–AI differences); using flow metrics as **intervention targets** (e.g., "generated trajectory's turn steepness approaching human levels") is the Intent Dynamics Engine's conception (§6) — **not yet implemented**. Therefore:
1. §5's "117%/103% oracle" proves the spectrum ceiling of **structural/expressive-layer intervention** (static core-clinging organization) — not the flow layer;
2. §3's flow metrics are **observational evidence** (human waves vs AI flat) — not the objective of existing interventions;
3. The **implementation path for flow-layer intervention** (wave patterns as target distributions) is in §6.4 — acceptance = end-to-end generation's five metrics approach the human portrait (only then can "flow-of-thought intervention" be claimed implemented).



### 5.1 Intervention Spectrum (201 runs / 22 conditions — oracle = Δ÷0.09×100)

Probability 17-25% → logits 40% → beam 51% → seed+beam **82%** → virtual-token channel (vt_ext 49% — **+16 points over the logits carrier**) → three-layer stacking **vt_seed_beam 117%** → **Kalman-gated vt_gate_beam 103% (67% cost reduction — 0.37 s/oracle-point — recommended form)**. No-beam tests: seed effect depends on beam (pure seed −1%) — beam's net contribution 36 points irreplaceable — gating 46% is the no-beam optimum — **"when to inject" matters more than "what to inject"** (minimal-intervention principle).

### 5.2 Kalman Repair (Prediction-Accuracy Path)

S2 r=0.15's root cause: Kalman implementation (segment start read stale x_pred) — after fixing the `predict()` semantics, r=0.389 (2.6–3×) — adaptive strategy (free segments skip beam — efficiency) — **intent state is predictable (segment-level r≈0.68 — lag structure)** — prediction-driven resource allocation = quality-cost optimum.

### 5.3 Causal-Control Loop

dim48/dim10 perturbations change reference/topic organization of generated text — **dimensions are not merely correlated but causally intervenable** — providing "knobs" for the engine's steering executor.

## 6. Intent Dynamics Engine: The Inference Chain from Phenomena to Architecture

### 6.1 Derivation of the Five Laws (phenomenon → law)

**Law 1 (Subconscious before surface control)** — Phenomenon: human clause jump d=+2.16 — human thought fluctuates **freely** around the core, not glued to it. Derivation: if intent were a per-sentence hard constraint (existing "core-clinging" interventions), fluctuation would be suppressed — making AI flatter and more mechanical (opposite of the goal) — **root intent must exist as a background field — surface intent flows freely — pull-back only when deviating beyond a safety boundary**.

**Law 2 (Flow features are more fundamental than static features)** — Phenomenon: clause-level jump-significant dimensions 56/64 vs static human-AI 39/64; turn steepness d=+2.09. Derivation: single-point projections tell whether "this clause clings"; flow rhythm/turn depth/coupling dynamics tell "how thought progresses" — **the higher-information discriminative and target signals live in flow features** — the engine's objective should be trajectory-morphology-based, not point-based.

**Law 3 (Dimension polarity must be treated separately)** — Phenomenon: 64 dims are non-uniform — human waves on human-high-activation dims (dim10/34/46 etc.), AI fluctuation on AI-high-activation dims (dim22/26/43). Derivation: intervention cannot be one-size-fits-all — **positive (human-organization) group should encourage wave-like undulation (wide fluctuation band); negative (AI-feature) group should suppress mechanical swinging (narrow band)**.

**Law 4 (Interpretability and discriminative power are inversely related)** — Phenomenon: explained d=0.32 vs unexplained-11 d=0.53. Derivation: the discriminator's deep signals are the unexplainable dimensions — **the core coordinates of intent state lie on deep dimensions — the engine's state representation should use all 64 dims (or the 11-dim core subspace), not only explainable ones** — explainable dims provide the "translation layer," deep dims the "discriminative power."

**Law 5 (Information-flow freedom is a new axis)** — Phenomenon: TE AI 0.019 vs human 0.012 — AI's topic→reference transitions are mechanically determined. Derivation: **human freedom vs AI mechanism is an axis independent of "core-clinging"** — the engine must not only "cling to the core" (projection) but also "flow freely" (low-determinism flow) — the target distribution should include an information-flow-freedom constraint.

### 6.2 Three-Layer Architecture

**Layer 1: Root Intent Field** — root intent = fixed vector R in the 64-dim (or 11-dim core subspace) space — the gravity-field minimum — close = comfortable, far = high potential — **correction triggered only when potential exceeds a threshold** (Law 1) — verified components: document-core/external-core anchors (document-core d 2.04–2.09; vt_ext 49% / vt_oracle 55% — the field's anchor is verified).

**Layer 2: Intent Trajectory Planner** — learns human fluctuation patterns (§3.2–3.6 target distributions: allowed fluctuation range / target jump / target coupling / target Hurst) — input: current intent state + history → output: next-clause target-state distribution — **wide band for positive dims, narrow for negative dims** (Law 3) — verified components: Kalman gating (predict-assign — r=0.389 — intent state predictable). **Fluctuation-pattern learning is COMPLETE (v0.77-1 — `stage3/engine_planner_bands.py` — fully offline)**:

| Target distribution (per-document — human p25–p75 band) | Value |
|---|---|
| Jump dim10 | 0.296–0.324 (AI 0.257 — strong constraint) |
| Turn steepness/depth dim10 | 0.445–0.486 / 0.446–0.494 (AI 0.388) |
| Jump dim48 / turn steepness dim48 | 0.353–0.382 / 0.523–0.572 |
| Hurst dim10 (long-range memory) | 0.594–0.696 (AI 0.568) |
| Coupling 10×48 | 0.468–0.558 (AI 0.432) |
| TE 10→48 (free flow — lower-bound constraint) | ≤0.019 (AI 0.023 mechanical) |

**Wave-band polarity verification (11/11 confirmed)**: human-organization group (dim10/11/34/46/48/59) segment-trajectory width ratios (human std / AI std) all **>1** — [1.41, 1.11, 1.23, 1.20, 1.05, 1.15] — median **1.17**; AI-feature group (dim22/26/43/52/5) all **<1** — [0.73, 0.60, 0.61, 0.55, 0.74] — median **0.61** (humans stay in a stable narrow band on these dims — swinging is AI's mechanical behavior). Overall 64 dims: 31/64 humans wider (directional, n.s.) — but the 11 target dims are 100% polarity-consistent — **wave-band polarity is a real dimension property**. → Planner target: positive dims widen to ≈1.17× AI level; negative dims tighten to ≈0.61× AI level.

**Layer 3: Intent Steering Executor** — target state → real-time LLM sampling guidance — options: magnetic guidance (logits bias) / intent mask (token subset) / virtual-token injection — **gating: enabled only on coupling drop / abnormal jump / trajectory deviation** (Law 1 — "when to inject" > "what to inject") — verified components: vt channel (injection 117%/103% — +16 points over logits), Kalman gating (no-beam 46%).

### 6.3 Essential Difference from the Existing Paradigm

| Dimension | Traditional autoregressive | Intent Dynamics Engine |
|---|---|---|
| Driver | Local probability | **Root-intent field + local probability** |
| State | No persistent intent state | **Measurable intent-state vector (64-dim fingerprint)** |
| Trajectory | Flat or random drift | **Human-like waves and loops (target-distribution constrained)** |
| Intervention | None | **Field-boundary trigger + trajectory planning + gated execution** |
| Interpretability | Black box | **Dimension atlas + polarity grouping** |
| Cost | Full-vocabulary compute | Optional intent mask (output-layer load reduction) |

### 6.4 Implementation Path (from verified components to end-to-end)

1. **Fluctuation-pattern learning** (planner core — ✅ COMPLETE v0.77-1): five-metric target distributions (jump/turn/Hurst/TE/coupling — human p25–p75 bands) + 64-dim wave-band polarity verification (positive 1.17× wide / negative 0.61× narrow — 11/11) — output `planner_targets.json` — see §6.2 target table
2. **Field end-to-end**: root intent (document/external core) → deviation detection (trajectory–field distance, clause-level) → above-threshold triggers vt-channel pull-back — "free fluctuation + field pull-back" generation loop
3. **Polarity steering**: positive dims encourage waves (wide target band); negative dims suppress swinging (narrow band) — implemented via the dim_perturb channel
4. **Training-time**: dim48/dim10 as loss terms (φ jointly optimized with LM — eliminating OOD injection)
5. **Verification**: end-to-end generation → five flow metrics (jump/turn/Hurst/TE/coupling) should approach the human portrait — the engine's acceptance criterion

### 6.6.10 Math-Operation Collapse Vectors across Abstraction Levels (v0.88 — longitudinal collapse — shared content pool — placeholder-Δg criterion — REJECTED)

**Question (user — directional correction)**: v0.85 compared four operations at a single language level and only measured operator-word surface differences. Real mathematization is *longitudinal*: for each operation, the collapse direction C = n(F_L4) − n(F_L1) from L1 (language entities) → L2 (action language) → L3 (symbols) → L4 (algebra) might be the geometric signature of the operation. **Telescoping identity**: C = sum of level-wise displacements — middle levels contribute only to diagnostics (pre-registered declaration).

**Design** (criteria pre-registered — 5 review points absorbed): 4 ops × 12 ladders × 4 levels; **shared content pool** (the same (A,B,object,variable-group) across all four ops — A,B value distributions pointwise identical; C values appear only in L2/L3 and telescope-cancel — the v0.85 numeric-range confound is structurally eliminated); placeholder arm = same texts with only operator slots swapped (P0 pre-check switched to random pinyin — sil 0.224→0.066); **both arms residualized against the same real-arm common direction Ĉ** (Δg same-baseline — the shared "narrative→algebra" jump, κ=0.534, removed); fresh seed 20260816.

**Results (honest)**:

| Metric | Real arm | Placeholder arm (same Ĉ) | Criterion |
|---|---|---|---|
| residual-space within | 0.092 (≈ random baseline 0.10) | 0.279 | M-C1 ✗ (needs ≥0.5) |
| g | 0.147 | 0.221 | M-C2 ✗ (just under 0.15) |
| Δg | — | −0.074 (CI [−0.124, −0.022] — significantly negative) | M-C2 ✗ |
| label permutation p | 0.0000 | — | separation exists but template-carried |
| cross-op shuffle p | 0.952 | — | M-C1 ✗ |
| level-wise Δg | d12 +0.015 / **d23 (Chinese word→symbol) +0.105** / d34 −0.036 | — | d23 marginal positive |
| M-C3 alignment | division·D_shared 0.296 (CI [0.227,0.342]); all flagged dims in HUMAN_ORG (dim11/34) | — | report-only |
| 512-d support arm | g 0.201 | Δg −0.104 (perm p=0.000) | report-only (same pattern) |

**Conclusion (REJECTED — converged wording)**: The collapse direction is **not** confirmed as operation-specific: ① residual-space within = 0.092 ≈ random baseline — after removing the common jump, op-specific residual directions are unstable across ladders (M-C1 fails badly); ② **Δg significantly negative** — the placeholder arm separates *more* — real operator words/symbols add no increment and destabilize the directions; ③ op separation exists (label permutation p=0.000) but is fully explained by the L1 entity-layer templates (shared by both arms, op-specific by construction) — **the "abstraction process adds operation-specific geometry" hypothesis is not supported in this space**; ④ narrow positives (do not change the verdict, report-only): d23 symbol-transition Δg=+0.105 (marginal), division direction aligned with D_shared, all flagged dims in HUMAN_ORG; ⑤ the structure (or its absence) lives in bge semantics — the 512-d arm shows the same pattern, the discriminator does not transform it. Methodological assets: shared content pool / same-Ĉ residualization / cross-op shuffle null / level-wise Δg / unbiased bootstrap.

### 6.6.11 Correctness Discriminability of Arithmetic Claims in Text-Surface Representations (v0.89 — semantic-layer sideline — parity-matched injection — per-pair LOO — REJECTED — architectural separation supported)

**Question (user — architectural separation)**: language (BGE) and mathematical-logic encoders are claimed to be structurally different classes (inductive biases: function composition / operator action / formal transformation vs lexical co-occurrence / topic / reference) and should be separated. Step ① of the user's four-step route: concatenate BGE semantic features + lightweight math-structure features → train a "math correctness discriminator." This experiment measures the **upper bound of correctness discriminability in text-surface representations** (bge semantics / style fingerprints / hand-written structure) — two-way pre-registration (FAIL expected → supports separation; PASS → route ② CodeBERT).

**Design** (criteria pre-registered — 4 review points frozen): 96 pairs (3 groups × 32: transcription 4 ops × 8 / word problems 4 ops × 8 / assertion-form equations 2 types × 16 — derivation clauses removed to kill the "internal consistency ⇒ correctness" isomorphism cue; vertical-arithmetic group excluded: stated-unit consistency cue ≈90%); **five-constraint error injection**: e≠c, |e−c|∈[1,max(2,c//10)], **same digit count**, **same parity** (without it, operand-determined result parity yields a 75% cue — pure surface rule, zero arithmetic — the most severe false-PASS hole), **same size-bin** (cross-bin leakage); direction balance 16/16; per-pair single-token-difference hard assertion (a real trap was caught: the equation template's R=x+b textual binding made two digits differ — rebuilt with the true value fixed so wrong texts need real arithmetic to detect); **per-pair LOO** (pair identity never leaks); LR primary estimator (nested C selection for the observed acc — fixed C=1.0 for the permutation null — parameter-sensitivity double check Δ=0.0625 honestly reported); 1000 label permutations p=(1+#{≥obs})/1001; M-M3 correct-value-pool reassignment as a conditional gate (only fires on PASS).

**Results (honest)**:

| Metric | Value | Criterion |
|---|---|---|
| M-M1 (F_all per-pair LOO) | acc=0.4167 (< 0.60 gate); permutation p=0.6913; CI [0.323, 0.521] | **FAIL** |
| M-M2 feature decomposition | bge 0.401 / fp 0.510 / struct 0.500 / all 0.417 — Δf = −0.094 | all ≈ chance (p ≥ 0.64) |
| M-M3 value-pool control | skipped (M-M1 FAIL — pre-registered conditional gate not fired) | gate |
| M-M4 within-pair direction κ | bge 0.046 / fp 0.060 (random baseline 0.016) | report-only |
| MLP / KNN-1 robustness | MLP 0.29–0.51 (all ≈ chance); KNN 0.16 (below chance — within-pair similarity structural) | report-only |
| F_fp degeneracy check | std<0.05 fraction 0.250 (non-degenerate) | report-only |

**Conclusion (REJECTED — converged wording)**: Under per-pair splitting, bge semantic embeddings, style fingerprints, and hand-written surface structure show **no readable correctness signal** for arithmetic claims (acc ≤ 0.60, permutation non-significant — power range θ ≤ 0.65, not "absolutely no signal") — text-surface representations do not carry the result–operand relation — correctness judgment requires **dedicated structural representations or symbolic computation** — **the architectural-separation claim receives reverse support**: together with v0.88's significantly negative Δg, the language space carries neither operation geometry (v0.88) nor operation correctness (v0.89). The user's route ② (CodeBERT second encoder) and route ③ (dedicated correctness discriminator over symbol-structure data) are empirically motivated as the next steps.

## 7. Boundaries and Decoupling (Engine Positioning)

The engine is confined to the **organizational layer** (focus/undulation/cohesion/memory — flow metrics). Three decoupling findings:
- **Reasoning zero transfer** (v0.70 + v0.73-5): intent anchoring / strongest intervention forms give zero reasoning-accuracy gain (none 39% vs vt 31% — organizational optimization does not convert to semantic correctness — "organized ≠ correct")
- **Long-range semantics zero difference** (v0.71): 0.94+ both high — "AI digression" is not semantic forgetting but organizational-memory fracture (Hurst 0.46)
- **Semantic layer needs training time**: the engine's organizational guidance does not touch semantics — semantic ability (reasoning/facts) still depends on model capacity and training

**Boundary statement**: the engine does not claim to "improve reasoning" — it claims to "make generation trajectories exhibit human-like organizational dynamics" (waves/memory/coupling/free flow) — semantic correctness is the model's responsibility; intent dynamics is the organization's responsibility.

## 8. Conclusion

This paper advances from "static resemblance" to "dynamic thought-flow":

1. **Intent flow is capturable** (§3): all five clause-level metrics capture systematic human–AI differences — humans = breathing waves (free fluctuation/deep turns/long memory/free flow/bonded coupling), AI = flat lines (no waves/shallow/goldfish/mechanical/disconnected);
2. **Intent state is dissectable** (§4): 64-dim atlas — unexplained-11 = deep discriminative-signal proper (interpretability × discrimination inverse) — dim48 = reference chain (triple-evidenced) — dim10 = topic organization — polarity grouping;
3. **Intent is intervenable** (§5): spectrum to 117%/103% — minimal-intervention principle — causal control (dim48/dim10 knobs);
4. **Intent is conceivable** (§6): five laws derived from phenomena — three-layer architecture — explicit implementation path (fluctuation learning → field end-to-end → polarity steering → training-time regularization → five-metric acceptance).

**Final proposition**: the essential human–AI writing difference lies not in vocabulary but in the **organization of intent state** — humans fluctuate freely in a root-intent gravity field (memorable loops, deep turns, free flow); AI glides flat without a field. The Intent Dynamics Engine aims to move AI from "passive prediction" to "**proactive thinking around its original intent**" — AI's first possession of an internal "why."

## Appendix: Reproduction Guide (per-conclusion paths)

**Environment**: Python 3.13 + torch/transformers/sentence-transformers/scipy/jieba/sklearn — offline analysis pure CPU (minutes) — generation needs GPU (MX570 2GB verified) — local models Qwen2.5-0.5B / bge-small-zh (HF_HUB_OFFLINE=1 cache). Repository: IntentDynamics (GitHub-ready — scripts/data/models/papers bundled).

### Bundled assets (no regeneration needed)

| Asset | Path | Validation |
|---|---|---|
| Clause fingerprint matrix | data/dim_analysis/fp_matrix.npz (26,734×64 + h1 256) | three regression validations PASS |
| Clause metadata | data/dim_analysis/rows.json | row-aligned |
| Discriminator weights | data/para_discriminator_v2.pt | fingerprint pipeline |
| Mapping MLP | data/intent_prior_model/mlp_checkpoint.pt | vt injection |
| Intervention spectrum | data/training_intervention/manifest.json + texts/ (201 runs full texts) | reproduces paper numbers |
| Probe corpus | data/independent_test/ (dim_probe/language_mechanism/manifest) | 11-dim list reproduction |

### Conclusion→command mapping

| Paper conclusion | Script | Command | Output |
|---|---|---|---|
| §3.2 waves vs flat (d=+2.16) | dim_flow_sent.py | `python stage3/dim_flow_sent.py` | flow_sent_analysis.json (dim10_jump) |
| §3.3 turn steepness (d=+2.09) | dim_flow_sent.py | same | turn10_peak_steep_mean |
| §3.4 Hurst | dim_flow.py | `python stage3/dim_flow.py` (corpus-optional) | flow_analysis.json |
| §3.5 transfer entropy | dim_flow_sent.py | same | te_10_to_48 |
| §3.6 coupling | dim_flow.py | same | coupling |
| §4.1 dim48/dim10 causal | dim_intervene.py (GPU) | `python stage3/dim_intervene.py` | intervene_manifest.json |
| §4.2 inverse relation | dim_64_full.py | `python stage3/dim_64_full.py` | dim64_full.json |
| §5.1 spectrum | gen_theme_guidance.py (GPU) | `--conditions vt_ext,vt_seed_beam,vt_gate_beam --seeds 0,1` | manifest append |
| 11-dim list | dim_activity.py | `python stage3/dim_activity.py` | activity.json |

### Full pipeline

```bash
python stage3/dim_activity.py          # activity
python stage3/dim_probe_deep.py        # probe + traceback
python stage3/dim_cross_validate.py    # dual-model
python stage3/dim_64_full.py           # 64-dim panorama
python stage3/dim_flow.py              # segment + Hurst
python stage3/dim_flow_sent.py         # clause + TE
python stage3/dim_flow_word.py         # word + 11-dim panorama
python stage3/gen_theme_guidance.py --conditions vt_ext --seeds 0,1   # GPU
python stage3/md_to_docx.py "论文/意图动力学：从意图流转捕捉到模拟智能的架构构想.md"
```

### Reproduction verification (2026-08-14 — all PASSED)

dim_activity (11-dim list ✓) → dim_64_full (dim59 d=+0.65 ✓) → dim_flow_sent (d=+2.16 / TE 0.0124 vs 0.0194 ✓) → dim_probe_deep (dim48: 21 sig-pairs/8-8 ✓) → gen_theme_guidance smoke (vt_ext 0.9602 ✓) → md_to_docx (✓) — **all numbers match the paper**.
