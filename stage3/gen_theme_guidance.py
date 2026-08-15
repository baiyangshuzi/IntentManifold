# -*- coding: utf-8 -*-
"""v0.68 Phase 2：生成+监测+引导主脚本（主题词保持——判别器作监测器）

用户规格：
  生成循环中判别器实时算 sent_proj——低于阈值 → 下段触发"主题词保持"规则
  （不改 logits——概率层：候选 token 命中主题词 → 概率 ×(1+β)——再归一化采样）

矩阵：condition{none,b03,b05} × prompt{1,2,3} × seed{0,1,2} = 27 runs × 3 段
用法：python gen_theme_guidance.py --pilot（6 runs）或全量——断点续传
产出：data/training_intervention/manifest.json + texts/{cond}/{run_id}.txt + run.log
"""
import sys, json, os, re, time, argparse
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path
import numpy as np

BASE = Path(os.environ.get('INTENT_DYNAMICS_BASE', 'C:/Users/bai/Desktop/小说系统'))
OUT = BASE / 'data' / 'training_intervention'
(OUT / 'texts').mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(BASE / 'stage3'))

PROMPTS = [
    ('P1', '雨夜，刑警老周在城中追缉一名连续作案的凶手。他沿着水洼里的脚印，走进一条旧巷。'),
    ('P2', '老渡口，江水浑黄，渡船在雾中缓缓靠岸。守渡人老陈坐在棚下，等一个三天没有出现的乘客。'),
    ('P3', '星舰在深空迷航，窗外是无边星海。导航官望着陌生的星图，试图辨认来路。'),
]
# 条件字典化（v0.68-2）：mode=none 无引导/prob 概率层乘法/logits 加法偏置（placebo=随机 token）
# v0.68-3：lg10u=logits β1.0+top-p 并集救回（消除 B2 损耗）/beam5=判别器选优（路径 4——触发段 N 候选）
CONDITIONS = {'none': {'mode': None, 'beta': 0.0},
              'b03': {'mode': 'prob', 'beta': 0.3},
              'b05': {'mode': 'prob', 'beta': 0.5},
              'lg05': {'mode': 'logits', 'beta': 0.5},
              'lg10': {'mode': 'logits', 'beta': 1.0},
              'lg_placebo': {'mode': 'logits', 'beta': 0.5, 'placebo': True},
              'lg10u': {'mode': 'logits', 'beta': 1.0, 'union': True},
              'beam5': {'mode': None, 'beam': 5},
              # v0.68-4 外部篇核心（用户理论修正——自指锚→外部锚——意图核心生成前固定）
              't2_prompt': {'mode': 'logits', 'beta': 1.0, 'theme_source': 'prompt', 'beam': 0},
              't3_prompt': {'mode': 'logits', 'beta': 1.0, 'theme_source': 'prompt', 'beam': 5},
              't3_human': {'mode': 'logits', 'beta': 1.0, 'theme_source': 'human', 'beam': 5},
              # v0.68-5 三层级自指锚（对照——好段 buffer 自指 + beam5——与外部锚同框架对比）
              't3_self': {'mode': 'logits', 'beta': 1.0, 'beam': 5},
              # v0.69 闭环控制（Kalman+PID——句元级动态 β——推理期终极边界测试）
              'pid_kalman': {'mode': 'pid_kalman'},
              # v0.69-2 外部锚定闭环（观测口径修正——句元对 prompt 外部核心投影——非自指动态核心）
              'pid_kalman_ext': {'mode': 'pid_kalman_ext'},
              # v0.69-3 段落级 Kalman 前置预测 + 策略级干预（beam5/3/自由——预判价值验证）
              'p_kalman_strategy': {'mode': 'p_kalman_strategy'},
              # v0.69-3 控制：段 1 同款基线 + 段 2/3 无条件 beam5（isolate 策略效应——段 1 混杂分离）
              'beam5_ctl': {'mode': 'beam5_ctl'},
              # v0.69-4 句级种子（最小有效起点测试）：第 1 句前 12 token β=0.5——之后自由——段 2/3 beam5
              'sentence_seed_beam': {'mode': 'sentence_seed_beam'},
              # v0.73 虚拟 token 意图注入（模块 5——MLP 映射层 fingerprint→embedding 空间）
              'vt_oracle': {'mode': 'vt_oracle'},
              'vt_ext': {'mode': 'vt_ext'},
              'vt_kalman': {'mode': 'vt_kalman'},
              # v0.73-2 组合：vt_ext 注入 + 句级种子（段 1 前 12 token β=0.5）+ 段 2/3 beam5
              'vt_seed_beam': {'mode': 'vt_seed_beam'},
              # v0.73-3 无 beam 测试（去掉最贵组件——种子+卡尔曼能否保持效果）
              'seed_only': {'mode': 'seed_only'},                    # 纯种子（段 2/3 自由）
              'vt_seed': {'mode': 'vt_seed'},                        # 种子 + prompt 核心注入（无 beam）
              'vt_kalman_seed': {'mode': 'vt_kalman_seed'},          # 种子 + 在线 EMA 注入（无 beam）
              'vt_kalman_gate': {'mode': 'vt_kalman_gate'},         # 种子 + Kalman 预测门控注入强度
              'vt_gate_beam': {'mode': 'vt_gate_beam'},             # 门控→{beam 开关, 注入强度}双自由度（待办①）
              # v0.78 根意图势场端到端（引擎待办 2——自由波动+势场回拉——根意图×表面意图交互）
              'vt_field': {'mode': 'vt_field', 'alpha': 0.1},                 # 自治环：R=prompt 核心→句元级偏离门控回拉+慢 EWMA 内化
              'vt_field_persist': {'mode': 'vt_field_persist', 'alpha': 0.1},  # 段 1-2 注入 φ(T)——段 3 关闭注入（R 继续更新）——内化检验
              'vt_field_frozen': {'mode': 'vt_field_frozen', 'alpha': 0.1},    # 同 persist 但 R 全程冻结（C-B4 对照——分离内化 vs 注入残存）
              'vt_field_full': {'mode': 'vt_field_full', 'alpha': 0.1}}        # 段 1-3 全程注入 φ(T)（C-B3 上限基准）

# 人类锚文档绑定（prompt → human_zh 文档——按序）
HUMAN_BIND = {'P1': 'ZH-H01', 'P2': 'ZH-H02', 'P3': 'ZH-H03'}
THRESHOLD = 0.85
SEG_MIN, SEG_MAX = 60, 120
SEED = 42  # placebo 随机 token 固定种子（可复现）
STOP_PUNC = '。！？…"”'
STOPWORDS = {'那', '这', '的', '了', '在', '是', '有', '和', '就', '也', '都', '一个', '他', '她',
             '它', '们', '着', '过', '很', '又', '把', '被', '让', '向', '从', '到', '去', '来',
             '说', '道', '想', '看', '听', '走', '进', '出', '上', '下', '里', '外'}


def load_gen_model(device='cuda'):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B')
    try:
        model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B',
                                                     torch_dtype=torch.float16).to(device).eval()
        return model, tok, device
    except Exception as e:
        print(f'GPU 加载失败 {str(e)[:80]}——降级 CPU fp32')
        model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B').to('cpu').eval()
        return model, tok, 'cpu'


def load_monitor(device='cpu'):
    """复用 para_dimensions——监测器常驻 CPU（显存策略）"""
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    import jieba
    import jieba.posseg as pseg
    jieba.setLogLevel(60)
    from para_dimensions import load_models
    enc, disc = load_models(device)
    return enc, disc, pseg


def monitor_segment(seg_text, enc, disc, pseg, device='cpu'):
    """段级七维（判别器口径——sent_proj 段内均值核心）"""
    from para_dimensions import para_dimensions
    from subclause_structure import split_subclauses
    if len(seg_text) < 30:
        return None
    return para_dimensions(seg_text, enc, disc, split_subclauses, pseg, device)


def extract_theme_words(seg_text, pseg, tok, K=5):
    """前段高频实义词 top-K（名/动/形 len≥2 剔停用词——encode ≤4 token 过滤）"""
    import jieba.posseg as p
    words = [w for w, f in pseg.cut(seg_text)
             if len(w) >= 2 and f[0] in ('n', 'v', 'a') and w not in STOPWORDS]
    from collections import Counter
    cw = Counter(words)
    cands = [w for w, _ in cw.most_common(20)]
    out = []
    for w in cands:
        if len(tok.encode(w)) <= 4:
            out.append(w)
        if len(out) >= K:
            break
    return out


def build_theme_lookup(theme_words, tok):
    """快路径（主题词→token id 并集）——g 向量延迟初始化（build_logits_g）"""
    fast_ids = set()
    for w in theme_words:
        fast_ids.update(tok.encode(w))
    return {'words': theme_words, 'fast_ids': fast_ids, 'decode_cache': {}, 'g_t': None, 'g': None}


def build_vocab_cache(tok, vocab_size=None):
    """全词表 token 字符串缓存——程序启动一次（静态——所有段复用）——单向 w in decoded 匹配基础
    transformers 5.x batch_decode 会把整批拼成一个字符串（不返回逐条）——改用 convert_ids_to_tokens（逐条不拼接）"""
    import time
    t0 = time.time()
    V = vocab_size if vocab_size else len(tok)
    toks = tok.convert_ids_to_tokens(list(range(V)))
    cache = {i: t for i, t in enumerate(toks) if t and t.strip()}
    print(f'vocab token 缓存构建 ✓（{len(cache)}/{V} 词条——{time.time()-t0:.1f}s）')
    return cache


def build_logits_g(lookup, tok, vocab_cache, device='cuda', vocab_size=None):
    """全词表指示向量 g∈{0,1}^V（V=模型词表——tokenizer len 可能小于 model vocab）
    快路径 ∪ 单向 `w in decoded`（去 decoded in w 防子词误匹配）
    每引导段一次（~15 万次包含检查 ~1-2s）——g_t 常驻 GPU（0.6MB）"""
    import torch
    V = vocab_size if vocab_size else len(tok)
    g = np.zeros(V)
    g[list(lookup['fast_ids'])] = 1.0
    words = lookup['words']
    for i, txt in vocab_cache.items():
        if i >= V or g[i] == 1.0:
            continue
        if any(w in txt for w in words):
            g[i] = 1.0
    lookup['g'] = g
    lookup['g_t'] = torch.from_numpy(g.astype(np.float32)).to(device)
    return lookup


def boost_mask(cand_ids, lookup, tok, beta):
    """候选命中主题词 → 概率系数 (1+β)——双向包含：w in decoded ∪ decoded in w"""
    coef = np.ones(len(cand_ids))
    for i, tid in enumerate(cand_ids):
        hit = tid in lookup['fast_ids']
        if not hit:
            txt = lookup['decode_cache'].get(tid)
            if txt is None:
                txt = tok.decode([tid])
                lookup['decode_cache'][tid] = txt
            hit = any(w in txt for w in lookup['words']) or any(txt in w for w in lookup['words'])
        if hit:
            coef[i] = 1.0 + beta
    return coef


def sample_next(model, tok, ids, lookup, cfg, device, rng, vocab_cache=None, vtok_emb=None, vtok_pos=None):
    """手动采样一步——logits_raw → /T → (+β·g——logits 分支) → top-k → top-p → softmax → sample
    复读防御：最近 8 token 重复检测——连续 3 次触发 → top-1 采样一步
    v0.68-2：logits 分支（加法偏置全词表——top-k 之前——可救回主题词）——rescue/top_p_drop 计数
    v0.73：vtok_emb 非 None → inputs_embeds 路径。vtok_pos 指定插入位置：
           - 端部（pos=None）：V 恒为末位——从 V 位置预测（v0.73-1 实测：病态——主题循环）
           - 段首（pos=段开始索引）：V 在段首——从真实末位采样——模型原生控制 token 模式（实测可行）
           V 参与后续全部 token 的注意力——与 logits 层引导完全正交"""
    import torch
    with torch.no_grad():
        if vtok_emb is not None:
            emb_full = model.get_input_embeddings()(ids)
            if vtok_pos is not None and 0 < vtok_pos < emb_full.shape[1]:
                emb = torch.cat([emb_full[:, :vtok_pos], vtok_emb, emb_full[:, vtok_pos:]], dim=1)
            else:
                emb = torch.cat([emb_full, vtok_emb], dim=1)
            logits = model(inputs_embeds=emb).logits[:, -1, :]
        else:
            logits = model(ids).logits[:, -1, :]
    logits = (logits.float() / cfg['temperature'])

    rescue_count = 0
    top_p_drop_count = 0
    if cfg.get('mode') in ('logits', 'pid_kalman', 'pid_kalman_ext') and lookup is not None:
        if lookup['g_t'] is None:
            build_logits_g(lookup, tok, vocab_cache, device, vocab_size=logits.shape[-1])
        raw_top = torch.topk(logits, cfg['top_k']).indices.squeeze(0).cpu().numpy()  # 偏置前
        logits = logits + cfg['beta'] * lookup['g_t']   # ← 唯一干预点：全词表加法偏置
        g = lookup['g']
        # 偏置后 top-k → 先记录被救回情况
        tv, ti = torch.topk(logits, cfg['top_k'])
        topk_ids = ti.squeeze(0).cpu().numpy()
        rescued_ids = [x for x in topk_ids if g[x] > 0 and x not in set(raw_top)]
        rescue_count = len(rescued_ids)
        # top-p 裁剪计数（下方 top-p 后补）+ union 救回用概率
        cfg['_topk_ids'] = topk_ids
        cfg['_g'] = g
        tv_after, ti_after = tv, ti
    else:
        tv, ti = torch.topk(logits, cfg['top_k'])
    probs = torch.softmax(tv, -1).squeeze(0).cpu().numpy()
    if cfg.get('mode') == 'logits' and lookup is not None:
        cfg['_topk_probs'] = probs.copy()  # union 救回用（偏置后 softmax 值）
    order = np.argsort(probs)[::-1]
    cum = np.cumsum(probs[order])
    kept = order[:max(1, int(np.searchsorted(cum, cfg['top_p']) + 1))]
    p = probs[kept].copy()
    cand_ids = ti.squeeze(0).cpu().numpy()[kept]

    matched = 0
    if lookup is not None:
        if cfg.get('mode') in ('logits', 'pid_kalman', 'pid_kalman_ext'):
            g = lookup['g']
            matched = int((g[cand_ids] > 0).sum())
            topk_ids = cfg.pop('_topk_ids', None)
            topk_probs = cfg.pop('_topk_probs', None)
            if topk_ids is not None:
                drop = [x for x in topk_ids if g[x] > 0 and x not in set(cand_ids)]
                top_p_drop_count = len(drop)
                if cfg.get('union') and drop and topk_probs is not None:
                    # v0.68-3：top-p 后并集救回——被裁主题词强制加入候选（p 取偏置后 softmax 值）
                    cand_ids = np.concatenate([cand_ids, np.array(drop)])
                    idxs = [int(np.where(topk_ids == x)[0][0]) for x in drop]
                    p = np.concatenate([p, topk_probs[idxs]])
            cfg['rescue'] += rescue_count
            cfg['top_p_drop'] += top_p_drop_count
        else:
            coef = boost_mask(cand_ids, lookup, tok, cfg['beta'])
            p = p * coef
            matched = int((coef > 1).sum())
    p = p / p.sum()

    # 复读防御（用户补充）：最近 8 token 重复——连续 3 次触发 → top-1
    recent = ids[0, -8:].cpu().tolist() if ids.shape[1] >= 8 else []
    cfg['rep_count'] = cfg.get('rep_count', 0)
    if len(recent) == 8 and len(set(recent)) <= 3:
        cfg['rep_count'] += 1
        if cfg['rep_count'] >= 3:
            cfg['rep_count'] = 0
            cfg['top1_breaks'] += 1
            return int(cand_ids[0]), matched, True  # 强制 top-1
    else:
        cfg['rep_count'] = 0

    j = int(rng.choice(len(p), p=p))
    return int(cand_ids[j]), matched, False


def segment_done(seg_text, n_tokens, min_tok=SEG_MIN, max_tok=SEG_MAX):
    if n_tokens >= max_tok:
        return True
    return n_tokens >= min_tok and seg_text and seg_text[-1] in STOP_PUNC


def generate_segment(model, tok, ids, lookup, cfg, device, rng, seg_start, vocab_cache=None, vtok_emb=None):
    """生成一段——返回 (段文本, 步数, match 数, top1 打破数)——v0.73: vtok_emb=虚拟 token 注入（段首位置）"""
    import torch
    out_ids = ids[0].cpu().tolist()
    n = 0
    matched_total = 0
    top1s = 0
    while not segment_done(tok.decode(out_ids[seg_start:]), n):
        cur = torch.tensor([out_ids], device=device)
        nxt, matched, top1 = sample_next(model, tok, cur, lookup, cfg, device, rng, vocab_cache,
                                         vtok_emb, vtok_pos=seg_start)
        out_ids.append(nxt)
        n += 1
        matched_total += matched
        top1s += top1
        if n >= SEG_MAX:
            break
    # hard-cut：max_tok 处截到最近句号
    text = tok.decode(out_ids[seg_start:])
    if len(out_ids) - seg_start >= SEG_MAX:
        last_p = max([text.rfind(c) for c in '。！？'] + [-1])
        if last_p > 10:
            text = text[:last_p + 1]
    return text, n, matched_total, top1s


def generate_segment_pid(model, tok, ids, lookup, cfg, device, rng, seg_start, vocab_cache, enc, disc,
                         ext_core=None):
    """v0.69 句元级闭环：逐句元生成 → 监测投影 z → Kalman 估计 → PID 合成 β → 动态偏置
    ext_core：外部核心（prompt 句元指纹均值——v0.69-2 观测口径修正——非 None 时 z=句元对外部核心投影）
    返回 (段文本, 步数, match, top1, 控制轨迹)"""
    import torch
    from kalman_pid import Kalman2D, PIDLoop
    from para_dimensions import fingerprint, norm_rows
    kf = Kalman2D(q=cfg.get('kq', 0.001), r=cfg.get('kr', 0.0025))
    pid = PIDLoop(kp=cfg.get('kp', 12), ki=cfg.get('ki', 0.3), kd=cfg.get('kd', 6),
                  alpha_feed=cfg.get('ka', 0.3), target=cfg.get('ktarget', 0.90))
    clause_buf = ''
    out_ids = ids[0].cpu().tolist()
    n = 0
    matched_total = 0
    top1s = 0
    while not segment_done(tok.decode(out_ids[seg_start:]), n):
        cur = torch.tensor([out_ids], device=device)
        nxt, matched, top1 = sample_next(model, tok, cur, lookup, cfg, device, rng, vocab_cache)
        out_ids.append(nxt)
        n += 1
        matched_total += matched
        top1s += top1
        clause_buf += tok.decode([nxt])
        if clause_buf and clause_buf[-1] in '。！？':
            # 句元完成——监测 + 控制
            try:
                sv = enc.encode([clause_buf], normalize_embeddings=True, batch_size=1,
                                show_progress_bar=False, device='cpu')
                SV = torch.from_numpy(sv.astype(np.float32)).to('cpu')
                with torch.no_grad():
                    F = norm_rows(fingerprint(SV, disc)).detach().cpu().numpy()[0]
                if ext_core is not None:
                    # v0.69-2 外部锚定观测：句元对外部核心（prompt）投影——自指高估消除
                    cn = ext_core / (np.linalg.norm(ext_core) + 1e-9)
                else:
                    cn = F / (np.linalg.norm(F) + 1e-9)  # 自指（单句元对自身——退化基线）
                z = float(F @ cn)
                x_est, x_pred = kf.step(z)
                cfg['beta'] = pid.update(z, x_est, x_pred, n)
            except Exception:
                pass
            clause_buf = ''
        if n >= SEG_MAX:
            break
    text = tok.decode(out_ids[seg_start:])
    return text, n, matched_total, top1s, pid.trace


def generate_segment_vt(model, tok, ids, lookup, cfg, device, rng, seg_start, vocab_cache, vtok_fn, enc, disc, state):
    """v0.73 虚拟 token 注入段（vt_kalman——句元级刷新意图方向）
    state['running']：64 维归一化意图方向（EMA 运行均值核心——跨段延续）
    每句元完成（。！？）：计算句元指纹 → running=0.7·running+0.3·F → vtok=φ(running)
    返回 (段文本, 步数, match, top1)"""
    import torch
    from para_dimensions import fingerprint, norm_rows
    out_ids = ids[0].cpu().tolist()
    n = 0
    matched_total = 0
    top1s = 0
    clause_buf = ''
    vtok = vtok_fn(state.get('running')) if state.get('running') is not None else None
    while not segment_done(tok.decode(out_ids[seg_start:]), n):
        cur = torch.tensor([out_ids], device=device)
        nxt, matched, top1 = sample_next(model, tok, cur, lookup, cfg, device, rng, vocab_cache,
                                         vtok_emb=vtok, vtok_pos=seg_start)
        out_ids.append(nxt)
        n += 1
        matched_total += matched
        top1s += top1
        clause_buf += tok.decode([nxt])
        if clause_buf and clause_buf[-1] in '。！？':
            try:
                sv = enc.encode([clause_buf], normalize_embeddings=True, batch_size=1,
                                show_progress_bar=False, device='cpu')
                SV = torch.from_numpy(sv.astype(np.float32)).to('cpu')
                with torch.no_grad():
                    F = norm_rows(fingerprint(SV, disc)).detach().cpu().numpy()[0]
                cur_r = state.get('running')
                cur_r = F if cur_r is None else cur_r * 0.7 + F * 0.3
                state['running'] = cur_r / (np.linalg.norm(cur_r) + 1e-9)
                vtok = vtok_fn(state['running'])
            except Exception:
                pass
            clause_buf = ''
        if n >= SEG_MAX:
            break
    text = tok.decode(out_ids[seg_start:])
    return text, n, matched_total, top1s


def generate_segment_field(model, tok, ids, lookup, cfg, device, rng, seg_start, vocab_cache,
                           vtok_fn, enc, disc, state, band_stds, freeze_R=False):
    """v0.78 根意图势场段循环（引擎待办 2——自由波动+势场回拉）

    根意图 R（64 维单位向量——构造性锚）——表层意图 F(t)（句元指纹）——双向交互：
    - 根→表：句元完成→F̄=最近 3 句元均值（降噪）→ p=F̄·R → e=0.90−p 门控分级(0.05/0.02)
      偏离超阈才回拉（定律一：潜意识优先——低于阈值完全自由）
    - 回拉方向：vt_field 用维度加权定向修正 v_pull=norm(W⊙R)（W_k=1+λ·|F̄−R|/band_std_k——
      只加重偏离贡献大的维度，避免全向量注入在无关维度引入偏移）；
      persist/frozen/full 用固定目标 T（人类核心——注入目标不随 R 漂移）
    - 表→根：R←norm((1−α)R+α·F̄)（慢 EWMA——内化时间常数≈10 句元）——freeze_R 时跳过
    state: {'R','R0','T','alpha','buf'(≤3),'inject_target','trace'}——trace 每句元记录
    返回 (段文本, 步数, match, top1)"""
    import torch
    from para_dimensions import fingerprint, norm_rows
    out_ids = ids[0].cpu().tolist()
    n = 0
    matched_total = 0
    top1s = 0
    clause_buf = ''
    alpha = state.get('alpha', 0.1)
    R = state.get('R')
    if R is None:
        R = state.get('R0')
        state['R'] = R
    T = state.get('T')
    mode_vt = cfg.get('mode')
    use_target = mode_vt in ('vt_field_persist', 'vt_field_frozen', 'vt_field_full')
    inject_target = state.get('inject_target', True)
    vtok = None
    trace = []
    while not segment_done(tok.decode(out_ids[seg_start:]), n):
        cur = torch.tensor([out_ids], device=device)
        nxt, matched, top1 = sample_next(model, tok, cur, lookup, cfg, device, rng, vocab_cache,
                                         vtok_emb=vtok, vtok_pos=seg_start)
        out_ids.append(nxt)
        n += 1
        matched_total += matched
        top1s += top1
        clause_buf += tok.decode([nxt])
        if clause_buf and clause_buf[-1] in '。！？':
            try:
                sv = enc.encode([clause_buf], normalize_embeddings=True, batch_size=1,
                                show_progress_bar=False, device='cpu')
                SV = torch.from_numpy(sv.astype(np.float32)).to('cpu')
                with torch.no_grad():
                    F = norm_rows(fingerprint(SV, disc)).detach().cpu().numpy()[0]
                # 滑动窗口均值（最近 3 句元——降噪——"表层意图流向"）
                state['buf'].append(F)
                if len(state['buf']) > 3:
                    state['buf'].pop(0)
                Fbar = np.mean(state['buf'], 0)
                Fbar = Fbar / (np.linalg.norm(Fbar) + 1e-9)
                # 偏离门控（标量投影——与 Kalman 观测口径同构）
                p = float(Fbar @ R)
                e = 0.90 - p
                if e > 0.05:
                    bf = 1.0
                elif e > 0.02:
                    bf = 0.5
                else:
                    bf = 0.0
                # 回拉方向（矢量定向——标量门控+矢量注入分离）
                if use_target:
                    v_pull = T if T is not None else R
                elif bf > 0:
                    z = np.abs(Fbar - R) / (band_stds + 1e-9)
                    v_pull = (1.0 + 1.0 * z) * R
                    v_pull = v_pull / (np.linalg.norm(v_pull) + 1e-9)
                else:
                    v_pull = R
                # 表→根更新（慢 EWMA——内化）
                if not freeze_R:
                    R = (1 - alpha) * R + alpha * Fbar
                    R = R / (np.linalg.norm(R) + 1e-9)
                    state['R'] = R
                # 注入执行（bf 门控——persist/frozen 段 3 注入关闭→完全自由）
                if bf > 0 and inject_target:
                    base = T if use_target else v_pull
                    vtok = vtok_fn(base) if bf >= 1.0 else vtok_fn(base) * bf
                else:
                    vtok = None
                trace.append({'p': round(p, 4), 'e': round(e, 4), 'bf': bf,
                              'cosF_R': round(float(Fbar @ R), 4),
                              'cosR_R0': round(float(R @ state['R0']), 4),
                              'cosR_T': round(float(R @ T), 4) if T is not None else None,
                              'cosF_T': round(float(Fbar @ T), 4) if T is not None else None})
            except Exception as ex:
                print(f'  field 句元监测失败: {str(ex)[:60]}')
                pass
            clause_buf = ''
        if n >= SEG_MAX:
            break
    text = tok.decode(out_ids[seg_start:])
    state['trace'].extend(trace)
    return text, n, matched_total, top1s


def run_one(condition, prompt, seed, enc, disc, pseg, model, tok, cfg, device, vocab_cache=None):
    """单篇 3 段：段1=纯续写基线——段2/3=触发后窗口——prompt=(id, text)
    v0.68-2：good_segment_buffer（好段主题词锚——防漂移锁定）+ placebo（随机 token）"""
    import torch
    cond = CONDITIONS[condition]
    prompt_id, prompt_text = prompt
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
    cfg = dict(cfg)
    cfg['rep_count'] = 0
    cfg['top1_breaks'] = 0
    cfg['rescue'] = 0
    cfg['top_p_drop'] = 0
    cfg['mode'] = cond['mode']
    cfg['beta'] = cond.get('beta', 0.0)
    cfg['union'] = cond.get('union', False)
    cfg['beam'] = cond.get('beam', 0)
    cfg['alpha'] = cond.get('alpha', cfg.get('alpha', 0.1))  # v0.78 势场 EWMA 系数
    placebo = cond.get('placebo', False)
    # v0.68-4 外部篇核心：生成前一次性提取（意图核心全程稳定——不随生成更新）
    ext_theme = None
    ts = cond.get('theme_source')
    if ts == 'prompt':
        ext_theme = extract_theme_words(prompt_text, pseg, tok, K=cfg['K'])
    elif ts == 'human':
        hf = BASE / 'data' / 'bilingual_test' / 'human_zh' / f"{HUMAN_BIND[prompt_id]}.txt"
        if hf.exists():
            ext_theme = extract_theme_words(hf.read_text(encoding='utf-8'), pseg, tok, K=cfg['K'])
    if ext_theme:
        print(f'  外部篇核心主题词（{ts}）: {ext_theme}')
    # v0.69-2 外部锚定闭环：prompt 外部核心（句元指纹均值——生成前固定——观测口径修正）
    ext_core = None
    if cfg.get('mode') == 'pid_kalman_ext':
        import torch as _T
        from subclause_structure import split_subclauses as _split
        from para_dimensions import fingerprint as _fp, norm_rows as _nr
        ss = [s for s in _split(prompt_text) if len(s) >= 3]
        if ss:
            with _T.no_grad():
                sv = enc.encode(ss, normalize_embeddings=True, batch_size=16,
                                show_progress_bar=False, device='cpu')
                SV = _T.from_numpy(sv.astype(np.float32)).to('cpu')
                F = _nr(_fp(SV, disc)).detach().cpu().numpy()
                ext_core = F.mean(0)
            print(f'  外部核心构建 ✓（prompt {len(ss)} 句元——观测口径修正）')

    ids = tok.encode(prompt_text, return_tensors='pt').to(device)
    # ===== v0.69-4 句级种子（最小有效起点测试）——v0.73-3 seed_only（无 beam 变体）=====
    if cfg.get('mode') in ('sentence_seed_beam', 'seed_only'):
        segs = []
        for si in range(3):
            if si == 0:
                # 段 1：第 1 句前 12 token β=0.5 概率层引导（prompt 主题词）——之后自由
                tw0 = extract_theme_words(prompt_text[:90], pseg, tok, K=cfg['K'])
                lookup0 = build_theme_lookup(tw0, tok) if tw0 else None
                cfg['mode'] = 'prob'
                cfg['beta'] = 0.5
                out_ids = ids[0].cpu().tolist()
                n = 0
                while not segment_done(tok.decode(out_ids[ids.shape[1]:]), n):
                    cur = torch.tensor([out_ids], device=device)
                    lk = lookup0 if n < 12 else None   # 句级种子：前 12 token 引导——之后自由
                    nxt, m, t1 = sample_next(model, tok, cur, lk, cfg, device, rng, vocab_cache)
                    out_ids.append(nxt)
                    n += 1
                    if n >= SEG_MAX:
                        break
                text = tok.decode(out_ids[ids.shape[1]:])
                n_steps, matched, top1s = n, 0, 0
                cfg['mode'] = condition
                strat = 'sentence_seed'
                dims = monitor_segment(text, enc, disc, pseg, device='cpu')
            elif condition == 'sentence_seed_beam':
                # 段 2/3：无条件 beam5（与 beam5_ctl 一致）
                strat, beam_n = 'beam5', 5
                cfg['beam'] = beam_n
                cands = []
                for k in range(beam_n):
                    rng_k = np.random.default_rng(seed * 100 + si * 10 + k)
                    cfg_k = dict(cfg)
                    cfg_k['rep_count'] = 0
                    cfg_k['top1_breaks'] = 0
                    t_k, n_k, m_k, t1_k = generate_segment(model, tok, ids, None, cfg_k, device,
                                                           rng_k, ids.shape[1], vocab_cache)
                    d_k = monitor_segment(t_k, enc, disc, pseg, device='cpu')
                    cands.append((t_k, n_k, m_k, t1_k, d_k))
                best = max(cands, key=lambda c: c[4]['sent_proj'] if c[4] else -1)
                text, n_steps, matched, top1s, dims = best
            else:
                # v0.73-3 seed_only：段 2/3 完全自由（无 beam 无注入——种子效应纯化）
                strat = 'free'
                cfg['beam'] = 0
                text, n_steps, matched, top1s = generate_segment(model, tok, ids, None, cfg, device,
                                                                 rng, ids.shape[1], vocab_cache)
                dims = monitor_segment(text, enc, disc, pseg, device='cpu')
            seg_rec = {'seg': si + 1, 'text': text, 'n_steps': n_steps,
                       'match_rate': 0.0, 'top1_breaks': top1s, 'mode': condition,
                       'rescue_rate': 0.0, 'top_p_drop': 0, 'theme_words': [],
                       'guided': si == 0, 'beam': None, 'strategy': {'strategy': strat}}
            if dims:
                seg_rec['dims'] = {k: round(float(v), 4) for k, v in dims.items() if k != 'disc'}
                seg_rec['disc'] = round(float(dims['disc']), 4)
            else:
                seg_rec['dims'] = None
            segs.append(seg_rec)
            ids = torch.tensor([ids[0].cpu().tolist() + tok.encode(text)], device=device)
            if device == 'cuda':
                torch.cuda.empty_cache()
            print(f'[{condition}|{prompt[0]}|s{seed}] 段{si+1} {strat}——sent_proj {dims["sent_proj"] if dims else "NA"}')
        return {'run_id': f'{condition}-{prompt[0]}-s{seed}', 'condition': condition,
                'prompt_id': prompt[0], 'seed': seed, 'status': 'done',
                'triggered': True, 'segs': segs}

    # ===== v0.69-3 段落级策略闭环（独立循环）=====
    if cfg.get('mode') in ('p_kalman_strategy', 'beam5_ctl'):
        from kalman_pid import Kalman2D
        kf = Kalman2D(q=cfg.get('kq', 0.001), r=cfg.get('kr', 0.0025))
        strat_trace = []
        segs = []
        is_strategy = cfg.get('mode') == 'p_kalman_strategy'
        for si in range(3):
            if si == 0:
                # 段 1：轻量基线（β=0.5 概率层——prompt 主题词——用户修正：减小段 1 方差——两条件同款）
                tw0 = extract_theme_words(prompt_text[:90], pseg, tok, K=cfg['K'])
                lookup = build_theme_lookup(tw0, tok) if tw0 else None
                cfg['mode'] = 'prob'
                cfg['beta'] = 0.5
                cfg['beam'] = 0
                text, n_steps, matched, top1s = generate_segment(model, tok, ids, lookup, cfg, device,
                                                                 rng, ids.shape[1], vocab_cache)
                cfg['mode'] = 'p_kalman_strategy' if is_strategy else 'beam5_ctl'
                strat = 'seg1_baseline'
                dims = monitor_segment(text, enc, disc, pseg, device='cpu')
            else:
                # 段 2/3：策略条件=Kalman 预测分配（强 beam5/弱 beam3/自由）——控制条件=无条件 beam5
                # v0.73-2 修复：predict() 推进上一段后验 → 预测本段的先验（旧实现读 kf.x_pred 是
                # 上一次 step 的 predict 输出=预测上一段——S2 r=0.15 的直接代码原因——离线验证修正后 r=0.509）
                x_pred = kf.predict()
                if is_strategy:
                    e_pred = cfg.get('ktarget', 0.90) - x_pred
                    if e_pred > 0.05:
                        strat, beam_n = 'strong', 5
                    elif e_pred > 0.02:
                        strat, beam_n = 'weak', 3
                    else:
                        strat, beam_n = 'free', 0
                else:
                    strat, beam_n, e_pred = 'strong(ctl)', 5, 0.0
                cfg['beam'] = beam_n
                if beam_n > 0:
                    cands = []
                    for k in range(beam_n):
                        rng_k = np.random.default_rng(seed * 100 + si * 10 + k)
                        cfg_k = dict(cfg)
                        cfg_k['rep_count'] = 0
                        cfg_k['top1_breaks'] = 0
                        t_k, n_k, m_k, t1_k = generate_segment(model, tok, ids, None, cfg_k, device,
                                                               rng_k, ids.shape[1], vocab_cache)
                        d_k = monitor_segment(t_k, enc, disc, pseg, device='cpu')
                        cands.append((t_k, n_k, m_k, t1_k, d_k))
                    best = max(cands, key=lambda c: c[4]['sent_proj'] if c[4] else -1)
                    text, n_steps, matched, top1s, dims = best
                else:
                    text, n_steps, matched, top1s = generate_segment(model, tok, ids, None, cfg, device,
                                                                     rng, ids.shape[1], vocab_cache)
                    dims = monitor_segment(text, enc, disc, pseg, device='cpu')
            # 段结束后：实测 → Kalman 更新（下一段先验）
            z = dims['sent_proj'] if dims else 0.85
            x_est, x_pred_next = kf.step(z)
            strat_trace.append({'seg': si + 1, 'strategy': strat, 'x_pred': round(x_pred if si else 0.85, 4),
                                'e_pred': round(e_pred if si else 0.0, 4), 'z': round(float(z), 4),
                                'x_est': round(x_est, 4)})
            seg_rec = {'seg': si + 1, 'text': text, 'n_steps': n_steps,
                       'match_rate': round(matched / n_steps, 4) if n_steps else 0,
                       'top1_breaks': top1s, 'mode': 'p_kalman_strategy' if is_strategy else 'beam5_ctl',
                       'rescue_rate': 0.0, 'top_p_drop': 0,
                       'theme_words': [], 'guided': lookup is not None,
                       'beam': None, 'strategy': strat_trace[-1]}
            if dims:
                seg_rec['dims'] = {k: round(float(v), 4) for k, v in dims.items() if k != 'disc'}
                seg_rec['disc'] = round(float(dims['disc']), 4)
            else:
                seg_rec['dims'] = None
            segs.append(seg_rec)
            ids = torch.tensor([ids[0].cpu().tolist() + tok.encode(text)], device=device)
            if device == 'cuda':
                torch.cuda.empty_cache()
            print(f'[{condition}|{prompt[0]}|s{seed}] 段{si+1} {strat}——sent_proj {dims["sent_proj"] if dims else "NA"}'
                  f'——x_pred {strat_trace[-1]["x_pred"]}——e_pred {strat_trace[-1]["e_pred"]}')
        return {'run_id': f'{condition}-{prompt[0]}-s{seed}', 'condition': condition,
                'prompt_id': prompt[0], 'seed': seed, 'status': 'done',
                'triggered': True, 'segs': segs, 'strategy_trace': strat_trace}

    # ===== v0.73 虚拟 token 意图注入（模块 5——MLP 映射层——fingerprint→embedding 空间）=====
    if cfg.get('mode') in ('vt_oracle', 'vt_ext', 'vt_kalman', 'vt_seed_beam', 'vt_seed', 'vt_kalman_seed',
                           'vt_kalman_gate', 'vt_gate_beam',
                           'vt_field', 'vt_field_persist', 'vt_field_frozen', 'vt_field_full'):
        import torch as _T
        from train_intent_mlp import MLP as _MLP
        from subclause_structure import split_subclauses as _split
        from para_dimensions import fingerprint as _fp, norm_rows as _nr
        _mlp = _MLP(64, 256, model.config.hidden_size).to(device).eval()
        _mlp.load_state_dict(torch.load(str(BASE / 'data/intent_prior_model/mlp_checkpoint.pt'),
                                         map_location=device))

        _TOK_NORM = 0.46  # Qwen2.5-0.5B token 嵌入平均范数（v0.73 实测——φ 输出需同尺度）

        def vtok_fn(fp):
            if fp is None:
                return None
            with _T.no_grad():
                x = _T.from_numpy(np.asarray(fp, dtype=np.float32).reshape(1, -1)).to(device)
                e = _mlp(x)
                # 范数校准：φ 原始输出范数 ~3.3（MSE 训练外推）——真实 token 嵌入均值 ~0.46
                # 注入向量与真实 token 同尺度——否则注意力被 V 主导（key 范数 ~7 倍）生成病态
                e = e / (e.norm() + 1e-9) * _TOK_NORM
                return e.unsqueeze(1).to(next(model.parameters()).dtype)  # [1, 1, 896]

        def core_of_texts(texts):
            """文本句元指纹均值核心（归一化 64 维方向）"""
            ss = []
            for t in texts:
                for p in t.split('\n'):
                    p = p.strip()
                    if len(p) >= 3:
                        ss += [s for s in _split(p) if len(s) >= 3]
            if not ss:
                return None
            with _T.no_grad():
                sv = enc.encode(ss, normalize_embeddings=True, batch_size=16,
                                show_progress_bar=False, device='cpu')
                SV = _T.from_numpy(sv.astype(np.float32)).to('cpu')
                F = _nr(_fp(SV, disc)).detach().cpu().numpy()
                c = F.mean(0)
                return c / (np.linalg.norm(c) + 1e-9)

        mode_vt = cfg['mode']
        v_dir = None
        field_T = None
        if mode_vt in ('vt_ext', 'vt_seed_beam', 'vt_seed', 'vt_kalman_gate', 'vt_gate_beam', 'vt_field'):
            v_dir = core_of_texts([prompt_text])
            print(f'  {mode_vt} 外部核心 ✓（prompt 句元指纹均值——可实现锚）')
        elif mode_vt in ('vt_field_persist', 'vt_field_frozen', 'vt_field_full'):
            # v0.78 势场目标 T=人类核心（离线 field_target.json——independent_test B- 文档）
            f_t = BASE / 'data' / 'dim_analysis' / 'field_target.json'
            if not f_t.exists():
                print(f'  {mode_vt}: 缺少 field_target.json——报错跳过（先跑 engine_field_evidence.py）')
                return {'run_id': f'{condition}-{prompt[0]}-s{seed}', 'condition': condition,
                        'prompt_id': prompt[0], 'seed': seed, 'status': 'skip_missing_target', 'segs': []}
            field_T = np.asarray(json.loads(f_t.read_text(encoding='utf-8'))['human_core'], dtype=np.float32)
            field_T = field_T / (np.linalg.norm(field_T) + 1e-9)
            v_dir = field_T
            print(f'  {mode_vt} 人类核心 T ✓（field_target.json——32 篇独立测试人类文档均值）')
        elif mode_vt == 'vt_oracle':
            # lookahead oracle：none 同 prompt/seed 的全篇核心（事后可知——oracle 模拟）
            f_man = OUT / 'manifest.json'
            runs_m = json.loads(f_man.read_text(encoding='utf-8')) if f_man.exists() else []
            none_run = next((r for r in runs_m if r.get('condition') == 'none'
                             and r.get('prompt_id') == prompt_id and r.get('seed') == seed
                             and r.get('status') == 'done'), None)
            if none_run is None:
                print(f'  vt_oracle: 缺少 none-{prompt_id}-s{seed}——跳过')
                return {'run_id': f'{condition}-{prompt[0]}-s{seed}', 'condition': condition,
                        'prompt_id': prompt[0], 'seed': seed, 'status': 'skip_no_none', 'segs': []}
            v_dir = core_of_texts([seg['text'] for seg in none_run['segs'] if seg.get('text')])
            print(f'  vt_oracle 全篇核心 ✓（none-{prompt_id}-s{seed} 参考文本——lookahead oracle）')
        # v0.74 阶段 C：维度扰动（dim_perturb=(dim, mode, val)——替换/缩放/加噪——因果干预入口）
        if v_dir is not None and cfg.get('dim_perturb'):
            dp_dim, dp_mode, dp_val = cfg['dim_perturb']
            if dp_mode == 'replace':
                v_dir[dp_dim] = dp_val
            elif dp_mode == 'scale':
                v_dir[dp_dim] *= dp_val
            elif dp_mode == 'noise':
                v_dir[dp_dim] += dp_val
            v_dir = v_dir / (np.linalg.norm(v_dir) + 1e-9)
            print(f'  dim_perturb: dim{dp_dim} {dp_mode}={dp_val:.4f} ✓')
        vtok = vtok_fn(v_dir) if v_dir is not None else None
        segs = []
        # v0.78 势场状态（R=根意图/R0=初始/T=目标/alpha/buf=句元窗/trace）
        # 修正（v0.78-2）：R0 统一=prompt 核心——R 从 prompt 出发被表层压力拉向 T——
        # ΔR_T = cos(R_end,T)−cos(R0,T) 从非 1.0 起点出发才可测"传导位移"（避免天花板效应）
        if mode_vt in ('vt_field', 'vt_field_persist', 'vt_field_frozen', 'vt_field_full'):
            _bands = json.loads((BASE / 'data' / 'dim_analysis' / 'planner_targets.json').read_text(
                encoding='utf-8'))['dim_bands']
            band_stds = np.zeros(64, dtype=np.float32)
            for b in _bands:
                band_stds[b['dim']] = b['human']['band_std']
            _alpha = cfg.get('alpha', 0.1)
            _r0 = core_of_texts([prompt_text])  # R0=构造性根意图起点（prompt 核心）
            state = {'running': None,
                     'R': None, 'R0': _r0.astype(np.float32).copy(),
                     'T': field_T.astype(np.float32).copy() if field_T is not None else None,
                     'alpha': _alpha, 'buf': [], 'trace': [], 'inject_target': True,
                     'band_stds': band_stds}
            print(f'  {mode_vt} 势场初始化 ✓——R0=prompt 核心——T={"人类核心" if field_T is not None else "无(自治)"}——alpha={_alpha}')
        else:
            state = {'running': None}
        for si in range(3):
            if mode_vt in ('vt_field', 'vt_field_persist', 'vt_field_frozen', 'vt_field_full') and si == 0:
                # v0.78 段 1：种子 + 全强度注入（同 vt_gate_beam 框架——可比）+ R 初始化
                tw0 = extract_theme_words(prompt_text[:90], pseg, tok, K=cfg['K'])
                lookup0 = build_theme_lookup(tw0, tok) if tw0 else None
                cfg['mode'] = 'prob'
                cfg['beta'] = 0.5
                vtok_s1 = vtok if mode_vt != 'vt_field' else vtok_fn(state['R0'])
                state['R'] = state['R0'].copy()
                out_ids = ids[0].cpu().tolist()
                n = 0
                while not segment_done(tok.decode(out_ids[ids.shape[1]:]), n):
                    cur = torch.tensor([out_ids], device=device)
                    lk = lookup0 if n < 12 else None
                    nxt, m, t1 = sample_next(model, tok, cur, lk, cfg, device, rng, vocab_cache,
                                             vtok_emb=vtok_s1, vtok_pos=ids.shape[1])
                    out_ids.append(nxt)
                    n += 1
                    if n >= SEG_MAX:
                        break
                text = tok.decode(out_ids[ids.shape[1]:])
                n_steps, matched, top1s = n, 0, 0
                cfg['mode'] = mode_vt
                dims = monitor_segment(text, enc, disc, pseg, device='cpu')
            elif mode_vt in ('vt_field', 'vt_field_persist', 'vt_field_frozen', 'vt_field_full'):
                # v0.78 段 2/3：势场循环（persist/frozen 段 3 注入关闭——内化检验；frozen R 不更新）
                state['inject_target'] = not (mode_vt in ('vt_field_persist', 'vt_field_frozen') and si >= 2)
                freeze_R = (mode_vt == 'vt_field_frozen')
                text, n_steps, matched, top1s = generate_segment_field(
                    model, tok, ids, None, cfg, device, rng, ids.shape[1], vocab_cache,
                    vtok_fn, enc, disc, state, state['band_stds'], freeze_R=freeze_R)
                dims = monitor_segment(text, enc, disc, pseg, device='cpu')
            elif mode_vt in ('vt_seed_beam', 'vt_seed') and si == 0:
                # 段 1：句级种子（前 12 token β=0.5 概率层引导——v0.69-4 最小干预）+ V 注入
                tw0 = extract_theme_words(prompt_text[:90], pseg, tok, K=cfg['K'])
                lookup0 = build_theme_lookup(tw0, tok) if tw0 else None
                cfg['mode'] = 'prob'
                cfg['beta'] = 0.5
                out_ids = ids[0].cpu().tolist()
                n = 0
                while not segment_done(tok.decode(out_ids[ids.shape[1]:]), n):
                    cur = torch.tensor([out_ids], device=device)
                    lk = lookup0 if n < 12 else None   # 句级种子：前 12 token 引导——之后自由
                    nxt, m, t1 = sample_next(model, tok, cur, lk, cfg, device, rng, vocab_cache,
                                             vtok_emb=vtok, vtok_pos=ids.shape[1])
                    out_ids.append(nxt)
                    n += 1
                    if n >= SEG_MAX:
                        break
                text = tok.decode(out_ids[ids.shape[1]:])
                n_steps, matched, top1s = n, 0, 0
                cfg['mode'] = mode_vt
                dims = monitor_segment(text, enc, disc, pseg, device='cpu')
            elif mode_vt == 'vt_seed_beam':
                # 段 2/3：beam5 判别器选优 + V 注入（每候选同 V）
                cfg['beam'] = 5
                cands = []
                for k in range(5):
                    rng_k = np.random.default_rng(seed * 100 + si * 10 + k)
                    cfg_k = dict(cfg)
                    cfg_k['rep_count'] = 0
                    cfg_k['top1_breaks'] = 0
                    t_k, n_k, m_k, t1_k = generate_segment(
                        model, tok, ids, None, cfg_k, device, rng_k, ids.shape[1], vocab_cache, vtok_emb=vtok)
                    d_k = monitor_segment(t_k, enc, disc, pseg, device='cpu')
                    cands.append((t_k, n_k, m_k, t1_k, d_k))
                best = max(cands, key=lambda c: c[4]['sent_proj'] if c[4] else -1)
                text, n_steps, matched, top1s, dims = best
            elif mode_vt == 'vt_kalman' and si == 0:
                state['running'] = core_of_texts([prompt_text])  # 段 1 初始意图方向=prompt 核心
                vtok = vtok_fn(state['running'])
            elif mode_vt == 'vt_kalman_seed' and si == 0:
                # v0.73-3：段 1 种子 + 在线意图方向初始化
                state['running'] = core_of_texts([prompt_text])
                vtok = vtok_fn(state['running'])
                tw0 = extract_theme_words(prompt_text[:90], pseg, tok, K=cfg['K'])
                lookup0 = build_theme_lookup(tw0, tok) if tw0 else None
                cfg['mode'] = 'prob'
                cfg['beta'] = 0.5
                out_ids = ids[0].cpu().tolist()
                n = 0
                while not segment_done(tok.decode(out_ids[ids.shape[1]:]), n):
                    cur = torch.tensor([out_ids], device=device)
                    lk = lookup0 if n < 12 else None
                    nxt, m, t1 = sample_next(model, tok, cur, lk, cfg, device, rng, vocab_cache,
                                             vtok_emb=vtok, vtok_pos=ids.shape[1])
                    out_ids.append(nxt)
                    n += 1
                    if n >= SEG_MAX:
                        break
                text = tok.decode(out_ids[ids.shape[1]:])
                n_steps, matched, top1s = n, 0, 0
                cfg['mode'] = mode_vt
                dims = monitor_segment(text, enc, disc, pseg, device='cpu')
            elif mode_vt == 'vt_kalman_gate' and si == 0:
                # v0.73-3：段 1 种子 + 全强度注入（门控从段 2 开始）
                from kalman_pid import Kalman2D
                _kf_gate = Kalman2D(q=0.001, r=0.0025)
                tw0 = extract_theme_words(prompt_text[:90], pseg, tok, K=cfg['K'])
                lookup0 = build_theme_lookup(tw0, tok) if tw0 else None
                cfg['mode'] = 'prob'
                cfg['beta'] = 0.5
                out_ids = ids[0].cpu().tolist()
                n = 0
                while not segment_done(tok.decode(out_ids[ids.shape[1]:]), n):
                    cur = torch.tensor([out_ids], device=device)
                    lk = lookup0 if n < 12 else None
                    nxt, m, t1 = sample_next(model, tok, cur, lk, cfg, device, rng, vocab_cache,
                                             vtok_emb=vtok, vtok_pos=ids.shape[1])
                    out_ids.append(nxt)
                    n += 1
                    if n >= SEG_MAX:
                        break
                text = tok.decode(out_ids[ids.shape[1]:])
                n_steps, matched, top1s = n, 0, 0
                cfg['mode'] = mode_vt
                strat = 'gate:1.0'
                dims = monitor_segment(text, enc, disc, pseg, device='cpu')
            elif mode_vt == 'vt_kalman_gate':
                # v0.73-3 段 2/3：修正 Kalman 预测 e_pred → 注入强度（无 beam 下用强度替代 beam 分配）
                x_pred = _kf_gate.predict()
                e_pred = 0.90 - x_pred
                if e_pred > 0.05:
                    bf = 1.0
                elif e_pred > 0.02:
                    bf = 0.5
                else:
                    bf = 0.0
                vtok_g = vtok if bf >= 1.0 else (vtok * bf if bf > 0 else None)
                strat = f'gate:{bf}'
                text, n_steps, matched, top1s = generate_segment(
                    model, tok, ids, None, cfg, device, rng, ids.shape[1], vocab_cache, vtok_emb=vtok_g)
                dims = monitor_segment(text, enc, disc, pseg, device='cpu')
                z = dims['sent_proj'] if dims else 0.85
                _kf_gate.step(z)
            elif mode_vt == 'vt_gate_beam' and si == 0:
                # 待办①段 1：种子 + 全强度注入（门控从段 2 开始——同 vt_kalman_gate）
                from kalman_pid import Kalman2D
                _kf_gate = Kalman2D(q=0.001, r=0.0025)
                tw0 = extract_theme_words(prompt_text[:90], pseg, tok, K=cfg['K'])
                lookup0 = build_theme_lookup(tw0, tok) if tw0 else None
                cfg['mode'] = 'prob'
                cfg['beta'] = 0.5
                out_ids = ids[0].cpu().tolist()
                n = 0
                while not segment_done(tok.decode(out_ids[ids.shape[1]:]), n):
                    cur = torch.tensor([out_ids], device=device)
                    lk = lookup0 if n < 12 else None
                    nxt, m, t1 = sample_next(model, tok, cur, lk, cfg, device, rng, vocab_cache,
                                             vtok_emb=vtok, vtok_pos=ids.shape[1])
                    out_ids.append(nxt)
                    n += 1
                    if n >= SEG_MAX:
                        break
                text = tok.decode(out_ids[ids.shape[1]:])
                n_steps, matched, top1s = n, 0, 0
                cfg['mode'] = mode_vt
                strat = 'gate:1.0'
                dims = monitor_segment(text, enc, disc, pseg, device='cpu')
            elif mode_vt == 'vt_gate_beam':
                # 待办①段 2/3：Kalman 门控 → {beam 开关, 注入强度} 双自由度
                x_pred = _kf_gate.predict()
                e_pred = 0.90 - x_pred
                if e_pred > 0.05:
                    bf, beam_n = 1.0, 5      # 预测强漂移：beam5 + 全强度注入
                elif e_pred > 0.02:
                    bf, beam_n = 0.5, 0      # 中等：半强度注入（无 beam）
                else:
                    bf, beam_n = 0.0, 0      # 达标：自由（无注入无 beam——省成本）
                vtok_g = vtok if bf >= 1.0 else (vtok * bf if bf > 0 else None)
                strat = f'gate:{bf},beam:{beam_n}'
                if beam_n > 0:
                    cands = []
                    for k in range(beam_n):
                        rng_k = np.random.default_rng(seed * 100 + si * 10 + k)
                        cfg_k = dict(cfg)
                        cfg_k['rep_count'] = 0
                        cfg_k['top1_breaks'] = 0
                        t_k, n_k, m_k, t1_k = generate_segment(
                            model, tok, ids, None, cfg_k, device, rng_k, ids.shape[1], vocab_cache,
                            vtok_emb=vtok_g)
                        d_k = monitor_segment(t_k, enc, disc, pseg, device='cpu')
                        cands.append((t_k, n_k, m_k, t1_k, d_k))
                    best = max(cands, key=lambda c: c[4]['sent_proj'] if c[4] else -1)
                    text, n_steps, matched, top1s, dims = best
                else:
                    text, n_steps, matched, top1s = generate_segment(
                        model, tok, ids, None, cfg, device, rng, ids.shape[1], vocab_cache, vtok_emb=vtok_g)
                    dims = monitor_segment(text, enc, disc, pseg, device='cpu')
                z = dims['sent_proj'] if dims else 0.85
                _kf_gate.step(z)
            if mode_vt in ('vt_kalman',) or (mode_vt == 'vt_kalman_seed' and si >= 1):
                text, n_steps, matched, top1s = generate_segment_vt(
                    model, tok, ids, None, cfg, device, rng, ids.shape[1], vocab_cache,
                    vtok_fn, enc, disc, state)
            elif mode_vt not in ('vt_seed_beam', 'vt_kalman_gate', 'vt_gate_beam'):
                text, n_steps, matched, top1s = generate_segment(
                    model, tok, ids, None, cfg, device, rng, ids.shape[1], vocab_cache, vtok_emb=vtok)
            dims = monitor_segment(text, enc, disc, pseg, device='cpu')
            seg_rec = {'seg': si + 1, 'text': text, 'n_steps': n_steps,
                       'match_rate': 0.0, 'top1_breaks': top1s, 'mode': mode_vt,
                       'rescue_rate': 0.0, 'top_p_drop': 0, 'theme_words': [],
                       'guided': True, 'beam': None,
                       'strategy': {'strategy': strat if mode_vt in ('vt_kalman_gate', 'vt_gate_beam') else 'vt_inject'}}
            if dims:
                seg_rec['dims'] = {k: round(float(v), 4) for k, v in dims.items() if k != 'disc'}
                seg_rec['disc'] = round(float(dims['disc']), 4)
            else:
                seg_rec['dims'] = None
            if mode_vt in ('vt_field', 'vt_field_persist', 'vt_field_frozen', 'vt_field_full'):
                seg_rec['field_trace'] = state['trace'][-30:]  # 本段句元轨迹（p/e/bf/cosR_R0/cosR_T/cosF_T）
            segs.append(seg_rec)
            ids = torch.tensor([ids[0].cpu().tolist() + tok.encode(text)], device=device)
            if device == 'cuda':
                torch.cuda.empty_cache()
            print(f'[{condition}|{prompt[0]}|s{seed}] 段{si+1} vt注入——'
                  f'sent_proj {dims["sent_proj"] if dims else "NA"}')
        # v0.78 run 级势场汇总（R 漂移轨迹端点）
        if mode_vt in ('vt_field', 'vt_field_persist', 'vt_field_frozen', 'vt_field_full'):
            R_end = state.get('R')
            R0 = state.get('R0')
            T = state.get('T')
            trace = state.get('trace', [])
            run_summary = {
                'alpha': state.get('alpha'),
                'n_clauses': len(trace),
                'n_pullback_bf1': sum(1 for t in trace if t['bf'] >= 1.0),
                'n_pullback_bf05': sum(1 for t in trace if t['bf'] == 0.5),
                'n_free_bf0': sum(1 for t in trace if t['bf'] == 0.0),
                'cos_R0_T': round(float(R0 @ T), 4) if T is not None else None,
                'cos_R_end_T': round(float(R_end @ T), 4) if (R_end is not None and T is not None) else None,
                'cos_R_end_R0': round(float(R_end @ R0), 4) if R_end is not None else None,
                'delta_R_T': round(float(R_end @ T) - float(R0 @ T), 4) if (R_end is not None and T is not None) else None,
                'mean_cosF_T': round(float(np.mean([t['cosF_T'] for t in trace if t['cosF_T'] is not None])), 4)
                if any(t['cosF_T'] is not None for t in trace) else None,
                'mean_cosF_R': round(float(np.mean([t['cosF_R'] for t in trace])), 4) if trace else None,
                'mean_e': round(float(np.mean([t['e'] for t in trace])), 4) if trace else None,
            }
            return {'run_id': f'{condition}-{prompt[0]}-s{seed}', 'condition': condition,
                    'prompt_id': prompt[0], 'seed': seed, 'status': 'done',
                    'triggered': True, 'segs': segs, 'field_summary': run_summary}
        return {'run_id': f'{condition}-{prompt[0]}-s{seed}', 'condition': condition,
                'prompt_id': prompt[0], 'seed': seed, 'status': 'done',
                'triggered': True, 'segs': segs}

    segs = []
    theme_words = None
    lookup = None
    trigger_any = False
    good_buffer = None  # (theme_words, text)——最近好段（sent_proj ≥ 阈值）——稳定意图核心
    prev_triggered = False
    for si in range(3):
        # v0.68-3：beam 选优（路径 4——判别器评估）——仅触发段（用户优化——成本可控）
        beam_active = bool(cfg.get('beam')) and si >= 1 and prev_triggered
        beam_info = None
        if beam_active:
            cands = []
            for k in range(cfg['beam']):
                rng_k = np.random.default_rng(seed * 100 + si * 10 + k)
                cfg_k = dict(cfg)
                cfg_k['rep_count'] = 0
                cfg_k['top1_breaks'] = 0
                cfg_k['rescue'] = 0
                cfg_k['top_p_drop'] = 0
                t_k, n_k, m_k, t1_k = generate_segment(model, tok, ids, lookup, cfg_k, device, rng_k,
                                                       ids.shape[1], vocab_cache)
                d_k = monitor_segment(t_k, enc, disc, pseg, device='cpu')
                cands.append((t_k, n_k, m_k, t1_k, d_k))
            best = max(cands, key=lambda c: c[4]['sent_proj'] if c[4] else -1)
            text, n_steps, matched, top1s, dims = best
            beam_info = {'n_cand': len(cands),
                         'cand_sent_proj': [round(c[4]['sent_proj'], 4) if c[4] else None for c in cands],
                         'pick_gain': round(float(best[4]['sent_proj'] - np.mean(
                             [c[4]['sent_proj'] for c in cands if c[4]])), 4) if any(c[4] for c in cands) else None}
        else:
            if cfg.get('mode') in ('pid_kalman', 'pid_kalman_ext'):
                # v0.69 闭环：句元级 Kalman+PID（动态 β）——v0.69-2 外部锚定观测
                text, n_steps, matched, top1s, ctrl_trace = generate_segment_pid(
                    model, tok, ids, lookup, cfg, device, rng, ids.shape[1], vocab_cache, enc, disc, ext_core)
            else:
                text, n_steps, matched, top1s = generate_segment(model, tok, ids, lookup, cfg, device, rng,
                                                                 ids.shape[1], vocab_cache)
            dims = monitor_segment(text, enc, disc, pseg, device='cpu')
        seg_rec = {'seg': si + 1, 'text': text, 'n_steps': n_steps,
                   'match_rate': round(matched / n_steps, 4) if n_steps else 0,
                   'top1_breaks': top1s,
                   'mode': cfg['mode'],
                   'rescue_rate': round(cfg['rescue'] / n_steps, 4) if n_steps else 0,
                   'top_p_drop': cfg['top_p_drop'],
                   'theme_words': theme_words or [],
                   'guided': lookup is not None,
                   'beam': beam_info,
                   'ctrl_trace': ctrl_trace if (cfg.get('mode') in ('pid_kalman', 'pid_kalman_ext') and 'ctrl_trace' in dir()) else None}
        if dims:
            seg_rec['dims'] = {k: round(float(v), 4) for k, v in dims.items() if k != 'disc'}
            seg_rec['disc'] = round(float(dims['disc']), 4)
        else:
            seg_rec['dims'] = None
        segs.append(seg_rec)
        cfg['rescue'] = 0
        cfg['top_p_drop'] = 0
        # 触发决策（v0.68-2——好段 buffer——防漂移锁定）
        triggered = bool(dims and dims['sent_proj'] is not None and dims['sent_proj'] < THRESHOLD)
        prev_triggered = triggered
        if triggered:
            if ext_theme is not None:
                # v0.68-4 外部篇核心（用户理论修正）：固定外部锚——不用好段 buffer（自我强化漏洞）
                theme_words = ext_theme
            elif good_buffer is not None:
                theme_words = good_buffer[0]          # 稳定锚：最近好段的主题词
            else:
                # 无好段——回退 prompt 前 3 句主题词（最接近初始意图）
                theme_words = extract_theme_words(prompt_text[:90], pseg, tok, K=cfg['K'])
            if placebo:
                # 安慰剂：随机 token（数量=主题词数 K——固定 seed 可复现）
                prng = np.random.default_rng(SEED + seed)
                rand_ids = prng.choice(len(tok), size=cfg['K'], replace=False)
                lookup = {'words': [f'<rand{i}>' for i in rand_ids],
                          'fast_ids': set(rand_ids.tolist()), 'decode_cache': {}, 'g_t': None, 'g': None}
            elif condition != 'none' and cfg.get('mode') in ('logits', 'pid_kalman', 'pid_kalman_ext') and theme_words:
                # 基于 mode 判定（v0.68-4 修复：t3=logits+beam 需引导——beam5 mode=None 安全——v0.69 pid 系列同）
                lookup = build_theme_lookup(theme_words, tok)
            else:
                lookup = None
            trigger_any = True
        else:
            # 未触发：更新好段 buffer（不立即使用——下段触发才用）——外部锚条件关闭（纯外部锚）
            if ext_theme is None:
                tw = extract_theme_words(text, pseg, tok, K=cfg['K'])
                if tw:
                    good_buffer = (tw, text)
            lookup = None
        ids = torch.tensor([ids[0].cpu().tolist() + tok.encode(text)], device=device)
        if device == 'cuda':
            torch.cuda.empty_cache()
        line = f'[{condition}|{prompt[0]}|s{seed}] 段{si+1} {"引导" if seg_rec["guided"] else "自由"}——' \
               f'{n_steps}步——sent_proj {dims["sent_proj"] if dims else "NA"}——' \
               f'trigger {"✓" if triggered else "-"}——match {seg_rec["match_rate"]}——' \
               f'rescue {seg_rec["rescue_rate"]}——主题词 {theme_words if theme_words else "-"}'
        print(line)
    return {'run_id': f'{condition}-{prompt[0]}-s{seed}', 'condition': condition,
            'prompt_id': prompt[0], 'seed': seed, 'status': 'done',
            'triggered': trigger_any, 'segs': segs}


def main():
    global THRESHOLD
    ap = argparse.ArgumentParser()
    ap.add_argument('--pilot', action='store_true', help='lg05+lg10 × 3 prompt × seed1 先行（6 runs）')
    ap.add_argument('--threshold', type=float, default=0.85)
    ap.add_argument('--conditions', default=None, help='逗号分隔（如 lg05,lg10,lg_placebo——默认全部）')
    ap.add_argument('--seeds', default=None, help='逗号分隔种子（默认 0,1,2——placebo 除外）')
    ap.add_argument('--field-alpha', type=float, default=None,
                    help='v0.78 势场 EWMA 系数 α（默认用 CONDITIONS 定义值——α 扫描用）')
    ap.add_argument('--temperature', type=float, default=None,
                    help='v0.82 温度扫描（覆盖 cfg 温度——none 条件——跳跃机制验证）')
    a = ap.parse_args()
    THRESHOLD = a.threshold
    seeds_arg = [int(s) for s in a.seeds.split(',')] if a.seeds else None

    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    enc, disc, pseg = load_monitor('cpu')
    model, tok, device = load_gen_model('cuda')
    cfg = {'temperature': 0.9, 'top_k': 50, 'top_p': 0.9, 'K': 5}
    vocab_cache = build_vocab_cache(tok, vocab_size=model.config.vocab_size)

    f_man = OUT / 'manifest.json'
    runs = json.loads(f_man.read_text(encoding='utf-8')) if f_man.exists() else []
    done = {r['run_id'] for r in runs if r.get('status') == 'done'}

    if a.temperature is not None:
        cfg['temperature'] = a.temperature
        print(f'--temperature {a.temperature} ✓')
    if a.field_alpha is not None:
        for c in ('vt_field', 'vt_field_persist', 'vt_field_frozen', 'vt_field_full'):
            if c in CONDITIONS:
                CONDITIONS[c]['alpha'] = a.field_alpha
        print(f'--field-alpha {a.field_alpha} ✓（覆盖势场条件 α）')
    conds = a.conditions.split(',') if a.conditions else list(CONDITIONS)
    if a.pilot:
        matrix = [(c, p, 1) for c in conds if c in ('lg05', 'lg10', 'lg_placebo') for p in PROMPTS]
        if not matrix:
            matrix = [(c, p, 1) for c in conds for p in PROMPTS]
    else:
        matrix = []
        for c in conds:
            seeds = (0,) if c == 'lg_placebo' else (seeds_arg if seeds_arg else (0, 1, 2))
            for p in PROMPTS:
                for s in seeds:
                    matrix.append((c, p, s))
    print(f'矩阵 {len(matrix)} runs——阈值 {THRESHOLD}——已完成 {len(done)}')

    for condition, prompt, seed in matrix:
        run_id = f'{condition}-{prompt[0]}-s{seed}'
        if run_id in done:
            print(f'{run_id} 已存在——跳过')
            continue
        _t0 = time.time()
        r = run_one(condition, prompt, seed, enc, disc, pseg, model, tok, cfg, device, vocab_cache)
        r['time_s'] = round(time.time() - _t0, 1)  # v0.73-2 效率记录（run 级总耗时——生成+监测+注入）
        (OUT / 'texts' / condition / f'{run_id}.txt').parent.mkdir(parents=True, exist_ok=True)
        (OUT / 'texts' / condition / f'{run_id}.txt').write_text(
            '\n\n'.join(f'【段{seg["seg"]}】{seg["text"]}' for seg in r['segs']), encoding='utf-8')
        runs = [x for x in runs if x['run_id'] != run_id] + [r]
        f_man.write_text(json.dumps(runs, ensure_ascii=False, indent=1), encoding='utf-8')
        print(f'{run_id} 完成 ✓——触发 {r["triggered"]}')
    print('全部完成')


if __name__ == '__main__':
    main()
