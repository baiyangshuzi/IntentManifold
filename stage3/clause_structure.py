# -*- coding: utf-8 -*-
"""v0.50 P1.9 A 阶段：句元标注器（用户"把文本拆分为句元——比对句元功能结构"）

句号级拆分（对话引语不拆——引语内句号跳过）——8 维标签规则粗标（每条带置信度 0-1）
——置信度 <0.7 的句元送 LLM 精标（6 句/批）——精标回填。

8 维：FUNC(叙事功能 A/D/R/S/P/I)/DIR(指向 0-4)/COG(认知 0-3)/EMO(情绪载体 0-4)/
CONN(连接 0-4)/POL(对话力学 0-5)/TMP(时间密度 0-3)/H(熵贡献——整体统计后回填)
"""
import re, json, sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

# ---------- 规则词表 ----------
ACTION_VERBS = ("走", "站", "坐", "立", "跪", "拜", "拔", "抽", "提", "举", "掷", "摔",
                "拍", "按", "握", "抓", "扯", "推", "踩", "踏", "跨", "跃", "跳", "奔",
                "驰", "回头", "转身", "抬头", "低头", "皱眉", "眯", "睁", "咬", "抿",
                "叹", "哭", "笑", "瞪", "瞥", "望", "看", "瞧", "端", "递", "接", "放",
                "搁", "斟", "倒", "饮", "喝", "吞", "拨", "抚", "拂", "系", "解", "束",
                "披", "脱", "穿", "戴", "拭", "揩", "攥", "劈", "砍", "刺", "挡", "架",
                "弹", "捻", "摩")
REACT_MARKERS = ("僵", "颤", "怔", "愣", "惊", "顿", "缩", "滞")
EMOTION_WORDS = ("怒", "愤", "悲", "哀", "伤", "怕", "惧", "恐", "慌", "惊", "喜",
                 "忧", "恨", "厌", "羞", "愧", "痛", "苦", "急", "焦", "闷")
COG_KNOWN = ("看穿", "明白", "知道", "识破", "醒悟", "了然", "清楚", "察觉")
COG_PUZZLED = ("？", "什么", "怎", "糊涂", "不知", "不懂", "为何", "为何会", "想不通")
ENV_WORDS = ("风", "雪", "火", "光", "影", "夜", "墙", "门", "路", "街", "烛", "灯",
             "月", "天", "尘", "烟", "雨", "泥", "霜")
OBJECT_WORDS = ("刀", "剑", "杯", "碗", "酒", "斧", "枪", "箭", "弓", "绳", "袍",
                "案", "桌", "椅", "帘", "令箭", "马", "缰")
FILLER_CONN = ("但", "却", "然而", "可是", "只是")
CAUSE_CONN = ("因为", "所以", "因此", "于是")
PARA_CONN = ("此时", "这时", "次日", "翌日", "三日后", "片刻后", "半晌")

POL_RHET = ("难道", "岂", "何尝", "怎么", "为何", "凭什么")


SPEAKER_RE = re.compile(r'([一-龥]{1,4}?)(?:道|说|问|喝道|沉声道|答道|问道|笑道|叹道|怒道|斥道|接口道|又道)[：:]')


def split_clauses(text):
    """句元级切分（v0.50 P1.11 引语级拆分——多轮对话拆句元——用户"没有考虑句元联系"修复）：
    引语对（""/「」）内不拆——每个引语对（含前导叙述）独立句元——引语对间切分——
    引语外按句号级切分——返回 [(句文本, 段首, turn_group)]"""
    # 引语对区间
    ranges = []
    for m in re.finditer(r'[“"「」『』]([^“"」『』]*)[”"」』]', text):
        ranges.append((m.start(), m.end()))
    clauses = []
    cur = ""
    para_break = True
    last_quote_end = -1
    i = 0
    n = len(text)
    while i < n:
        # 引语开始→收集前导叙述+整个引语对=1 个句元
        in_q = False
        for s, e in ranges:
            if s <= i < e:
                in_q = True
                break
        if in_q:
            # 找当前引语对——收集后立即切分（引语对=句元边界——多轮对话拆句元）
            for s, e in ranges:
                if s <= i < e:
                    cur += text[i:e]
                    i = e
                    last_quote_end = e
                    c = cur.strip()
                    if len(c) >= 2:
                        clauses.append((c, para_break))
                    cur = ""
                    para_break = False
                    break
            continue
        ch = text[i]
        cur += ch
        if ch in "。！？!?":
            c = cur.strip()
            if len(c) >= 2:
                clauses.append((c, para_break))
            cur = ""
            para_break = False
        elif ch == "\n":
            para_break = True
        i += 1
    c = cur.strip()
    if len(c) >= 2:
        clauses.append((c, para_break))

    # turn_group：相邻含引语句元（中间无实质叙述）同轮次
    out = []
    turn = 0
    prev_d = -2
    for idx, (c, pb) in enumerate(clauses):
        has_quote = bool(re.search(r'[“"「」『』]', c))
        if has_quote and idx - prev_d <= 1:
            pass  # 同轮次
        elif has_quote:
            turn += 1
        prev_d = idx if has_quote else prev_d
        out.append((c, pb, turn if has_quote else 0))
    return out


# ---------- 规则粗标（每标签带置信度） ----------
def _rule_label(text, para_break):
    """8 维标签粗标——返回 (labels, conf, has_action)"""
    lb = {"FUNC": "S", "DIR": 0, "COG": 0, "EMO": 0, "CONN": 0, "POL": 0, "TMP": 0}
    conf = {}
    has_action = False

    # FUNC（v0.50 P1.11 含引语对=对话句——"是。"类独立引语无引导词也标 D）
    if re.search(r'[“"「」『』]', text) or \
            re.search(r'道[：:]|说[：:]|问[：:]|喝道|沉声道|答道|问道', text):
        lb["FUNC"] = "D"
        conf["FUNC"] = 0.9
        if any(v in text for v in ACTION_VERBS):
            has_action = True
    elif re.search(r'感到|觉得|意识到|心想|暗自|心头|脑子里', text):
        lb["FUNC"] = "I"
        conf["FUNC"] = 0.8
    elif re.search(r'看见|听见|听到|闻到|望见|瞥见', text):
        lb["FUNC"] = "P"
        conf["FUNC"] = 0.8
    elif re.search(r'[一](?:僵|颤|怔|愣|惊|顿|缩|滞)', text) or \
            any(m in text for m in REACT_MARKERS):
        lb["FUNC"] = "R"
        conf["FUNC"] = 0.7
    elif any(v in text for v in ACTION_VERBS):
        lb["FUNC"] = "A"
        conf["FUNC"] = 0.7
        has_action = True
    else:
        lb["FUNC"] = "S"
        conf["FUNC"] = 0.6

    # DIR
    if re.search(r'我|自己', text):
        lb["DIR"] = 4
        conf["DIR"] = 0.7
    elif any(o in text for o in OBJECT_WORDS):
        lb["DIR"] = 2
        conf["DIR"] = 0.7
    elif any(e in text for e in ENV_WORDS) and lb["FUNC"] in ("S", "P"):
        lb["DIR"] = 3
        conf["DIR"] = 0.6
    elif re.search(r'他|她|楚休红|武侯|蒲安礼|钱文义|陆经渔|张小凡|林惊羽', text):
        lb["DIR"] = 1
        conf["DIR"] = 0.7
    else:
        lb["DIR"] = 0
        conf["DIR"] = 0.6

    # COG（对话句联动 POL——对话本质=信息交换——追问=困惑/反问揭露=看穿——
    # v0.50 P1.9 改进：对话句 COG 不落 0——提高认知分布区分度）
    if any(k in text for k in COG_PUZZLED):
        lb["COG"] = 1
        conf["COG"] = 0.7
    elif any(k in text for k in COG_KNOWN):
        lb["COG"] = 2
        conf["COG"] = 0.7
    elif text.rstrip().endswith("……") or "半截" in text:
        lb["COG"] = 3
        conf["COG"] = 0.6
    elif lb["FUNC"] == "D":
        # 对话句认知联动：反问/揭露→看穿（2）追问→困惑（1）——默认对话=信息差载体
        if any(r in text for r in POL_RHET) or "？" in text:
            lb["COG"] = 1 if "？" in text and not any(r in text for r in POL_RHET) else 2
        else:
            lb["COG"] = 2  # 陈述对话=说话人知道更多（信息差）
        conf["COG"] = 0.6
    else:
        lb["COG"] = 0
        conf["COG"] = 0.6

    # EMO
    if any(w in text for w in EMOTION_WORDS):
        lb["EMO"] = 1
        conf["EMO"] = 0.8
    elif re.search(r'发白|渗血|收紧|握紧|发紧|干涩|发苦|发冷|颤抖|攥', text):
        lb["EMO"] = 2
        conf["EMO"] = 0.7
    elif "……" in text or "——" in text:
        lb["EMO"] = 4
        conf["EMO"] = 0.6
    elif lb["FUNC"] in ("S", "P") and any(e in text for e in ENV_WORDS):
        lb["EMO"] = 3
        conf["EMO"] = 0.5
    else:
        lb["EMO"] = 0
        conf["EMO"] = 0.6

    # CONN
    if any(w in text for w in CAUSE_CONN):
        lb["CONN"] = 2
        conf["CONN"] = 0.9
    elif any(w in text for w in FILLER_CONN):
        lb["CONN"] = 3
        conf["CONN"] = 0.9
    elif re.search(r'同时|一边|一面', text):
        lb["CONN"] = 1
        conf["CONN"] = 0.7
    elif para_break and not re.match(r'^(?:他|她|那|这)', text):
        lb["CONN"] = 4
        conf["CONN"] = 0.6
    else:
        lb["CONN"] = 0
        conf["CONN"] = 0.6

    # POL（仅对话句）
    if lb["FUNC"] == "D":
        if "？" in text:
            lb["POL"] = 2 if any(r in text for r in POL_RHET) else 1
            conf["POL"] = 0.7
        elif text.rstrip().endswith("……"):
            lb["POL"] = 3
            conf["POL"] = 0.6
        elif any(v in text for v in ("岔开", "不答", "没有回答", "转移")):
            lb["POL"] = 4
            conf["POL"] = 0.5
        elif any(v in text for v in ("弹", "拂", "转身", "走", "离开")):
            lb["POL"] = 5
            conf["POL"] = 0.5
        else:
            lb["POL"] = 3
            conf["POL"] = 0.5
    else:
        lb["POL"] = 0
        conf["POL"] = 0.9

    # TMP
    if re.search(r'渐|渐渐|缓缓|一直|久久', text):
        lb["TMP"] = 2
        conf["TMP"] = 0.7
    elif re.search(r'[一](?:弹|拂|僵|顿|震|收)', text):
        lb["TMP"] = 1
        conf["TMP"] = 0.7
    elif any(w in text for w in PARA_CONN):
        lb["TMP"] = 3
        conf["TMP"] = 0.7
    else:
        lb["TMP"] = 0
        conf["TMP"] = 0.6

    return lb, conf, has_action


def _speaker(text):
    """说话人/主人物提取（v0.50 P1.11——"XX道："引导语→说话人——独立引语继承）"""
    m = SPEAKER_RE.search(text)
    if m:
        return m.group(1)
    # 独立引语（"是。"）无引导语——无说话人（由轮次层轮换推断）
    return ""


def annotate(text, llm_client=None, model=None):
    """全文句元标注——规则粗标→置信度<0.7 送 LLM 精标→回填
    返回 [{idx, text, para_break, labels, conf, has_action, turn, speaker}]"""
    clauses = split_clauses(text)
    items = []
    low = []
    # 说话人轮换（独立引语继承：同轮次内 A→B→A 轮换）
    last_speaker = ""
    for idx, (c, pb, turn) in enumerate(clauses):
        lb, conf, ha = _rule_label(c, pb)
        sp = _speaker(c)
        if sp:
            last_speaker = sp
        items.append({"idx": idx, "text": c, "para_break": pb,
                      "labels": lb, "conf": conf, "has_action": ha,
                      "turn": turn, "speaker": sp or (last_speaker if turn else "")})
        if min(conf.values()) < 0.7:
            low.append(idx)
    # LLM 精标（批量 6 句/批——补标 FUNC/COG/POL）
    if low and llm_client:
        items = _llm_refine(items, low, llm_client, model)
    return items


def _llm_refine(items, low_idxs, client, model):
    """LLM 精标置信度低句元（COG/POL/FUNC——批量）"""
    for i in range(0, len(low_idxs), 6):
        batch = low_idxs[i:i + 6]
        paras = "\n".join(f"[{k}] {items[k]['text']}" for k in batch)
        try:
            resp = client.chat.completions.create(
                model=model, messages=[
                    {"role": "system",
                     "content": "你是叙事结构标注器。为每个句元标注 3 个标签（只输出 JSON 数组）："
                                "FUNC(动作A/对话D/反应R/状态S/感知P/内省I)、"
                                "COG(0无/1困惑/2看穿/3隐瞒)、"
                                "POL(0非对话/1追问/2反问/3揭露/4回避/5终止)。"
                                "判断依据：信息不对等（谁知道什么）、对话的攻守力学。"},
                    {"role": "user", "content": paras + "\n只输出 JSON 数组，每项 {FUNC,COG,POL}。"}],
                temperature=0.1, max_tokens=800,
                extra_body={"thinking": {"type": "disabled"}})
            txt = (resp.choices[0].message.content or "").strip()
            m = re.search(r"\[[\s\S]*\]", txt)
            if m:
                data = json.loads(m.group(0))
                for k, lb in zip(batch, data):
                    if isinstance(lb, dict):
                        for key in ("FUNC", "COG", "POL"):
                            if key in lb and lb[key] is not None:
                                items[k]["labels"][key] = lb[key]
                                items[k]["conf"][key] = 0.9
        except Exception as e:
            print(f"  精标批失败: {e}")
    return items


if __name__ == "__main__":
    t = open(sys.argv[1], encoding="utf-8").read() if len(sys.argv) > 1 else ""
    items = annotate(t)
    print(f"句元数: {len(items)}")
    for it in items[:15]:
        print(f"[{it['idx']}] {it['labels']} conf={min(it['conf'].values()):.1f} "
              f"{it['text'][:38]}")
