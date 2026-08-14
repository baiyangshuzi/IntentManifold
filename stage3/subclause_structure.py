# -*- coding: utf-8 -*-
"""v0.50 P1.13 补测：子句（逗号句元）切分+标注（用户——"句元和逗号句元结合使用，
分析句元结构与句元联系"）

切分规则（数据驱动——人类句元 89% 含逗号/引语内逗号占 19%——引语内不拆保住对话完整）：
  1. 引语对区间（复用 clause_structure 正则）整体 = 1 子句——内部逗号不拆
  2. 引语外按 ，； 切——顿号/破折号/省略号/冒号不切（枚举/未完标记不拆）
  3. 前导叙述与引语拆分（'沈彻开口，嗓子干涩，"你来了。"' → [沈彻开口][嗓子干涩]["你来了。"]）
  4. 子句 strip 后 len>=2 保留（与 split_clauses 同口径）

标注：逐子句复用 clause_structure._rule_label（无状态纯函数——毫秒级重算）；
  有效维 FUNC/TMP/CONN（子句级 DTW 用）；引语子句强制 FUNC=D、继承父句元 COG/POL；
  COG/POL 在子句级规则恒低区分度——仅继承展示，不参与子句匹配成本。
"""
import re

from clause_structure import _rule_label

QUOTE_RE = re.compile(r'[“"「」『』]([^“"」』]*)[”"」』]')
SUB_BOUNDARY = "，；"   # 引语外切分边界（顿号、破折号、省略号、冒号不切）


_SENT_END = "。！？!?"   # 句元末标点（引语外不进入子句——子句不含句子结束符）


def split_subclauses(text):
    """句元文本 → 子句列表（引语对整体=1 子句——内部逗号不拆；
    引语后叙述独立成子句——'沈彻开口，嗓子干涩，"你来了。"' → [沈彻开口][嗓子干涩]["你来了。"]）"""
    quote_ranges = [(m.start(), m.end()) for m in QUOTE_RE.finditer(text)]
    subs = []
    cur = ""
    i, n = 0, len(text)
    while i < n:
        # 引语区间内：整段即时产出为 1 子句（不等边界——引语后叙述才能独立）
        in_q = False
        for s, e in quote_ranges:
            if s <= i < e:
                in_q = True
                break
        if in_q:
            if cur.strip():
                subs.append(cur.strip())
                cur = ""
            for s, e in quote_ranges:
                if s <= i < e:
                    sub = text[i:e].strip()
                    if len(sub) >= 2:
                        subs.append(sub)
                    i = e
                    break
            continue
        ch = text[i]
        if ch in SUB_BOUNDARY:
            if cur.strip():
                subs.append(cur.strip())
            cur = ""
        elif ch in _SENT_END:
            if cur.strip():
                subs.append(cur.strip())
            cur = ""
        else:
            cur += ch
        i += 1
    if cur.strip():
        subs.append(cur.strip())
    return subs


def subclause_labels(text, parent=None):
    """子句 → 标签 dict（复用 _rule_label 重算；引语子句强制 FUNC=D + 继承父 COG/POL）
    parent: 父句元 labels dict（继承 COG/POL——对话本质=信息交换，子句级规则不区分）"""
    is_quote = bool(QUOTE_RE.search(text))
    labels, conf, has_action = _rule_label(text, False)
    if is_quote:
        labels["FUNC"] = "D"
        conf["FUNC"] = 0.9
        if parent is not None:
            labels["COG"] = parent.get("COG", 0)
            labels["POL"] = parent.get("POL", 0)
    return {"text": text, "labels": labels, "conf": conf,
            "has_action": has_action, "is_quote": is_quote}


def attach_subclauses(items):
    """句元 items → 每句元挂 sc 字段（子句列表——纯函数派生，不落盘缓存）
    items: annotate 输出（每项含 text/labels/para_break…）——原地派生"""
    for it in items:
        sc = []
        for sub in split_subclauses(it["text"]):
            sc.append(subclause_labels(sub, it.get("labels")))
        it["sc"] = sc
    return items


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    # 断言（计划步骤 1 验证）
    cases = [
        ("沈彻靠着墙角，数到第七十三滴的时候，门开了。",
         ["沈彻靠着墙角", "数到第七十三滴的时候", "门开了"]),
        ("沈彻开口，嗓子干涩，“你来了。”",
         ["沈彻开口", "嗓子干涩", "“你来了。”"]),
        ("“沈彻。”", ["“沈彻。”"]),
        ("油灯先探进来，然后是狱卒的靴子，最后是那个人。",
         ["油灯先探进来", "然后是狱卒的靴子", "最后是那个人"]),
    ]
    ok = True
    for text, expect in cases:
        got = split_subclauses(text)
        status = "✓" if got == expect else "✗"
        if got != expect:
            ok = False
        print(f"{status} 『{text[:24]}』 → {got}")
    # 引语内逗号不拆 + 引语后叙述独立
    t = "“沈彻，”他开口，嗓子干涩，“你来了。”"
    got = split_subclauses(t)
    ok = ok and got == ["“沈彻，”", "他开口", "嗓子干涩", "“你来了。”"]
    print(f"{'✓' if got == ['“沈彻，”','他开口','嗓子干涩','“你来了。”'] else '✗'} 引语内逗号不拆: {got}")
    print("ALL PASS" if ok else "FAIL")
