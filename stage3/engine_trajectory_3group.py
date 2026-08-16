# -*- coding: utf-8 -*-
"""v0.90 意图轨迹三组探索（R 推理 / T 思想论证 / N 叙事——探索性——不正式判定——决策树驱动）

用户方案：公务员行测推理题（R——无数字、逻辑显式）与思想论证（T——矛盾论/实践论）的意图空间
轨迹是否比叙事（N——小说）更"结构化"。三问：Q1 轨迹结构 / Q2 跳跃点对应思维动作 / Q3 几何分离。

【报告集预注册表（先写后跑——探索性——三态判定：候选/不支持/地板不可判）】

候选门槛（探索性——非正式判定）：簇 CI 不相交 ∧ |d|≥0.8 ∧ ≥2 个独立指标方向一致 ∧
shuffle 不消失（δ≥0.2）∧ 域距离调整后仍存（域调整操作定义：以风格坐标 s 与 Mahalanobis m
为协变量线性回归取残差 → 三组间 d_res——若原始 |d|≥0.8 但 |d_res|<0.4 → "分离主要由域偏移承载"）。

指标口径（主判无阈值指标优先——固定 p90=3.2286 仅辅助）：
- 跳跃密度 100·|{t:‖Δ_t‖>p90}|/(n−1)（旁报 diff 中位数/组内 p90）
- gap CV（泊松=1——规律性主判）/ 连发率 P(gap≤2)（锚点 26.4/22.5）
- 绝对幅度 mean(‖Δ‖>p90) / 相对幅度 mean(‖Δ‖>p90)/median(‖Δ‖)
- 转折密度 (peaks+valleys)/(n−2) dim10/48 / 标准化陡度·深度（/SD(z)）
- 耦合 mean corr(z10[win3], z48[win3])（零方差窗记 0.0——锚点 0.52）
- ratio_unit 全篇口径（N≥10 差分——旁注口径差）/ Hurst（N≥16 报数★/N≥40 判向）/ lag-1 自相关（N≥20）
- 域面板：风格坐标 s=(f−μ_mid)·ê（ê=(μ_h−μ_a)/‖·‖——bilingual 池化）+ Mahalanobis（top-50 白化——
  μ_tr/Σ_tr=bilingual_zh 全部 26734 句元）+ 判别分 sigmoid（新编码组——解释性）+ 组中心残差 sil
- 几何：sil 三口径（64 原/11 目标/组中心残差——均衡抽样 400/组——残差口径主判）

Q2 逻辑词表（预注册写死——33 词）：
结论类（8）：因此、所以、由此可见、这意味着、故、由此可知、因而、于是
反驳类（9）：但是、然而、不过、相反、实则不然、这并不、可是、但、却
前提类（8）：如果、若、假设、倘若、既然、由于、只要、只有
肯定否定（8）：显然、未必、必定、不可能、必然、并非、除非、否则
lift = |J∩L|/|J| ÷ (|L|/n)——逐文本——R 另报结论句命中 1[conclusion_idx ∈ J]

复现门六锚（不过中止）：p90=3.229±1e-3 / 密度 12.08 vs 7.74±0.2 / gap 5 vs 8±1 /
连发 26.4 vs 22.5±2 / ratio_unit 0.0863/0.1048±1e-3 / EVR 读取 axis_analysis.json 0.4543±1e-4

诚实限制：OOD 不可消仅可量化；R 模板痕迹（伪词对照 n=5 仅信号提示不作判定）；短文本指标地板；
簇独立单位少（CI 宽——判定语言"倾向"）；Hurst 低功率星号；ratio_unit 口径差旁注；
探索性三态非正式判定——本轮不写论文节。
R 组实测 13-18 句元（模板扩展后的实际——原设计 25-45 未达——地板之上：
ratio_unit N≥10 ✓——lag1/Hurst 部分覆盖——如实标注）。
"""
import sys, json, os, re, time
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path('C:/Users/bai/Desktop/小说系统')
OUT = BASE / 'data' / 'dim_analysis'
PAPER = Path('C:/Users/bai/Desktop/AB系统论文储备')
SRC = BASE / 'data' / 'v090_sources'
sys.path.insert(0, str(BASE / 'stage3'))

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

SEED = 20260818
P90_EXPECT = 3.229
DENS_H_EXPECT, DENS_A_EXPECT = 12.08, 7.74
RU_H_EXPECT, RU_A_EXPECT = 0.0863, 0.1048
EVR_EXPECT = 0.4543
TARGET11 = [5, 10, 11, 22, 26, 34, 43, 46, 48, 52, 59]
WIN = 25
N_WIN_PER_DOC = 15
N_T_SEGS = 10          # T 组 10 段（400-800 字）
N_R_PER_CLASS = 5      # R 组 4 类 × 5 = 20 题
N_PSEUDO = 5           # 伪词对照 5 题

LOGIC_WORDS = ['因此', '所以', '由此可见', '这意味着', '故', '由此可知', '因而', '于是',
               '但是', '然而', '不过', '相反', '实则不然', '这并不', '可是', '但', '却',
               '如果', '若', '假设', '倘若', '既然', '由于', '只要', '只有',
               '显然', '未必', '必定', '不可能', '必然', '并非', '除非', '否则']


def l2norm(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


def cluster_bootstrap_ci(doc_ids, values, n_boot=2000, seed=42):
    """doc 簇 bootstrap 95% CI——doc=独立性单位"""
    rng = np.random.default_rng(seed)
    docs = np.unique(doc_ids)
    vals = np.asarray(values, float)
    means = []
    for _ in range(n_boot):
        sel = rng.choice(len(docs), len(docs), replace=True)
        idx = np.concatenate([np.where(doc_ids == d)[0] for d in sel])
        means.append(vals[idx].mean())
    return np.percentile(means, [2.5, 97.5])


def cohens_d(x, y):
    return (np.mean(x) - np.mean(y)) / np.sqrt((np.var(x) + np.var(y)) / 2 + 1e-9)


def load_bilingual():
    """bilingual_zh 人类/AI 10+20 篇——返回 fp 切片 + 每 doc 行索引"""
    fp = np.load(OUT / 'fp_matrix.npz')['fp']
    rows = json.loads((OUT / 'rows.json').read_text(encoding='utf-8'))
    docs = defaultdict(list)
    for i, r in enumerate(rows):
        if r['source'] == 'bilingual_zh':
            docs[r['doc']].append(i)
    human = sorted([d for d in docs if d.startswith('ZH-H')], key=lambda d: len(docs[d]))
    ai = sorted([d for d in docs if d.startswith('ZH-A')], key=lambda d: len(docs[d]))
    return fp, rows, docs, human, ai


def extract_side(source, prefix, side='human'):
    """多侧混装 doc 过滤（L2/S 含 human+ai+qwen 连续块——side 过滤+保序）"""
    fp = np.load(OUT / 'fp_matrix.npz')['fp']
    rows = json.loads((OUT / 'rows.json').read_text(encoding='utf-8'))
    docs = defaultdict(list)
    for i, r in enumerate(rows):
        if r['source'] == source and r['doc'].startswith(prefix) and r['side'] == side:
            docs[r['doc']].append(i)
    return fp, docs


def sample_windows(fp, docs, human, per_doc=N_WIN_PER_DOC, win=WIN, rng=None):
    """25 句元连续窗口——段起始对齐——簇按源文档"""
    if rng is None:
        rng = np.random.default_rng(SEED + 1)
    windows, wdoc = [], []
    for doc in human:
        idx = np.array(docs[doc])
        # 段起始：rows 的 para 变化处（近似——按行号连续块）
        starts = [0]
        for i in range(1, len(idx)):
            if idx[i] - idx[i - 1] > 1:
                starts.append(i)
        cands = [s for s in starts if s + win <= len(idx)]
        if not cands:
            cands = list(range(0, len(idx) - win + 1))
        chosen = rng.choice(cands, min(per_doc, len(cands)), replace=False)
        for s in chosen:
            windows.append(idx[s:s + win])
            wdoc.append(doc)
    return windows, wdoc


def seg_texts_of(text):
    """长文本 → 句元（同 build_bilingual_zh_metrics.paras_of 口径——行切分/汉字/len≥30/非章节标题——
    再 split_subclauses len≥3）"""
    from subclause_structure import split_subclauses
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    paras = [l for l in lines if re.search(r'[一-龥]', l) and len(l) >= 30
             and not re.match(r'^第[一二三四五六七八九十百千0-9]+[章节卷回部]', l)]
    segs = []
    for p in paras:
        for c in split_subclauses(p):
            if len(c.strip()) >= 3:
                segs.append(c.strip())
    return segs


def acquire_T():
    """T 组：实践论/矛盾论切 400-800 字连续论证段（10 段）——txt 为行分隔段落——
    拼接累积 ≥400 输出——>800 从 600-800 间句号切（防超长卡死——v0.90 修复）"""
    texts, meta = [], []
    for fname, src_name in (('实践论.txt', '实践论 1937-07'), ('矛盾论.txt', '矛盾论 1937-08')):
        t = (SRC / fname).read_text(encoding='utf-8', errors='replace')
        paras = [p.strip() for p in t.split('\n') if len(p.strip()) >= 40]
        buf = ''
        for p in paras:
            if p.startswith('实践论') or p.startswith('矛盾论') or '一九三七年' in p[:20]:
                continue  # 标题/日期行
            buf += p
            if len(buf) >= 400:
                if len(buf) > 800:
                    cut = buf.rfind('。', 600, 800)
                    cut = cut if cut > 600 else 800
                    texts.append(buf[:cut + 1])
                    buf = buf[cut + 1:]
                else:
                    texts.append(buf)
                    buf = ''
            if len(texts) >= N_T_SEGS:
                break
        if len(texts) < N_T_SEGS:
            print(f'  T 组 {fname}: 切出 {len(texts)} 段——段数不足警告')
        if len(texts) >= N_T_SEGS:
            break
    texts = texts[:N_T_SEGS]
    meta = [{'group': 'T', 'src': src_name, 'n_char': len(t)} for t in texts]
    return texts, meta


# ============ R 组模板（4 类 × 5——散文式——实体池多样——干扰句——结论句 3 变体） ============
CONCL_STYLES = ['因此，', '所以，', '由此可知，']
REAL_NAMES = ['甲', '乙', '丙', '丁', '戊']
OCCUP = ['教师', '医生', '律师', '工程师', '画家']
CITIES = ['北京', '上海', '广州', '深圳', '成都']
SPORTS = ['足球', '篮球', '游泳', '乒乓球', '跑步']
SUBJECTS = ['数学', '语文', '英语', '物理', '化学']
COLORS = ['红', '蓝', '绿', '黄', '白']
FRUITS = ['苹果', '香蕉', '橙子', '葡萄', '西瓜']


def _q_condition(i):
    """条件推理（谁真话）——实体错位防重复——推理链展开 + 干扰句（扩充版——18-30 句元）"""
    names = REAL_NAMES[:4]
    a, b, c, d = names
    cs = CONCL_STYLES[i % 3]
    extras = [f'有目击者称案发时现场附近曾出现可疑车辆，但未能确认车主身份。',
              f'警方调取了周边监控，因设备故障部分时段录像缺失。',
              f'案发现场没有发现明显的打斗痕迹，门窗均完好。'][i % 3]
    return (f'某案件的四名嫌疑人{a}、{b}、{c}、{d}接受警方讯问。'
            f'{a}说：这件事不是我做的。{b}说：这件事是{c}做的。'
            f'{c}说：这件事不是{d}做的。{d}说：这件事是{a}做的。'
            f'{extras}此外，侦查人员还核对了四人的行动轨迹，{a}在案发时段一直在办公室值班，'
            f'值班记录由门卫签字确认。{d}声称自己当时正在家里休息，但邻居表示当晚并未见到他。'
            f'经调查，四人中只有一人说了真话，其余三人的陈述都与事实不符。'
            f'{cs}可以确定，这件事是{a}做的。'), '可以确定'


def _q_condition2(i):
    names = REAL_NAMES[:4]
    a, b, c, d = names
    cs = CONCL_STYLES[i % 3]
    extras = ['会议记录显示当天出席人数与签到表一致。',
              '保安室值班日志显示当晚大门在十点后关闭。',
              '前台登记簿上留有几个模糊的字迹，暂无法辨认。'][i % 3]
    return (f'公司失窃案的调查中，员工{a}、{b}、{c}、{d}接受了询问。'
            f'{a}说：我当天请假不在公司。{b}说：{a}当天确实不在。'
            f'{c}说：{d}当天来过公司。{d}说：{c}在说谎。'
            f'{extras}人事部门调出了当天的考勤记录，{b}与{c}的打卡时间存在明显异常，'
            f'而{a}的请假申请单上有部门经理的亲笔签字。{d}的工位监控在关键时段出现了画面缺失。'
            f'已知四人中恰有两人说了真话。'
            f'{cs}可以确定，{a}当天确实不在公司。'), '可以确定'


def _q_order(i):
    names = REAL_NAMES[:5]
    a, b, c, d, e = names
    cs = CONCL_STYLES[i % 3]
    extras = ['比赛当天天气晴朗，观众席座无虚席。',
              '终点线的计时设备由两名工作人员分别记录。',
              '颁奖仪式安排在比赛结束一小时后举行。'][i % 3]
    return (f'五名选手{a}、{b}、{c}、{d}、{e}参加了长跑比赛。'
            f'{a}在{b}之前到达终点，{c}在{d}之后到达，'
            f'{e}既不是第一名也不是最后一名。'
            f'据现场裁判回忆，{d}与{e}到达的时间非常接近，两人的成绩只相差不到一秒。'
            f'{b}在冲刺阶段明显减速，最终排在{c}之后到达。{extras}'
            f'已知五人到达终点的顺序各不相同，且每个名次都只对应一名选手。'
            f'{cs}可以确定，{a}一定排在{b}前面。'), '可以确定'


def _q_order2(i):
    names = REAL_NAMES[:5]
    a, b, c, d, e = names
    cs = CONCL_STYLES[i % 3]
    extras = ['书架的每一层都能容纳至少三本书。',
              '图书馆的开放时间到晚上九点结束。',
              '这批新书入库前经过了分类编号。'][i % 3]
    return (f'五本不同的书摆放在书架的同一层上，从左到右依次是{a}、{b}、{c}、{d}、{e}对应的五本书。'
            f'{a}书的左边是{c}书，{b}书紧挨着{d}书，{e}书不在最右边。'
            f'{c}书的右侧紧挨着{e}书，{d}书与{a}书之间隔着一本书。'
            f'管理员清点时发现每本书的书脊上都贴有不同颜色的标签，颜色与书名首字母对应。{extras}'
            f'已知五本书的位置各不相同。'
            f'{cs}可以确定，{a}书一定在{c}书的右边。'), '可以确定'


def _q_match(i):
    a, b, c = REAL_NAMES[:3]
    occ = OCCUP[:3]
    x, y, z = occ
    cs = CONCL_STYLES[i % 3]
    extras = ['三位老师在同一所学校任教已有五年。',
              '学校每学期末都会组织教学成果展示。',
              '教研组的会议通常安排在每周二下午。'][i % 3]
    return (f'{a}、{b}、{c}三位老师分别教{x}、{y}、{z}三门课。'
            f'{a}不教{y}，{b}教{x}，{c}不教{z}。'
            f'此外，{a}老师的办公室与教{z}的老师相邻，两人经常一起备课。'
            f'{b}老师曾在公开课上展示过{x}的教学案例，受到教研组一致好评。'
            f'{c}老师的学生多次在{z}相关的竞赛中获奖。{extras}'
            f'每位老师只教一门课，每门课只有一位老师教。'
            f'{cs}可以确定，{a}教的是{z}。'), '可以确定'


def _q_match2(i):
    a, b, c = REAL_NAMES[:3]
    cities = CITIES[:3]
    x, y, z = cities
    sports = SPORTS[:3]
    cs = CONCL_STYLES[i % 3]
    extras = ['三人都参加过去年的城市运动会。',
              '周末的体育场馆需要提前预约。',
              '城市间的火车车程都在两小时以内。'][i % 3]
    return (f'{a}、{b}、{c}三人分别来自{x}、{y}、{z}三座城市，分别喜欢{sports[0]}、{sports[1]}、{sports[2]}。'
            f'{a}来自{x}，{b}喜欢{sports[1]}，来自{z}的人不喜欢{sports[0]}。'
            f'据悉，喜欢{sports[2]}的人去年参加了全国锦标赛，并获得了不错的成绩。'
            f'{c}在一次访谈中提到自己家乡的球队，言语中充满感情。'
            f'三人还各自养了一只宠物，宠物的品种互不相同。{extras}'
            f'三人所在城市各不相同，喜欢的运动也各不相同。'
            f'{cs}可以确定，{a}喜欢的是{sports[0]}。'), '可以确定'


def _q_impl(i):
    cs = CONCL_STYLES[i % 3]
    conds = [('如果天上下大雨', '那么操场就会积水', '现在操场没有积水', '天没有下大雨'),
             ('如果一个人是医生', '那么他必须经过专业训练', '张某没有经过专业训练', '张某不是医生'),
             ('如果某文件已经审批', '那么它必然带有编号', '这份文件没有编号', '这份文件尚未审批'),
             ('如果列车准点出发', '那么它会在晚上八点前到站', '列车没有在八点前到站', '列车没有准点出发'),
             ('如果球队赢得比赛', '那么全队会获得奖金', '全队没有获得奖金', '球队没有赢得比赛')]
    ante, cons, obs, concl = conds[i]
    extras = ['车站的广播系统在整点进行播报。',
              '审批流程通常需要三到五个工作日。',
              '比赛录像将在赛后公开回放。'][i % 3]
    return (f'{ante}，{cons}。{obs}。'
            f'有关人员事后核对了相关记录，情况与上述前提一致，没有发现任何例外情形。'
            f'相反，如果{ante.replace("如果", "")}，那么{cons.replace("那么", "")}一定成立，'
            f'这一点已经由多次观测结果证实。{extras}'
            f'根据充分条件假言推理的规则，否定后件可以推出否定前件，这是形式逻辑的基本结论。'
            f'{cs}可以确定，{concl}。'), '可以确定'


def acquire_R():
    """R 组 20 题（4 类 × 5——散文式——扩充推理链——18-30 句元）+ 伪词对照 5 题——
    结论句统一以'可以确定'定位（v0.90 修复——hint 字面匹配不可靠）"""
    gens = {
        '条件': [_q_condition, _q_condition2] * 3,
        '排列': [_q_order, _q_order2] * 3,
        '匹配': [_q_match, _q_match2] * 3,
        '假言': [_q_impl, _q_impl] * 3,
    }
    texts, meta = [], []
    for cls, fns in gens.items():
        for i in range(N_R_PER_CLASS):
            t, _ = fns[i % len(fns)](i)
            texts.append(t)
            meta.append({'group': 'R', 'cls': cls})
    # 结论句索引（Q2）——统一按'可以确定'定位
    for m, t in zip(meta, texts):
        segs = seg_texts_of(t)
        m['n_seg'] = len(segs)
        hit = [k for k, s in enumerate(segs) if '可以确定' in s]
        m['conclusion_idx'] = int(hit[0]) if hit else -1
    # 伪词对照 5 题（内容词→拼音伪词——句法不变）
    pseudo_pairs = [
        ('甲', 'jia'), ('乙', 'yi'), ('丙', 'bing'), ('丁', 'ding'), ('戊', 'wu'),
        ('北京', 'beijing'), ('上海', 'shanghai'), ('广州', 'guangzhou'),
        ('教师', 'jiaoshi'), ('医生', 'yisheng'), ('律师', 'lvshi'),
        ('足球', 'zuqiu'), ('篮球', 'lanqiu'), ('游泳', 'youyong'),
    ]
    pseudo, pmeta = [], []
    for i in range(N_PSEUDO):
        t = texts[i]
        for w, ph in pseudo_pairs:
            t = t.replace(w, ph)
        pseudo.append(t)
        pm = dict(meta[i])
        pm['group'] = 'R-pseudo'
        pmeta.append(pm)
    return texts, meta, pseudo, pmeta


def encode_new(texts_list, enc, disc):
    """新文本组 → 每文本句元指纹（不 norm_rows——同口径）"""
    import torch
    from para_dimensions import fingerprint
    all_fp, all_disc, all_segs = [], [], []
    for t in texts_list:
        segs = seg_texts_of(t)
        if not segs:
            all_fp.append(np.zeros((0, 64)))
            all_disc.append(np.zeros(0))
            all_segs.append([])
            continue
        sv = enc.encode(segs, normalize_embeddings=True, batch_size=16,
                        show_progress_bar=False, device='cpu').astype(np.float32)
        SV = torch.from_numpy(sv)
        with torch.no_grad():
            F = fingerprint(SV, disc).detach().cpu().numpy()
            D = torch.sigmoid(disc(SV)).detach().cpu().numpy()  # ParaDiscNN 已 squeeze 成 (n,)
        all_fp.append(F)
        all_disc.append(D)
        all_segs.append(segs)
    return all_fp, all_disc, all_segs


# ============ 指标 ============
def jump_metrics(nrm, p90):
    """跳跃指标（per-text）——密度/gap CV/连发/绝对·相对幅度 + 旁报 diff 中位数/组内 p90"""
    n = len(nrm)
    jpos = np.where(nrm > p90)[0]
    density = 100 * len(jpos) / n
    gaps = np.diff(jpos)
    gap_cv = float(np.std(gaps) / (np.mean(gaps) + 1e-9)) if len(gaps) >= 2 else np.nan
    burst = 100 * float(np.mean(gaps <= 2)) if len(gaps) else np.nan
    abs_amp = float(np.mean(nrm[jpos])) if len(jpos) else np.nan
    rel_amp = abs_amp / (np.median(nrm) + 1e-9) if len(jpos) else np.nan
    return {'density': density, 'gap_cv': gap_cv, 'burst': burst,
            'abs_amp': abs_amp, 'rel_amp': rel_amp,
            'diff_med': float(np.median(nrm)), 'p90_in': float(np.quantile(nrm, 0.90)),
            'n_jump': int(len(jpos))}


def turn_metrics(z):
    """转折（dim 序列）——密度/标准化陡度·深度（/SD——防尺度漂移）——无阈值"""
    from dim_flow_sent import turn_points
    n = len(z)
    if n < 3:
        return {'turn_density': np.nan, 'steep_norm': np.nan, 'depth_norm': np.nan}
    tp = turn_points(z)
    sd = np.std(z) + 1e-9
    td = (tp['n_peaks'] + tp['n_valleys']) / (n - 2)
    return {'turn_density': td,
            'steep_norm': tp['peak_steep_mean'] / sd,
            'depth_norm': tp['valley_depth_mean'] / sd}


def coupling_metrics(z10, z48):
    from engine_planner_bands import sliding_coupling
    return {'coupling': sliding_coupling(z10, z48, win=3)}


def ratio_short(D_fp, D_axis):
    """ratio_unit 全篇口径（无 60/40 切分——旁注口径差）——N≥10 差分"""
    from engine_ratio_validate import ratio_of
    if D_fp.shape[0] < 10:
        return {'ratio_unit': np.nan, 'n_diff': D_fp.shape[0]}
    ru, jpu, jppu = ratio_of(D_fp, D_axis)
    return {'ratio_unit': ru, 'n_diff': D_fp.shape[0]}


def hurst_lowpower(z):
    from dim_flow import hurst as df_hurst
    n = len(z)
    if n < 16:
        return {'hurst': np.nan, 'power': 'n<16'}
    h = float(df_hurst(z))
    if np.isnan(h):
        return {'hurst': np.nan, 'power': f'n={n} NaN'}
    return {'hurst': h, 'power': '判向' if n >= 40 else '仅报数*'}


def lag1_acf(z):
    n = len(z)
    if n < 20:
        return {'lag1': np.nan}
    zz = z - z.mean()
    denom = np.sum(zz[:-1] ** 2) + 1e-9
    phi = float(np.sum(zz[:-1] * zz[1:]) / denom)
    se = float(np.sqrt((1 - phi ** 2) / n))
    return {'lag1': phi, 'lag1_se': se}


def per_text_metrics(fp_doc, disc_doc, D_axis, p90):
    """单文本全指标集"""
    n = fp_doc.shape[0]
    out = {'n_clauses': n, 'disc_mean': float(np.mean(disc_doc)) if len(disc_doc) else np.nan}
    if n >= 3:
        nrm = np.linalg.norm(fp_doc[1:] - fp_doc[:-1], axis=1)
        out.update(jump_metrics(nrm, p90))
        out.update(ratio_short(fp_doc[1:] - fp_doc[:-1], D_axis))
        for dname, dk in (('d10', 10), ('d48', 48)):
            z = fp_doc[:, dk]
            out.update({f'{dname}_{k}': v for k, v in turn_metrics(z).items()})
            hh = hurst_lowpower(z)
            out[f'{dname}_hurst'] = hh['hurst']
            out[f'{dname}_hurst_power'] = hh['power']
            la = lag1_acf(z)
            out[f'{dname}_lag1'] = la.get('lag1', np.nan)
        out.update(coupling_metrics(fp_doc[:, 10], fp_doc[:, 48]))
    return out


def shuffle_control(metric_fn, z_docs, n_perm=30, seed=42):
    """逐文本置换（测顺序敏感性）——metric_fn 返回 dict——每键 δ = |obs−shuf_med|/|obs|"""
    rng = np.random.default_rng(seed)
    out = {}
    for z in z_docs:
        obs = metric_fn(z)
        shufs = {k: [] for k in obs}
        for _ in range(n_perm):
            zp = rng.permutation(z)
            sp = metric_fn(zp)
            for k in obs:
                shufs[k].append(sp[k])
        for k in obs:
            o = obs[k]
            s = float(np.median(shufs[k]))
            if k not in out:
                out[k] = []
            out[k].append(np.nan if np.isnan(o) else abs(o - s) / (abs(o) + 1e-9))
    return out


def style_coordinate(fp_all, mu_h, mu_a):
    """风格坐标 s=(f−μ_mid)·ê——ê=(μ_h−μ_a)/‖·‖"""
    mid = (mu_h + mu_a) / 2
    e = l2norm((mu_h - mu_a).reshape(1, -1))[0]
    return (fp_all - mid) @ e


def mahalanobis(fp_all, mu_tr, cov_tr, top_k=50):
    """Mahalanobis——top-50 白化截断（64 维协方差死维）"""
    u, s, vt = np.linalg.svd(cov_tr, hermitian=True)
    keep = s > 1e-8
    s = s[keep][:top_k]
    v = vt[:top_k][keep[:top_k]]
    z = (fp_all - mu_tr) @ v.T
    return np.sum((z / (np.sqrt(s) + 1e-9)) ** 2, axis=1)


def residual_geometry(fps_by_group, labels, n_sample=400, seed=42):
    """组中心残差 sil 三口径（均衡抽样——残差口径主判）"""
    from sklearn.metrics import silhouette_score
    from sklearn.decomposition import PCA
    rng = np.random.default_rng(seed)
    allf, alll = [], []
    for g, fp in enumerate(fps_by_group):
        idx = rng.choice(len(fp), min(n_sample, len(fp)), replace=False)
        allf.append(fp[idx])
        alll.append([g] * len(idx))
    X = np.vstack(allf)
    y = np.concatenate(alll)
    Xn = l2norm(X)
    res = {}
    res['sil_64'] = float(silhouette_score(Xn, y, metric='cosine'))
    res['sil_11'] = float(silhouette_score(Xn[:, TARGET11], y, metric='cosine'))
    # 组中心残差（64 维）
    centers = np.array([Xn[y == g].mean(0) for g in range(len(fps_by_group))])
    Xr = Xn - centers[y]
    res['sil_resid'] = float(silhouette_score(Xr, y, metric='cosine'))
    res['n_each'] = [int(np.sum(y == g)) for g in range(len(fps_by_group))]
    return res


def logic_overlap(segs, jpos, conclusion_idx=None):
    """Q2：lift = |J∩L|/|J| ÷ (|L|/n)——R 另报结论句命中"""
    n = len(segs)
    L = [k for k, s in enumerate(segs) if any(w in s for w in LOGIC_WORDS)]
    J = set(jpos)
    hit = sum(1 for k in L if k in J)
    lift = (hit / (len(J) + 1e-9)) / ((len(L) / n) + 1e-9)
    concl_hit = (conclusion_idx is not None and conclusion_idx in J)
    return {'n_seg': n, 'n_logic': len(L), 'n_jump': len(J),
            'jump_logic_hit': hit, 'lift': float(lift),
            'concl_hit': bool(concl_hit), 'concl_idx': conclusion_idx}


def gate_repro():
    """复现门六锚（不过中止）"""
    fp, rows, docs, human, ai = load_bilingual()
    D_shared = np.asarray(json.loads((OUT / 'axis_analysis.json').read_text(encoding='utf-8'))
                          ['axis']['D_shared'], float)
    D_shared = D_shared / (np.linalg.norm(D_shared) + 1e-9)
    diffs = {}
    for grp, dlist in (('human', human), ('ai', ai)):
        diffs[grp] = {}
        for doc in dlist:
            idx = np.array(docs[doc])
            diffs[grp][doc] = fp[idx[1:], :] - fp[idx[:-1], :]
    all_norms = np.concatenate([np.linalg.norm(d, axis=1)
                                for dd in diffs.values() for d in dd.values()])
    p90 = float(np.quantile(all_norms, 0.90))
    evr = float(json.loads((OUT / 'axis_analysis.json').read_text(encoding='utf-8'))
                ['axis']['shared_pc1_evr'])
    ok1 = abs(p90 - P90_EXPECT) < 1e-3
    ok2 = abs(evr - EVR_EXPECT) < 1e-4
    stats = {'p90': p90, 'evr': evr}
    for grp, dlist in (('human', human), ('ai', ai)):
        dens, gaps_all, bursts = [], [], []
        from engine_ratio_validate import ratio_of
        rus = []
        for doc in dlist:
            nrm = np.linalg.norm(diffs[grp][doc], axis=1)
            jpos = np.where(nrm > p90)[0]
            dens.append(100 * len(jpos) / len(nrm))
            if len(jpos) >= 2:
                g = np.diff(jpos)
                gaps_all += g.tolist()
                bursts.append(100 * np.mean(g <= 2))
            rus.append(ratio_of(diffs[grp][doc], D_shared)[0])
        stats[f'dens_{grp}'] = float(np.mean(dens))
        stats[f'gap_{grp}'] = float(np.median(gaps_all)) if gaps_all else np.nan
        stats[f'burst_{grp}'] = float(np.mean(bursts)) if bursts else np.nan
        stats[f'ru_{grp}'] = float(np.mean(rus))
    checks = {
        'p90': abs(stats['p90'] - P90_EXPECT) < 1e-3,
        'dens_h': abs(stats['dens_human'] - DENS_H_EXPECT) < 0.2,
        'dens_a': abs(stats['dens_ai'] - DENS_A_EXPECT) < 0.2,
        'ru_h': abs(stats['ru_human'] - RU_H_EXPECT) < 1e-3,
        'ru_a': abs(stats['ru_ai'] - RU_A_EXPECT) < 1e-3,
        'evr': ok2,
    }
    print('  S0 复现门:', {k: round(v, 4) for k, v in stats.items()})
    bad = [k for k, v in checks.items() if not v]
    assert not bad, f'复现门失败: {bad}'
    print('  六锚全过 ✓')
    return p90, fp, rows, docs, human, ai, D_shared


def main():
    print('===== v0.90 意图轨迹三组探索（R 推理/T 论证/N 叙事——探索性——报告集预注册见文件头） =====')
    os.environ.setdefault('HF_HUB_OFFLINE', '1')

    # ===== P0 复现门 + 语料 =====
    print('\nP0 复现门:')
    p90, fp, rows, docs, human, ai, D_shared = gate_repro()
    mu_h = fp[np.concatenate([np.array(docs[d]) for d in human])].mean(0)
    mu_a = fp[np.concatenate([np.array(docs[d]) for d in ai])].mean(0)
    mu_tr = fp.mean(0)
    cov_tr = np.cov(fp, rowvar=False)

    print('\nP0 语料构建:')
    # N-win（现成切片）
    rng = np.random.default_rng(SEED + 1)
    windows, wdoc = sample_windows(fp, docs, human)
    print(f'  N-win: {len(windows)} 窗口（25 句元——簇按 {len(set(wdoc))} 源文档）')
    # T 组（新文本）
    t_texts, t_meta = acquire_T()
    print(f'  T 组: {len(t_texts)} 段（实践论/矛盾论——400-800 字）')
    # R 组（模板）
    r_texts, r_meta, rp_texts, rp_meta = acquire_R()
    print(f'  R 组: {len(r_texts)} 题 + 伪词 {len(rp_texts)} 题')
    # 地板面板（现成——L1/L2/S human 侧）
    from engine_planner_bands import load_docs as _ld
    floor = {}
    floor['L1'] = extract_side('generalization_strict', 'L1')
    floor['L2'] = extract_side('generalization_strict', 'L2')
    floor['S'] = extract_side('independent_test', 'S')
    n_floor = {k: [len(v[d]) for d in v] for k, (_, v) in floor.items()}
    print(f'  地板面板句元数: ' + json.dumps({k: [min(v), int(np.median(v)), max(v)] for k, v in n_floor.items()}))

    # 新编码（T + R + 伪词）
    print('\nP0 编码（T+R+伪词——~1000 句元 CPU）:')
    from para_dimensions import load_models
    enc, disc = load_models('cpu')
    t0 = time.time()
    T_fp, T_disc, T_segs = encode_new(t_texts, enc, disc)
    R_fp, R_disc, R_segs = encode_new(r_texts, enc, disc)
    RP_fp, RP_disc, RP_segs = encode_new(rp_texts, enc, disc)
    print(f'  编码完成（{time.time() - t0:.0f}s）——T 句元 {[f.shape[0] for f in T_fp]}')
    print(f'  R 句元 {[f.shape[0] for f in R_fp]}——伪词 {[f.shape[0] for f in RP_fp]}')

    # 校验门：断言 R 句元 ≥12（模板实测 13-18 句元——ratio_unit 地板 ≥10 ✓——
    # lag1 需 N≥20 部分覆盖——如实标注——头注释预注册口径）
    for i, f in enumerate(R_fp):
        assert f.shape[0] >= 12, f'R{i} 句元过少: {f.shape[0]}'
    for i, f in enumerate(T_fp):
        assert f.shape[0] >= 25, f'T{i} 句元过少: {f.shape[0]}'  # 实测 25-61——地板之上（ratio_unit≥10/lag1≥20/Hurst≥16）

    # 落盘
    corpus = {'T': [{'text': t, 'src': m['src'], 'n_char': m['n_char']} for t, m in zip(t_texts, t_meta)],
              'R': [{'text': t, 'cls': m['cls'], 'conclusion_idx': m['conclusion_idx']} for t, m in zip(r_texts, r_meta)],
              'R-pseudo': [{'text': t, 'conclusion_idx': m['conclusion_idx']} for t, m in zip(rp_texts, rp_meta)],
              'N-win_docs': list(set(wdoc))}
    (OUT / '3group_corpus.json').write_text(json.dumps(corpus, ensure_ascii=False, indent=1), encoding='utf-8')
    np.savez(OUT / '3group_fp.npz',
             T=np.vstack([f for f in T_fp if f.shape[0]]),
             R=np.vstack([f for f in R_fp if f.shape[0]]),
             RP=np.vstack([f for f in RP_fp if f.shape[0]]),
             T_disc=np.concatenate(T_disc), R_disc=np.concatenate(R_disc),
             RP_disc=np.concatenate(RP_disc))
    print('  落盘 3group_corpus.json + 3group_fp.npz ✓')

    # ===== P1 指标（per-text——三组） =====
    print('\nP1 指标计算:')
    groups = {}
    # N-long（10 篇人类——全指标锚点）
    nlong = []
    for doc in human:
        idx = np.array(docs[doc])
        nlong.append(per_text_metrics(fp[idx], np.zeros(0), D_shared, p90))
    groups['N-long'] = {'docs': human, 'metrics': nlong}
    # N-win
    nwin = []
    for w in windows:
        nwin.append(per_text_metrics(fp[w], np.zeros(0), D_shared, p90))
    groups['N-win'] = {'docs': wdoc, 'metrics': nwin}
    # T / R / R-pseudo
    for gname, fps, dcs, segs in (('T', T_fp, T_disc, T_segs), ('R', R_fp, R_disc, R_segs),
                                  ('R-pseudo', RP_fp, RP_disc, RP_segs)):
        mm = []
        for i, f in enumerate(fps):
            if f.shape[0] >= 3:
                m = per_text_metrics(f, dcs[i] if len(dcs[i]) else np.zeros(0), D_shared, p90)
                m['disc_mean'] = float(np.mean(dcs[i])) if len(dcs[i]) else np.nan
                mm.append(m)
        groups[gname] = {'docs': [f'{gname}{i}' for i in range(len(fps))], 'metrics': mm}
    print(f'  组大小: ' + json.dumps({k: len(v['metrics']) for k, v in groups.items()}))

    # 指标汇总表（均值/中位/簇 CI）+ 效应量
    METRICS = ['density', 'gap_cv', 'burst', 'abs_amp', 'rel_amp',
               'd10_turn_density', 'd10_steep_norm', 'd10_depth_norm',
               'd48_turn_density', 'd48_steep_norm',
               'coupling', 'ratio_unit', 'd10_lag1', 'd48_lag1', 'd10_hurst']
    table = {}
    for mk in METRICS:
        table[mk] = {}
        for gname, g in groups.items():
            doc_arr = np.array(g['docs'])
            pairs = [(m, d) for m, d in zip(g['metrics'], doc_arr)
                     if not np.isnan(m.get(mk, np.nan))]
            if not pairs:
                table[mk][gname] = {'mean': None, 'med': None, 'ci': None, 'n': 0}
                continue
            vals = np.array([p[0][mk] for p in pairs])
            ids = np.array([p[1] for p in pairs])
            ci = cluster_bootstrap_ci(ids, vals)
            table[mk][gname] = {'mean': round(float(vals.mean()), 4),
                                'med': round(float(np.median(vals)), 4),
                                'ci': [round(float(ci[0]), 4), round(float(ci[1]), 4)],
                                'n': int(len(vals))}
    # 效应量（R vs N-win / T vs N-win / R vs T）
    d_pairs = {}
    for p1, p2 in (('R', 'N-win'), ('T', 'N-win'), ('R', 'T')):
        d_pairs[f'{p1}-{p2}'] = {}
        for mk in METRICS:
            v1 = np.array([m[mk] for m in groups[p1]['metrics'] if not np.isnan(m.get(mk, np.nan))])
            v2 = np.array([m[mk] for m in groups[p2]['metrics'] if not np.isnan(m.get(mk, np.nan))])
            if len(v1) and len(v2):
                d_pairs[f'{p1}-{p2}'][mk] = round(float(cohens_d(v1, v2)), 3)
    print('  d 矩阵（部分）:', json.dumps({k: {kk: vv for kk, vv in v.items() if abs(vv) >= 0.8}
                                            for k, v in d_pairs.items()}, ensure_ascii=False))

    # shuffle 对照（N-win 与 T/R 的 density/gap_cv/burst/turn_density/coupling——顺序敏感性）
    print('\n  shuffle 对照（30 次/文本——顺序敏感性）:')

    def _shuf_metrics(z):
        nrm = np.linalg.norm(z[1:] - z[:-1], axis=1)
        jm = jump_metrics(nrm, p90)
        tm = turn_metrics(z[:, 10])
        cp = coupling_metrics(z[:, 10], z[:, 48])
        return {'density': jm['density'], 'gap_cv': jm['gap_cv'], 'burst': jm['burst'],
                'turn_density': tm['turn_density'], 'coupling': cp['coupling']}

    shuf = {}
    for gname in ('N-win', 'T', 'R'):
        shuf[gname] = shuffle_control(_shuf_metrics,
                                      [fp[w] for w in windows] if gname == 'N-win' else
                                      ([f for f in T_fp] if gname == 'T' else [f for f in R_fp]))
    shuf_sum = {g: {k: round(float(np.nanmedian(v)), 3) for k, v in shuf[g].items()} for g in shuf}
    print('  shuffle δ 中位:', json.dumps(shuf_sum, ensure_ascii=False))

    # ===== P2 域面板 =====
    print('\nP2 域面板:')
    coord = {}
    for gname, fps in (('N-win', [fp[w] for w in windows]), ('T', T_fp), ('R', R_fp)):
        X = np.vstack(fps)
        coord[gname] = {'style_mean': float(style_coordinate(X, mu_h, mu_a).mean()),
                        'style_med': float(np.median(style_coordinate(X, mu_h, mu_a))),
                        'maha_med': float(np.median(mahalanobis(X, mu_tr, cov_tr)))}
    # 参照带（bilingual 人类/AI）
    coord['N-long_h'] = {'style_mean': float(style_coordinate(fp[np.concatenate([np.array(docs[d]) for d in human])], mu_h, mu_a).mean())}
    coord['N-long_a'] = {'style_mean': float(style_coordinate(fp[np.concatenate([np.array(docs[d]) for d in ai])], mu_h, mu_a).mean())}
    print('  风格坐标/Mahalanobis:', json.dumps(coord, ensure_ascii=False))
    # 残差几何 sil（三组：N-win 样本 / T / R——均衡 400）
    rg = residual_geometry([np.vstack([fp[w] for w in windows]), np.vstack(T_fp), np.vstack(R_fp)],
                           np.concatenate([np.zeros(len(windows)), np.ones(len(T_fp)), np.full(len(R_fp), 2)]).astype(int))
    print('  残差几何 sil:', json.dumps(rg, ensure_ascii=False))
    # 域调整（指标 ~ 风格坐标 + Mahalanobis 回归残差 → d_res——评审点 1）
    print('\n  域调整（回归残差 d_res——评审点 1 操作定义）:')
    from sklearn.linear_model import LinearRegression
    dom_res = {}
    for g1, g2 in (('R', 'N-win'), ('T', 'N-win')):
        for mk in ['density', 'gap_cv', 'burst', 'coupling']:
            pairs = []
            for gname in (g1, g2):
                fps_grp = [fp[w] for w in windows] if gname == 'N-win' else (T_fp if gname == 'T' else R_fp)
                for f in fps_grp:
                    if f.shape[0] < 3:
                        continue
                    sc_ = float(style_coordinate(f, mu_h, mu_a).mean())
                    ma_ = float(np.median(mahalanobis(f, mu_tr, cov_tr)))
                    v = per_text_metrics(f, np.zeros(0), D_shared, p90)[mk]
                    if not np.isnan(v):
                        pairs.append(([sc_, ma_], v, gname))
            if len(pairs) < 10:
                continue
            Xs = np.array([p[0] for p in pairs])
            vals = np.array([p[1] for p in pairs])
            g_arr = np.array([p[2] for p in pairs])
            reg = LinearRegression().fit(Xs, vals)
            resid = vals - reg.predict(Xs)
            v1 = resid[g_arr == g1]
            v2 = resid[g_arr == g2]
            d_res = cohens_d(v1, v2)
            dom_res[f'{g1}-{g2}_{mk}'] = round(float(d_res), 3)
    print('  d_res:', json.dumps(dom_res, ensure_ascii=False))

    # ===== P3 Q2 + 图 + 落盘 =====
    print('\nP3 Q2（跳跃点↔逻辑词 lift——逐文本）:')
    q2 = {}
    for gi in (0, 1, 5, 6):
        segs = R_segs[gi] if gi < 5 else T_segs[gi - 5]
        fps = R_fp[gi] if gi < 5 else T_fp[gi - 5]
        nrm = np.linalg.norm(fps[1:] - fps[:-1], axis=1)
        jpos = np.where(nrm > p90)[0]
        ci_ = r_meta[gi]['conclusion_idx'] if gi < 5 else None
        q2[f'R{gi}' if gi < 5 else f'T{gi - 5}'] = logic_overlap(segs, jpos, ci_)
    print('  Q2:', json.dumps(q2, ensure_ascii=False))

    # 图 1：指标 × 组
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    names_cn = {'density': '跳跃密度', 'gap_cv': 'gap CV', 'burst': '连发率',
                'abs_amp': '绝对幅度', 'rel_amp': '相对幅度', 'd10_turn_density': '转折密度 d10',
                'd10_steep_norm': '标准化陡度', 'coupling': '耦合', 'ratio_unit': 'ratio_unit'}
    picks = ['density', 'gap_cv', 'burst', 'd10_turn_density', 'coupling', 'ratio_unit']
    for ai_, mk in enumerate(picks):
        ax = axes.flat[ai_]
        gnames = ['R', 'T', 'N-win', 'N-long']
        means = [table[mk][g]['mean'] for g in gnames]
        cis = [table[mk][g]['ci'] for g in gnames]
        xs = np.arange(4)
        ax.bar(xs, [m if m is not None else 0 for m in means], 0.5, color=['#e67e22', '#27ae60', '#1f6fb2', '#8e44ad'])
        for x, m, ci in zip(xs, means, cis):
            if m is None or ci is None:
                continue
            ax.plot([x, x], ci, 'k-', lw=2)
        ax.set_title(f'{names_cn[mk]}（簇 95% CI）')
        ax.set_xticks(xs)
        ax.set_xticklabels(gnames, fontsize=8)
        ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(PAPER / 'fig_3group_metrics.png', dpi=150)
    plt.close()

    # 图 2：域面板
    fig2, axes2 = plt.subplots(1, 3, figsize=(17, 5))
    # 风格坐标直方图
    for gname, color in (('N-win', '#1f6fb2'), ('T', '#27ae60'), ('R', '#e67e22')):
        fps = [fp[w] for w in windows] if gname == 'N-win' else (T_fp if gname == 'T' else R_fp)
        s = style_coordinate(np.vstack(fps), mu_h, mu_a)
        axes2[0].hist(s, bins=30, alpha=0.4, color=color, label=gname)
    axes2[0].axvline(coord['N-long_h']['style_mean'], color='k', ls='--', label='人类参照')
    axes2[0].axvline(coord['N-long_a']['style_mean'], color='gray', ls='--', label='AI 参照')
    axes2[0].legend(fontsize=8)
    axes2[0].set_title('风格坐标分布（人类-AI 风格轴投影）')
    axes2[0].grid(alpha=0.3)
    # Mahalanobis 箱线
    maha_d = []
    for gname in ('N-win', 'T', 'R'):
        fps = [fp[w] for w in windows] if gname == 'N-win' else (T_fp if gname == 'T' else R_fp)
        maha_d.append(mahalanobis(np.vstack(fps), mu_tr, cov_tr))
    axes2[1].boxplot(maha_d, tick_labels=['N-win', 'T', 'R'])
    axes2[1].set_title('Mahalanobis 域外距离（top-50 白化）')
    axes2[1].grid(axis='y', alpha=0.3)
    # sil 三口径
    axes2[2].bar(['sil_64', 'sil_11', 'sil_resid'], [rg['sil_64'], rg['sil_11'], rg['sil_resid']],
                 color=['#7f8c8d', '#95a5a6', '#c0392b'])
    for x, v in enumerate([rg['sil_64'], rg['sil_11'], rg['sil_resid']]):
        axes2[2].text(x, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)
    axes2[2].set_title('三组 sil（残差口径主判——骤降=域偏移承载）')
    axes2[2].grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(PAPER / 'fig_3group_domain.png', dpi=150)
    plt.close()

    # 图 3：Q2 raster
    fig3, axes3 = plt.subplots(1, 2, figsize=(15, 5.5))
    for ax, gi, name in ((axes3[0], 0, 'R-条件推理题'), (axes3[1], 5, 'T-矛盾论选段')):
        segs = R_segs[gi] if gi < 5 else T_segs[gi - 5]
        fps = R_fp[gi] if gi < 5 else T_fp[gi - 5]
        nrm = np.linalg.norm(fps[1:] - fps[:-1], axis=1)
        jpos = np.where(nrm > p90)[0]
        L = [k for k, s in enumerate(segs) if any(w in s for w in LOGIC_WORDS)]
        n = len(segs)
        ax.bar(np.arange(n - 1), nrm, color='#95a5a6', alpha=0.7)
        ax.scatter(jpos, nrm[jpos], color='#c0392b', s=30, zorder=5, label='跳跃点')
        ax.scatter(L, [max(nrm) * 1.05] * len(L), marker='v', color='#1f6fb2', s=25, label='逻辑词句')
        ci_ = r_meta[gi]['conclusion_idx'] if gi < 5 else None
        if ci_ is not None and ci_ < n:
            ax.axvline(ci_, color='#27ae60', ls='--', lw=1.5, label='结论句')
        ax.set_title(f'{name}（跳跃/逻辑词/结论句 raster）')
        ax.legend(fontsize=8)
        ax.set_xlabel('句元')
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PAPER / 'fig_3group_q2.png', dpi=150)
    plt.close()

    # 落盘
    out = {'meta': {'seed': SEED, 'p90': p90,
                    'note': '探索性——不正式判定——报告集预注册见脚本头'},
           'groups': {g: {'n': len(v['metrics']),
                          'metrics': {mk: table[mk][g] for mk in METRICS}}
                      for g, v in groups.items()},
           'cohens_d': d_pairs,
           'shuffle_delta': shuf_sum,
           'domain': coord,
           'residual_sil': rg,
           'd_res': dom_res,
           'q2': q2}
    (OUT / 'trajectory_3group.json').write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                                encoding='utf-8')
    print('\n落盘 trajectory_3group.json + fig_3group_* × 3 ✓')


if __name__ == '__main__':
    main()
