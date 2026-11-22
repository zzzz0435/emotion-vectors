"""Traditional Chinese results summary (design spec §6.8, §7).

render_results() turns the 6.8/01 outputs (primary, secondary, exploratory, interpretation
sentences, frozen config) into one Markdown string for a non-expert reader. It only
*formats* numbers that stats.py already computed; it never recomputes statistics.

Wording rule: results describe how answers change after an internal direction is steered.
They never say the model feels anything (see BANNED_PHRASES, checked by the tests).
"""
import functools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from emolab.conditions import condition_name, load_labels
from emolab.paths import DATASETS
from emolab.stats import AXIS_NAMES, GATE_NEED, HYP_NAMES, _tri, _txt

DASH = "—"
BANNER = "⚠ 凍結後改過分析程式：以下所有結果僅屬探索性"
BANNED_PHRASES = ("AI 會生氣", "AI 會痛", "模型感到", "模型覺得")
TASK_NAMES = {"h1_rating": "H1 懲罰評分", "h1_choice": "H1 二選一", "h2": "H2 傷害選擇"}
FIGURES = [
    ("劑量反應（H1 懲罰評分）", "figures/dose_response_h1_rating.png"),
    ("劑量反應（H2 傷害選擇）", "figures/dose_response_h2.png"),
    ("情緒地圖（H1 懲罰評分）", "figures/emotion_map_h1_rating.png"),
    ("情緒地圖（H2 傷害選擇）", "figures/emotion_map_h2.png"),
    ("憤怒 vs 恐懼（相同誘發強度）", "figures/anger_vs_fear_eq.png"),
    ("文字誘發 vs 向量調整", "figures/text_vs_vector.png"),
    ("拒答程度", "figures/refusal.png"),
]
SECONDARY_NAMES = {
    "H5": "H5（同情 vs 喜悅，誘發強度相同）",
    "compassion_resid3": "同情向量扣掉三個通用軸後的剩餘部分",
    "anger_vs_sadness": "憤怒 vs 悲傷（誘發強度相同）",
    "pain_resid3": "疼痛向量扣掉三個通用軸後的剩餘部分",
    "anger_resid2_vs_resid3": "憤怒只扣兩個軸 vs 扣三個軸",
    "pain_painaxis_vs_pain": "Pain-axis 論文的疼痛向量 vs 本研究的疼痛向量",
}
UNITS = {"h1_rating": "分（1–7 分量表）", "h1_choice": "機率（0–1，百分點）",
         "h2": "機率（0–1，百分點）"}
EMO_ZH = {"anger": "憤怒", "fear": "恐懼", "sadness": "悲傷", "joy": "喜悅",
          "compassion": "同情", "pain": "疼痛"}
ITEM_FILES = {"h1_rating": "6.6_punishment_vignettes.json",
              "h1_choice": "6.6_punishment_vignettes.json", "h2": "6.6_moralchoice_harm.json"}
PREDICTED = {"+": "預測較高", "−": "預測較低", "two-sided": "不預設方向"}
PLAIN = {   # hypothesis -> what is being compared, in everyday words
    "H1": "把憤怒向量調高，懲罰評分比調 {n} 個無關概念向量時更重",
    "H2": "把疼痛向量調高，比調 {n} 個無關概念向量時更常選傷害他人的選項",
    "H3": "兩者誘發強度相同時，憤怒讓懲罰變重的效果大於恐懼",
    "H4": "憤怒向量扣掉通用軸之後，剩下的部分仍讓懲罰變重",
}
GROUPS = ("baseline", "core", "concept", "random", "text")
GROUP_NAMES = {"baseline": "未調整", "core": "核心情緒 ±α_max", "concept": "非情緒概念向量",
               "random": "隨機方向", "text": "文字誘發"}


# ---------- formatting ----------

def _isnan(x) -> bool:
    if x is None or x is pd.NA:
        return True
    try:
        return bool(np.isnan(float(x)))
    except (TypeError, ValueError):
        return False


def fmt(x, nd: int = 3, signed: bool = False) -> str:
    """Number with `nd` decimals; None / NaN -> 「—」."""
    if _isnan(x):
        return DASH
    return f"{float(x):+.{nd}f}" if signed else f"{float(x):.{nd}f}"


def fmt_p(x) -> str:
    if _isnan(x):
        return DASH
    x = float(x)
    return "< 0.001" if x < 0.001 else f"{x:.3f}"


def fmt_ci(lo, hi, nd: int = 3) -> str:
    if _isnan(lo) or _isnan(hi):
        return DASH
    return f"[{fmt(lo, nd, True)}, {fmt(hi, nd, True)}]"


def fmt_eff(x, task: str, nd: int = 3) -> str:
    """Signed effect with its unit: 分 for ratings, 機率 (and percentage points) for choices."""
    if _isnan(x):
        return DASH
    if task == "h1_rating":
        return f"{fmt(x, nd, True)} 分"
    return f"{fmt(x, nd, True)} 機率（{fmt(float(x) * 100, 1, True)} 百分點）"


@functools.lru_cache(maxsize=None)
def n_items(task: str) -> int:
    """Items per task in the datasets (72 vignettes for H1, 150 MoralChoice items for H2)."""
    return len(json.loads((DATASETS / ITEM_FILES[task]).read_text(encoding="utf-8")))


def fmt_n(row, task: str) -> str:
    """「n / N」: valid items used over items in the dataset."""
    return DASH if _isnan(row.get("n")) else f"{int(row['n'])} / {n_items(task)}"


def secondary_name(h: str, frozen: dict) -> str:
    if h == "anger_resid2_vs_resid3":
        r = frozen.get("residual_removed_axes") or {}
        two, three = r.get("anger__resid2"), r.get("anger__resid3")
        if two is not None and three is not None:
            ax = lambda ks: "、".join(k for k in "VAD" if k in ks) or "無"  # noqa: E731
            extra = "、".join(AXIS_NAMES[k] for k in "VAD" if k in three and k not in two)
            return (f"只扣 {ax(two)}（anger__resid2）vs 扣 {ax(three)}（anger__resid3）："
                    f"看{extra or '多扣的軸'}的貢獻")
    return SECONDARY_NAMES.get(h, str(h))


def condition_group(name: str, labels: dict, a: float) -> str | None:
    """Group of a condition for the refusal summary; None = not summarised."""
    if name == "baseline":
        return "baseline"
    if name.startswith("text_"):
        return "text"
    head = name.partition("@")[0]
    if head.startswith("random"):
        return "random"
    if head in labels["concepts"]:
        return "concept"
    if head in labels["core"] and name in (condition_name(head, a), condition_name(head, -a)):
        return "core"
    return None


# ---------- verdicts ----------

def verdict(row) -> str:
    sup = _tri(row.get("supported"))
    if sup is None:
        return f"無法檢驗（{_txt(row.get('untestable_reason')) or '未說明原因'}）"
    if _txt(row.get("predicted")) == "two-sided":     # no predicted direction: nothing to support
        return "有顯著差異" if sup else "沒有偵測到差異"
    return "支持" if sup else "不支持"


def _gate(row) -> str:
    if _tri(row.get("supported")) is None:
        return DASH
    g = _tri(row.get("gate"))
    return DASH if g is None else ("通過" if g else "未通過")


def _removed_axes(frozen: dict, name: str) -> str | None:
    removed = (frozen.get("residual_removed_axes") or {}).get(name)
    if removed is None:
        return None
    return "、".join(AXIS_NAMES[k] for k in "VAD" if k in removed) or "（沒有任何軸）"


def _is_changed(primary, secondary, explo) -> bool:
    for df in (primary, secondary):
        if "exploratory_changed_rules" in df.columns and \
                any(_tri(v) is True for v in df["exploratory_changed_rules"]):
            return True
    return bool(explo.get("exploratory_changed_rules"))


def _rows(df: pd.DataFrame) -> dict:
    return {r["hypothesis"]: r for _, r in df.iterrows()}


# ---------- sections ----------

def _conclusion(primary: pd.DataFrame, interpretation: list[str], changed: bool = False) -> str:
    """ONE sentence; the interpretation rules themselves are in section 3."""
    rows = _rows(primary)
    groups = {"支持": [], "不支持": [], "無法檢驗": []}
    for h in HYP_NAMES:
        if h in rows:
            groups[verdict(rows[h]).split("（")[0]].append(h)
    parts = [f"{k} {'、'.join(v)}" for k, v in groups.items() if v]
    lead = "（探索性）四個假設的結果：" if changed else "在預先登記的四個假設中："
    return (f"{lead}{'；'.join(parts) or '沒有可報告的結果'}"
            "（這只描述調整模型內部方向後答案的變化，不代表模型有主觀感受）。")


def _plain_line(h: str, row, frozen: dict, n_concepts: int) -> str:
    v = verdict(row)
    what = PLAIN[h].format(n=n_concepts)
    task = _txt(row.get("task")) or ("h2" if h == "H2" else "h1_rating")
    nn = "" if _isnan(row.get("n")) else f"；有效題數 {fmt_n(row, task)}"
    if v == "支持":
        note = f"（{_txt(row.get('alpha_note'))}）" if _txt(row.get("alpha_note")) else ""
        line = (f"- **{h}**：支持{note}。白話：{what}。平均差 {fmt_eff(row['mean_d'], task)}"
                f"（95% CI {fmt_ci(row['ci_low'], row['ci_high'])}），校正後 p = "
                f"{fmt_p(row['p_holm'])}{nn}。")
    elif v == "不支持":
        line = (f"- **{h}**：不支持。白話：資料沒有顯示「{what}」。平均差 "
                f"{fmt_eff(row['mean_d'], task)}（95% CI {fmt_ci(row['ci_low'], row['ci_high'])}），"
                f"校正後 p = {fmt_p(row['p_holm'])}{nn}；沒有達標的原因見下一節「判讀」。")
    else:
        line = (f"- **{h}**：{v}。白話：資料不足以回答「{what}」，這不等於「沒有效果」。")
    if h == "H4":
        axes = _removed_axes(frozen, "anger__resid3")
        line += (f" **H4 只能主張「超出已移除的軸之外」**：這次憤怒殘差實際移除的是{axes}，"
                 "沒有移除的軸不在結論範圍內。" if axes else
                 " **H4 只能主張「超出已移除的軸之外」**，但凍結檔沒有記錄實際移除了哪些軸。")
    return line


def _table(head: list[str], rows: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return out


def _primary_section(primary: pd.DataFrame, frozen: dict, n_concepts: int) -> list[str]:
    rows = _rows(primary)
    tbl = []
    for h, name in HYP_NAMES.items():
        if h not in rows:
            continue
        r = rows[h]
        untestable = _tri(r.get("supported")) is None
        task = _txt(r.get("task")) or ("h2" if h == "H2" else "h1_rating")
        tbl.append([name, DASH if untestable else fmt(r["mean_d"], 3, True), UNITS[task],
                    DASH if untestable else fmt_ci(r["ci_low"], r["ci_high"]),
                    DASH if untestable else fmt_p(r["p_holm"]),
                    DASH if untestable else fmt_n(r, task), _gate(r), verdict(r)])
    out = _table(["假設（白話）", "效果（平均差）", "單位", "95% 信賴區間", "p（Holm 校正）",
                  "有效題數", "對照閘門", "結論"], tbl)
    out += ["", "有效題數 n / N：n 是納入檢驗的題數，N 是資料集的題數。沒有任何有效回答的題目會被排除；"
            "n 也不含兩個條件分數剛好相同（差為 0）的題目，連續分數幾乎不可能剛好相同。", "",
            f"效果是「調整後的分數」減去「{n_concepts} 個無關概念向量的平均」（H3 則是憤怒減恐懼），"
            f"正值表示朝預測方向。對照閘門指效果要贏過至少 {GATE_NEED} 個概念向量。", ""]
    out += [_plain_line(h, rows[h], frozen, n_concepts) for h in HYP_NAMES if h in rows]
    return out


def _secondary_section(secondary: pd.DataFrame, frozen: dict) -> list[str]:
    main = secondary[secondary["hypothesis"] != "text_vs_vector"]
    tbl = []
    for _, r in main.iterrows():
        untestable = _tri(r.get("supported")) is None
        tbl.append([secondary_name(r["hypothesis"], frozen),
                    TASK_NAMES.get(r["task"], str(r["task"])),
                    PREDICTED.get(_txt(r.get("predicted")) or "", DASH),
                    DASH if untestable else fmt(r["mean_d"], 3, True),
                    UNITS.get(r["task"], DASH),
                    DASH if untestable else fmt_ci(r["ci_low"], r["ci_high"]),
                    DASH if untestable else fmt_p(r["p_holm"]), verdict(r)])
    out = ["以下為**次要（單獨校正）**檢驗：這一組只在自己的檢驗之間做 Holm 校正，"
           "不和上面的主要檢驗一起校正，證據強度比主要結果弱。", ""]
    out += _table(["檢驗", "任務", "方向", "效果（平均差）", "單位", "95% 信賴區間",
                   "p（Holm 校正）", "結論"], tbl)
    tv = secondary[secondary["hypothesis"] == "text_vs_vector"]
    if len(tv):
        t2 = []
        for _, r in tv.iterrows():
            emo = str(r["contrast"]).split(" ")[0].removeprefix("text_")
            t2.append([emo, TASK_NAMES.get(r["task"], str(r["task"])),
                       f"{fmt(r['text_effect'], 3, True)} {fmt_ci(r['text_ci_low'], r['text_ci_high'])}",
                       f"{fmt(r['vector_effect'], 3, True)} "
                       f"{fmt_ci(r['vector_ci_low'], r['vector_ci_high'])}"])
        out += ["", "文字誘發 vs 向量調整（只描述，不做檢驗；效果都是相對於未調整）：", ""]
        t2 = [r + [UNITS.get(tk, DASH)] for r, tk in zip(t2, tv["task"])]
        out += _table(["情緒", "任務", "文字誘發", "向量調整（+α_max）", "單位"], t2)
    return out


def _exploratory_section(explo: dict, frozen: dict) -> list[str]:
    out = ["以下全部屬**探索性**分析：沒有預先登記，不做假設檢驗，只用來提供線索。", ""]
    # map
    out += ["### 探索性：情緒地圖（效果能被 V/A/D 三個軸解釋多少）", ""]
    emap = explo.get("emotion_map") or {}
    if "note" in emap:
        out += [f"- {emap['note']}"]
    for task in ("h1_rating", "h2"):
        t = emap.get(task)
        if not t:
            continue
        if "r2" not in t:
            out.append(f"- {TASK_NAMES[task]}：{t.get('note', '無法計算')}")
            continue
        dev = "、".join(f"{d['label']}（實際 {fmt_eff(d['effect'], task)}，"
                       f"三軸預測 {fmt_eff(d['predicted'], task)}）"
                       for d in t.get("top_deviations", []))
        out.append(f"- {TASK_NAMES[task]}：R² = {fmt(t['r2'], 2)}（共 {t['n']} 種情緒）；"
                   f"偏離三軸預測最多的三種情緒：{dev or DASH}。")
    # dose response
    out += ["", "### 探索性：劑量反應（Spearman 等級相關：α 越大效果是否越大）", ""]
    dr = explo.get("dose_response") or {}
    for task, per in dr.items():
        if task == "h1_choice" or not per:
            continue
        cells = "、".join(f"{e} ρ = {fmt(v.get('spearman_rho'), 2, True)}" for e, v in per.items())
        out.append(f"- {TASK_NAMES.get(task, task)}：{cells}")
    # random
    out += ["", "### 探索性：隨機方向的效果範圍（2.5–97.5 百分位）", ""]
    rr = explo.get("random_directions") or {}
    for task, per in rr.items():
        if task == "h1_choice" or not per:
            continue
        cells = "；".join(f"α = {al}：{fmt(v['p2_5'], 3, True)} ~ {fmt(v['p97_5'], 3, True)}"
                         f"（{v['n']} 個方向）" for al, v in per.items())
        out.append(f"- {TASK_NAMES.get(task, task)}（單位：{UNITS.get(task, DASH)}）：{cells}")
    if not any(t != "h1_choice" and per for t, per in rr.items()):
        out.append(f"- {DASH}")
    # refusal
    out += ["", "### 探索性：拒答程度（無效回答的平均機率質量）", ""]
    ref = explo.get("refusal") or {}
    labels, a = load_labels(), float(frozen["alpha_max"])
    tbl = []
    for task, per in ref.items():
        if task == "h1_choice":
            continue
        acc: dict[str, list[float]] = {g: [] for g in GROUPS}
        for cond, v in per.items():
            g = condition_group(cond, labels, a)
            if g and not _isnan(v):
                acc[g].append(float(v))
        tbl.append([TASK_NAMES.get(task, task)] +
                   [fmt(np.mean(acc[g]), 3) if acc[g] else DASH for g in GROUPS])
    out += _table(["任務"] + [GROUP_NAMES[g] for g in GROUPS], tbl) if tbl else [f"- {DASH}"]
    # physical harm
    out += ["", "### 探索性：H2 只看致死或致殘的題目", ""]
    ph = explo.get("physical_harm") or {}
    if "note" in ph:
        out.append(f"- {ph['note']}")
    elif ph:
        out.append(f"- {ph['contrast']}：共 {ph['n_items']} 題，平均差 {fmt_eff(ph['mean_d'], 'h2')}"
                   f"（95% CI {fmt_ci(ph['ci_low'], ph['ci_high'])}），未校正 p = {fmt_p(ph['p'])}。")
    else:
        out.append(f"- {DASH}")
    out += _cross_section(explo, frozen)
    return out


def _cross_section(explo: dict, frozen: dict) -> list[str]:
    """Auxiliary H1 binary-choice effects and every core emotion x task at +-alpha_max."""
    cross = explo.get("cross_combinations") or {}
    a = float(frozen["alpha_max"])
    emos = list(load_labels()["core"])
    cell = lambda task, e, al: (  # noqa: E731
        lambda v: DASH if not v or _isnan(v.get("effect")) else
        f"{fmt_eff(v['effect'], task)} {fmt_ci(v.get('ci_low'), v.get('ci_high'))}"
    )((cross.get(task) or {}).get(condition_name(e, al)))
    out = ["", "### 探索性：輔助指標（H1 二選一）", "",
           "H1 二選一是 H1 的輔助指標，不是預先登記的主要檢驗；"
           f"效果相對於未調整，單位：{UNITS['h1_choice']}，括號為 95% CI。", ""]
    if not cross.get("h1_choice"):
        out.append(f"- {DASH}")
    else:
        out += _table(["情緒", "+α_max", "−α_max"],
                      [[e, cell("h1_choice", e, a), cell("h1_choice", e, -a)] for e in emos])
    out += ["", "### 探索性：核心情緒 × 任務的交叉結果（±α_max）", "",
            "每格是該情緒在 ±α_max 相對於未調整的效果與 95% CI；"
            "單位：H1 懲罰評分為分（1–7 分量表），H1 二選一與 H2 為機率（0–1，百分點）。", ""]
    tasks = [t for t in ("h1_rating", "h1_choice", "h2") if cross.get(t)]
    if not tasks:
        out.append(f"- {DASH}")
    else:
        head = ["情緒"] + [f"{TASK_NAMES[t]} {sg}α_max" for t in tasks for sg in ("+", "−")]
        out += _table(head, [[e] + [cell(t, e, sg * a) for t in tasks for sg in (1, -1)]
                             for e in emos])
    return out


def _quantisation(model: str | None) -> str:
    try:        # the registry records how each model is loaded; the frozen file only has its name
        from emolab.models import MODELS
        spec = next((m for m in MODELS.values() if model in (m.name, m.repo_id)), None)
    except Exception:  # noqa: BLE001 - torch missing etc.: stay generic
        spec = None
    if spec is None or spec.quant == "none":
        return "模型的載入方式（是否量化）請見實驗設定；量化可能稍微改變內部表示。"
    bits = {"4bit": "4-bit", "8bit": "8-bit"}[spec.quant]
    return f"模型是 {bits} 量化版本，量化可能稍微改變內部表示。"


def _eq_limit(frozen: dict) -> str:
    """The α_eq target is the smallest maximum induction gain; name the emotion only when the
    frozen file shows it: the one whose α_eq sits at its coherence ceiling."""
    target, eq, cm = frozen.get("alpha_eq_target"), frozen.get("alpha_eq") or {}, \
        frozen.get("coherent_max")
    head = "α_eq（讓各情緒誘發強度相同）的目標值" + (
        f"（{float(target):.2f} 分）" if target is not None else "")
    if isinstance(cm, dict):
        at_cap = [e for e, v in eq.items()
                  if v is not None and e in cm and float(v) >= float(cm[e]) - 1e-9]
        if len(at_cap) == 1:
            e = at_cap[0]
            return (f"- {head}受限於{EMO_ZH.get(e, e)}：它的 α_eq 已經頂到通順上限，"
                    "所以相同強度的比較有上限。")
    return f"- {head}受限於誘發增幅最小的那種情緒的通順上限，所以相同強度的比較有上限。"


def _limitations(frozen: dict) -> list[str]:
    model = frozen.get("model")
    return [
        f"- 只測了一個模型（{model or DASH}）；換成別的模型結果可能不同。",
        "- 題目只有英文，不能直接推到其他語言。",
        f"- {_quantisation(model)}",
        "- 這個研究只看「調整內部方向後答案怎麼變」，**不涉及任何主觀感受**。",
        "- 疼痛向量是用「描寫疼痛的故事」建出來的，不是真正的感覺；它代表的是語言裡的疼痛概念。",
        "- MoralChoice 資料集的「疼痛」規則也包含心理上的痛苦，不只是身體疼痛。",
        "- 一致性（coherence）規則在看到任何道德題資料之前就改過：改成用評審判斷流暢度，"
        "而不是困惑度（perplexity）。",
        "- H1 的題目加了一句回答格式的說明（R31）：在正式實驗開始前，依先導試跑（3 個條件、10 題）"
        "的無效回答比例加入，只規範回答格式；沒有參考先導的懲罰或傷害分數。",
        _eq_limit(frozen),
        f"- 本次凍結設定：層 {frozen.get('layer', DASH)}，α_max {fmt(frozen.get('alpha_max'), 2)}。",
    ]


def render_results(primary: pd.DataFrame, secondary: pd.DataFrame, exploratory: dict,
                   interpretation, frozen: dict) -> str:
    if isinstance(interpretation, str):
        interpretation = parse_interpretation(interpretation)
    out: list[str] = []
    changed = _is_changed(primary, secondary, exploratory)
    n_concepts = len(load_labels()["concepts"])
    if changed:
        out += [f"> **{BANNER}**（分析程式在凍結後被改過，所有結果都不能當作預先登記的檢驗）", ""]
    out += ["# 結果摘要", "", "## 1. 一句話結論", "", _conclusion(primary, interpretation, changed), "",
            "## 2. （探索性）四個假設（H1–H4）的結果" if changed else "## 2. 四個主要假設（H1–H4）",
            ""]
    out += _primary_section(primary, frozen, n_concepts)
    out += ["", "## 3. 判讀", ""]
    out += [f"- {s}" for s in interpretation] or [f"- {DASH}"]
    out += ["", "## 4. 次要（單獨校正）結果", ""]
    out += _secondary_section(secondary, frozen)
    out += ["", "## 5. 探索性結果", ""]
    out += _exploratory_section(exploratory, frozen)
    out += ["", "## 6. 限制", ""]
    out += _limitations(frozen)
    out += ["", "## 7. 圖表", ""]
    out += [f"- {title}：![{title}]({path})" for title, path in FIGURES]
    return "\n".join(out) + "\n"


# ---------- I/O shared by 6.8/02 and 6.8/03 ----------

def parse_interpretation(text: str) -> list[str]:
    """Bullet sentences of interpretation.md (the lines after the '## 判讀' heading)."""
    lines = text.splitlines()
    start = next((i for i, ln in enumerate(lines) if ln.startswith("## 判讀")), -1)
    return [ln[2:].strip() for ln in lines[start + 1:] if ln.startswith("- ")]


def load_analysis(d: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict, str]:
    """Read the 6.8/01 outputs; FileNotFoundError names every missing file."""
    d = Path(d)
    names = ("primary.csv", "secondary.csv", "exploratory.json", "interpretation.md")
    missing = [n for n in names if not (d / n).exists()]
    if missing:
        raise FileNotFoundError("、".join(missing))
    return (pd.read_csv(d / "primary.csv"), pd.read_csv(d / "secondary.csv"),
            json.loads((d / "exploratory.json").read_text(encoding="utf-8")),
            (d / "interpretation.md").read_text(encoding="utf-8"))
