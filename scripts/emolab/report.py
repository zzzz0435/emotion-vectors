"""Simplified-Chinese checkpoint-1 summary (plain language, honest about failures)."""
import os

import pandas as pd

CORE = ["anger", "fear", "sadness", "joy", "compassion", "pain"]
PAIN_VEC = "pain_painaxis"
LOW_AUROC = 0.7
SEP_THRESHOLD = 0.75
AXIS_NAMES = {"V": "效价 V（好坏）", "A": "唤起度 A（激动程度）", "D": "支配度 D（掌控感）"}


def _f(x: float) -> str:
    return f"{x:.2f}"


def _best(validation: pd.DataFrame) -> pd.DataFrame:
    idx = validation.groupby("label")["auroc_vs_neutral"].idxmax()
    return validation.loc[idx].set_index("label")


def _auroc_rows(best: pd.DataFrame, labels: list[str]) -> list[str]:
    return [f"| {l} | {int(best.loc[l, 'layer'])} | {_f(best.loc[l, 'auroc_vs_neutral'])} |"
            for l in labels]


def _auroc_section(validation: pd.DataFrame, core: list[str]) -> list[str]:
    best = _best(validation)
    first = [c for c in [*core, PAIN_VEC] if c in best.index]
    rest = sorted(l for l in best.index if l not in first)
    head = ["| 情绪 | 最佳层 | AUROC |", "|---|---|---|"]
    lines = ["## 一、每种情绪的 AUROC（最佳层）", "",
             "AUROC 衡量「向量能不能把这种情绪的故事和中性故事分开」，"
             "用的是没参与训练的故事：1.00 是完全分开，0.50 等于乱猜。", "",
             "核心情绪（含 Pain-axis 疼痛向量）：", "", *head,
             *_auroc_rows(best, first), "", "其余情绪（地图用）：", "", *head,
             *_auroc_rows(best, rest), "",
             "注意：「最佳层」是在这批同样的留出故事上挑出来的，所以这里的 AUROC 会略微偏乐观。", ""]
    weak = [c for c in core if c in best.index and best.loc[c, "auroc_vs_neutral"] < LOW_AUROC]
    if weak:
        lines.append(f"**警告：** 下列核心情绪在每一层的 AUROC 都低于 {LOW_AUROC}，"
                     f"也就是说向量连这种情绪的故事和中性故事都分不太开，"
                     f"后面用它做调整不可信：{'、'.join(weak)}。")
    else:
        lines.append(f"六种核心情绪至少在某一层的 AUROC 都不低于 {LOW_AUROC}。")
    lines.append("")
    return lines


def _separability_section(sep: dict) -> list[str]:
    lines = ["## 二、愤怒 vs 恐惧 能不能分开", "",
             f"愤怒和恐惧是本研究要比较的一对情绪。如果它们的向量方向分不开，就无法说「愤怒」和「恐惧」"
             f"各自有独立的作用。门槛：没参与训练的故事上 AUROC ≥ {SEP_THRESHOLD}。", "",
             "| 层 | AUROC | 结果 |", "|---|---|---|"]
    ok_layers = []
    for L in sorted(sep, key=int):
        ok = bool(sep[L]["separable"])
        if ok:
            ok_layers.append(L)
        lines.append(f"| {L} | {_f(sep[L]['auroc'])} | {'可以分开' if ok else '不能分开'} |")
    lines.append("")
    if ok_layers:
        lines.append(f"白话：在第 {'、'.join(ok_layers)} 层，愤怒和恐惧可以分开，"
                     f"检验愤怒与恐惧差异的假设（H3）可以进行。")
        if len(ok_layers) < len(sep):
            lines.append("在其余层则分不开，之后只应使用能分开的层。")
    else:
        lines.append("白话：**愤怒和恐惧在所有层都没有分开**，两个向量基本在指同一件事。"
                     "这意味着检验愤怒与恐惧差异的假设（H3）没有基础，"
                     "研究报告里 H3 会写成「无法检验」，而不是「没有效果」。")
    lines.append("")
    return lines


def _contam_section(contam: dict) -> list[str]:
    lines = ["## 三、污染检查", "",
             "检查每个情绪向量里，是否已经混进「惩罚」或「伤害他人」这两个概念的成分"
             "（如果混进去了，调整该情绪时等于同时在调整惩罚概念，结果会被混淆）。"
             "被标记的情绪会另外产生一份去除这些成分的干净向量。", ""]
    hits: dict[tuple[str, str], list[str]] = {}
    for L in sorted(contam, key=int):
        for probe, by_lab in contam[L].items():
            for lab, r in by_lab.items():
                if r.get("flagged"):
                    hits.setdefault((lab, probe), []).append(L)
    if not hits:
        lines.append("没有任何情绪被标记，所有向量都没有明显混入惩罚或伤害他人的成分。")
    else:
        lines += ["被标记的情绪：", "", "| 情绪 | 混入的概念 | 被标记的层 |", "|---|---|---|"]
        lines += [f"| {lab} | {probe} | {'、'.join(ls)} |" for (lab, probe), ls in sorted(hits.items())]
    lines.append("")
    return lines


def _axes_section(axes_info: dict) -> list[str]:
    fit = axes_info.get("fits", {}).get("all", {})
    layers = sorted(fit, key=int)
    lines = ["## 四、三个轴（效价 / 唤起度 / 支配度）可不可靠", "",
             "我们用已有的心理学评分（NRC VAD）把情绪向量拟合成三条通用的轴。"
             "「留一法相关」的做法是：每次拿掉一种情绪，用其余情绪预测它的评分，再看预测和真实评分有多像。"
             "相关 ≥ 0.5 才算这条轴「可靠」；不可靠的轴不能用来做后面的分析。", "",
             "| 轴 | " + " | ".join(f"第 {L} 层" for L in layers) + " |",
             "|---|" + "---|" * len(layers)]
    for k, name in AXIS_NAMES.items():
        cells = [f"{_f(fit[L][k]['loo_r'])}（{'可靠' if fit[L][k]['reliable'] else '不可靠'}）" for L in layers]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.append("")
    fa = axes_info.get("fits", {}).get("for_anger", {})
    fa_layers = sorted(fa, key=int)
    if fa_layers:
        lines += ["检验 H4 用的是「不含愤怒和恐惧」拟合的轴（避免循环论证），它的效价轴 V 可靠性：", "",
                  "| 轴 | " + " | ".join(f"第 {L} 层" for L in fa_layers) + " |",
                  "|---|" + "---|" * len(fa_layers),
                  "| V（不含愤怒、恐惧） | " + " | ".join(
                      f"{_f(fa[L]['V']['loo_r'])}（{'可靠' if fa[L]['V']['reliable'] else '不可靠'}）"
                      for L in fa_layers) + " |", ""]
    if not layers:
        lines.append("（没有通用轴的拟合结果。）")
    elif any(fit[L]["V"]["reliable"] for L in layers):
        lines.append("白话：用全部情绪拟合时，效价轴至少在部分层可靠。")
        for k in "AD":
            if not any(fit[L][k]["reliable"] for L in layers):
                lines.append(f"但{AXIS_NAMES[k]}在所有层都不可靠，后续分析不使用它。")
    else:
        lines.append("白话：用全部情绪拟合时，**效价轴在每一层都不可靠**，"
                     "拟合出来的方向没法预测情绪的好坏评分。")
    res = axes_info.get("residuals", {})
    if res:
        lines += ["", "去掉轴之后的「残差向量」是否可用（untestable 表示当层效价轴不可靠，无法产生）：", ""]
        for name, st in sorted(res.items()):
            lines.append(f"- {name}：" + "，".join(
                f"第 {L} 层 {s}" for L, s in sorted(st.items(), key=lambda x: int(x[0]))))
    lines.append("")
    # H4 verdict: anger__resid3 per layer, else V reliability of the for_anger fit
    st = res.get("anger__resid3")
    if st:
        usable = sorted((L for L, s in st.items() if s == "ok"), key=int)
    elif fa_layers:
        usable = [L for L in fa_layers if fa[L]["V"]["reliable"]]
    else:
        usable = None
    if usable is None:
        lines.append("H4 判定：没有不含愤怒、恐惧的轴拟合结果，**H4 无法检验**。")
    elif usable:
        lines.append(f"H4 判定：H4 可以在第 {'、'.join(usable)} 层检验（这些层能得到去掉轴后的愤怒残差向量）。")
        removed = axes_info.get("residual_axes", {}).get("anger__resid3", {})
        parts = [f"第 {L} 层去掉 {'、'.join(removed[L]) or '（无）'}" for L in usable if L in removed]
        if parts:
            lines.append("各层实际去掉的轴（只去掉可靠的轴）：" + "；".join(parts) + "。")
    else:
        lines.append("H4 判定：不含愤怒、恐惧的效价轴在每一层都不可靠，去掉轴后的愤怒向量做不出来，"
                     "研究报告里会写成「H4 无法检验」。")
    lines.append("")
    return lines


def _sae_section(sae_tops: dict) -> list[str]:
    lines = ["## 五、愤怒、恐惧最接近的特征字典条目", "",
             "「特征字典」（SAE）是把模型内部拆成许多单一含义特征的现成词典。"
             "下面是与我们的向量余弦最高的特征（编号、余弦）。注意该词典训练于基础版模型，"
             "而我们的向量来自指令版，只能当粗略参考；每个特征的含义要看故事片段才能判断，"
             "详见 6.9 的 sae_L18.md。", ""]
    for lab in ("anger", "fear"):
        feats = sae_tops.get(lab)
        if feats:
            text = "；".join(f"特征 {i}（{_f(c)}）" for i, c in feats)
        elif lab in sae_tops:  # sae_L18.md exists but this label's section has no results
            text = "该向量没有结果（已跳过）。"
        else:
            text = "未提供（没有 sae_L18.md）。"
        lines.append(f"- {lab}：{text}")
    lines.append("")
    return lines


def _pain_section(pc: dict | None) -> list[str]:
    lines = ["## 六、与 Pain-axis 疼痛向量的相似度", ""]
    if not pc:
        return lines + ["未提供（没有 pain_axis_compare.json，该部分略过）。", ""]
    v = pc["vectors"][pc.get("primary", "s2")]
    lines += [f"把我们的疼痛向量，和 Pain-axis 论文公开的疼痛向量（第 {pc['their_layer']} 层）算余弦相似度，"
              f"并和其他情绪向量对比，看疼痛是否特别突出。", "",
              "| 向量 | 与对方疼痛向量的余弦 |", "|---|---|",
              f"| 我们的 pain | {_f(v['cos_pain'])} |",
              f"| 我们的 pain_painaxis | {_f(v['cos_pain_painaxis'])} |"]
    lines += [f"| {k}（对照） | {_f(c)} |" for k, c in v["cos_baselines"].items()]
    lines.append("")
    base_max = max(v["cos_baselines"].values()) if v["cos_baselines"] else float("-inf")
    pain = v["cos_pain"]
    if pain > base_max:
        lines.append(f"白话：我们用故事提取的疼痛向量与对方的相似度（{_f(pain)}）高于所有对照情绪"
                     f"（最高 {_f(base_max)}），疼痛比较突出。")
    else:
        lines.append(f"白话：我们用故事提取的疼痛向量与对方的相似度（{_f(pain)}）并不比对照情绪"
                     f"（最高 {_f(base_max)}）高，疼痛没有特别突出，不能说两者抓到了同一个东西。")
    lines.append(f"另外，pain_painaxis 的余弦是 {_f(v['cos_pain_painaxis'])}。它是用 Pain-axis 自己的句子"
                 f"构建的，和对方相似本来就是预期的，所以只是较弱的证据，不用它来判断疼痛是否突出。")
    cav = []
    if not pc.get("layer_match", True):
        cav.append(f"层不一致（对方 {pc['their_layer']}，我们 {pc['our_layer']}）")
    if pc.get("extraction_mismatch"):
        cav.append(f"取值方式不同（{pc['extraction_mismatch']}）")
    if pc.get("model_mismatch"):
        cav.append(f"模型不同（{pc['model_mismatch']}）")
    if pc.get("category_mismatch"):
        cav.append(f"句子类别不同（{pc['category_mismatch']}）")
    if cav:
        lines.append("注意的差异：" + "；".join(cav) + "，所以这个数字只能当粗略参考。")
    lines.append("")
    return lines


def render_checkpoint1(validation: pd.DataFrame, separability: dict, contamination: dict,
                       axes_info: dict, sae_tops: dict, examples_path: str,
                       pain_compare: dict | None = None, core: list[str] | None = None) -> str:
    """sae_tops: {"anger": [(feature_idx, cos), ...], "fear": [...]}; axes_info: axes.json."""
    base = os.path.basename(examples_path)
    out = ["# 检查点 1 摘要：情绪向量够不够好？", "",
           "这份摘要说明：从模型里提取出的情绪向量是否可信，决定要不要继续做后面的调整实验。"
           "坏消息会直接写出来，不做美化。", ""]
    out += _auroc_section(validation, core or CORE) + _separability_section(separability)
    out += _contam_section(contamination) + _axes_section(axes_info)
    out += _sae_section(sae_tops) + _pain_section(pain_compare)
    out += ["## 七、范例文字", "",
            f"在第 18 层对每种核心情绪施加不同强度（α = 0、0.1、0.2、0.4）后，模型的实际输出见："
            f"[{base}]({base})。这些 α 只是暂定值，正式数值会在计划二校准。"
            f"疼痛只生成了这 12 段范例，没有做更多。", ""]
    return "\n".join(out)
