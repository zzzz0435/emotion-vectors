"""Simplified-Chinese checkpoint-1 summary (plain language, honest about failures)."""
import os

import pandas as pd

CORE = ["anger", "fear", "sadness", "joy", "compassion", "pain"]
PAIN_VEC = "pain_painaxis"
LOW_AUROC = 0.7
SEP_THRESHOLD = 0.75
AXIS_NAMES = {"V": "效價 V（好壞）", "A": "喚起度 A（激動程度）", "D": "支配度 D（掌控感）"}


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
    head = ["| 情緒 | 最佳層 | AUROC |", "|---|---|---|"]
    lines = ["## 一、每種情緒的 AUROC（最佳層）", "",
             "AUROC 衡量「向量能不能把這種情緒的故事和中性故事分開」，"
             "用的是沒參與訓練的故事：1.00 是完全分開，0.50 等於亂猜。", "",
             "核心情緒（含 Pain-axis 疼痛向量）：", "", *head,
             *_auroc_rows(best, first), "", "其餘情緒（地圖用）：", "", *head,
             *_auroc_rows(best, rest), "",
             "注意：「最佳層」是在這批同樣的留出故事上挑出來的，所以這裡的 AUROC 會略微偏樂觀。", ""]
    weak = [c for c in core if c in best.index and best.loc[c, "auroc_vs_neutral"] < LOW_AUROC]
    if weak:
        lines.append(f"**警告：** 下列核心情緒在每一層的 AUROC 都低於 {LOW_AUROC}，"
                     f"也就是說向量連這種情緒的故事和中性故事都分不太開，"
                     f"後面用它做調整不可信：{'、'.join(weak)}。")
    else:
        lines.append(f"六種核心情緒至少在某一層的 AUROC 都不低於 {LOW_AUROC}。")
    lines.append("")
    return lines


def _separability_section(sep: dict) -> list[str]:
    lines = ["## 二、憤怒 vs 恐懼 能不能分開", "",
             f"憤怒和恐懼是本研究要比較的一對情緒。如果它們的向量方向分不開，就無法說「憤怒」和「恐懼」"
             f"各自有獨立的作用。門檻：沒參與訓練的故事上 AUROC ≥ {SEP_THRESHOLD}。", "",
             "| 層 | AUROC | 結果 |", "|---|---|---|"]
    ok_layers = []
    for L in sorted(sep, key=int):
        ok = bool(sep[L]["separable"])
        if ok:
            ok_layers.append(L)
        lines.append(f"| {L} | {_f(sep[L]['auroc'])} | {'可以分開' if ok else '不能分開'} |")
    lines.append("")
    if ok_layers:
        lines.append(f"白話：在第 {'、'.join(ok_layers)} 層，憤怒和恐懼可以分開，"
                     f"檢驗憤怒與恐懼差異的假設（H3）可以進行。")
        if len(ok_layers) < len(sep):
            lines.append("在其餘層則分不開，之後只應使用能分開的層。")
    else:
        lines.append("白話：**憤怒和恐懼在所有層都沒有分開**，兩個向量基本在指同一件事。"
                     "這意味著檢驗憤怒與恐懼差異的假設（H3）沒有基礎，"
                     "研究報告裡 H3 會寫成「無法檢驗」，而不是「沒有效果」。")
    lines.append("")
    return lines


def _contam_section(contam: dict) -> list[str]:
    lines = ["## 三、汙染檢查", "",
             "檢查每個情緒向量裡，是否已經混進「懲罰」或「傷害他人」這兩個概念的成分"
             "（如果混進去了，調整該情緒時等於同時在調整懲罰概念，結果會被混淆）。"
             "被標記的情緒會另外產生一份去除這些成分的乾淨向量。", ""]
    hits: dict[tuple[str, str], list[str]] = {}
    for L in sorted(contam, key=int):
        for probe, by_lab in contam[L].items():
            for lab, r in by_lab.items():
                if r.get("flagged"):
                    hits.setdefault((lab, probe), []).append(L)
    if not hits:
        lines.append("沒有任何情緒被標記，所有向量都沒有明顯混入懲罰或傷害他人的成分。")
    else:
        lines += ["被標記的情緒：", "", "| 情緒 | 混入的概念 | 被標記的層 |", "|---|---|---|"]
        lines += [f"| {lab} | {probe} | {'、'.join(ls)} |" for (lab, probe), ls in sorted(hits.items())]
    lines.append("")
    return lines


def _axes_section(axes_info: dict) -> list[str]:
    fit = axes_info.get("fits", {}).get("all", {})
    layers = sorted(fit, key=int)
    lines = ["## 四、三個軸（效價 / 喚起度 / 支配度）可不可靠", "",
             "我們用已有的心理學評分（NRC VAD）把情緒向量擬合成三條通用的軸。"
             "「留一法相關」的做法是：每次拿掉一種情緒，用其餘情緒預測它的評分，再看預測和真實評分有多像。"
             "相關 ≥ 0.5 才算這條軸「可靠」；不可靠的軸不能用來做後面的分析。", "",
             "| 軸 | " + " | ".join(f"第 {L} 層" for L in layers) + " |",
             "|---|" + "---|" * len(layers)]
    for k, name in AXIS_NAMES.items():
        cells = [f"{_f(fit[L][k]['loo_r'])}（{'可靠' if fit[L][k]['reliable'] else '不可靠'}）" for L in layers]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.append("")
    fa = axes_info.get("fits", {}).get("for_anger", {})
    fa_layers = sorted(fa, key=int)
    if fa_layers:
        lines += ["檢驗 H4 用的是「不含憤怒和恐懼」擬合的軸（避免循環論證），它的效價軸 V 可靠性：", "",
                  "| 軸 | " + " | ".join(f"第 {L} 層" for L in fa_layers) + " |",
                  "|---|" + "---|" * len(fa_layers),
                  "| V（不含憤怒、恐懼） | " + " | ".join(
                      f"{_f(fa[L]['V']['loo_r'])}（{'可靠' if fa[L]['V']['reliable'] else '不可靠'}）"
                      for L in fa_layers) + " |", ""]
    if not layers:
        lines.append("（沒有通用軸的擬合結果。）")
    elif any(fit[L]["V"]["reliable"] for L in layers):
        lines.append("白話：用全部情緒擬合時，效價軸至少在部分層可靠。")
        for k in "AD":
            if not any(fit[L][k]["reliable"] for L in layers):
                lines.append(f"但{AXIS_NAMES[k]}在所有層都不可靠，後續分析不使用它。")
    else:
        lines.append("白話：用全部情緒擬合時，**效價軸在每一層都不可靠**，"
                     "擬合出來的方向沒法預測情緒的好壞評分。")
    res = axes_info.get("residuals", {})
    if res:
        lines += ["", "去掉軸之後的「殘差向量」是否可用（untestable 表示當層效價軸不可靠，無法產生）：", ""]
        for name, st in sorted(res.items()):
            lines.append(f"- {name}：" + "，".join(
                f"第 {L} 層 {s}" for L, s in sorted(st.items(), key=lambda x: int(x[0]))))
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
        lines.append("H4 判定：沒有不含憤怒、恐懼的軸擬合結果，**H4 無法檢驗**。")
    elif usable:
        lines.append(f"H4 判定：H4 可以在第 {'、'.join(usable)} 層檢驗（這些層能得到去掉軸後的憤怒殘差向量）。")
        removed = axes_info.get("residual_axes", {}).get("anger__resid3", {})
        parts = [f"第 {L} 層去掉 {'、'.join(removed[L]) or '（無）'}" for L in usable if L in removed]
        if parts:
            lines.append("各層實際去掉的軸（只去掉可靠的軸）：" + "；".join(parts) + "。")
    else:
        lines.append("H4 判定：不含憤怒、恐懼的效價軸在每一層都不可靠，去掉軸後的憤怒向量做不出來，"
                     "研究報告裡會寫成「H4 無法檢驗」。")
    lines.append("")
    return lines


def _sae_section(sae_tops: dict) -> list[str]:
    lines = ["## 五、憤怒、恐懼最接近的特徵字典條目", "",
             "「特徵字典」（SAE）是把模型內部拆成許多單一含義特徵的現成詞典。"
             "下面是與我們的向量餘弦最高的特徵（編號、餘弦）。注意該詞典訓練於基礎版模型，"
             "而我們的向量來自指令版，只能當粗略參考；每個特徵的含義要看故事片段才能判斷，"
             "詳見 6.9 的 sae_L18.md。", ""]
    for lab in ("anger", "fear"):
        feats = sae_tops.get(lab)
        if feats:
            text = "；".join(f"特徵 {i}（{_f(c)}）" for i, c in feats)
        elif lab in sae_tops:  # sae_L18.md exists but this label's section has no results
            text = "該向量沒有結果（已跳過）。"
        else:
            text = "未提供（沒有 sae_L18.md）。"
        lines.append(f"- {lab}：{text}")
    lines.append("")
    return lines


def _pain_section(pc: dict | None) -> list[str]:
    lines = ["## 六、與 Pain-axis 疼痛向量的相似度", ""]
    if not pc:
        return lines + ["未提供（沒有 pain_axis_compare.json，該部分略過）。", ""]
    v = pc["vectors"][pc.get("primary", "s2")]
    lines += [f"把我們的疼痛向量，和 Pain-axis 論文公開的疼痛向量（第 {pc['their_layer']} 層）算餘弦相似度，"
              f"並和其他情緒向量對比，看疼痛是否特別突出。", "",
              "| 向量 | 與對方疼痛向量的餘弦 |", "|---|---|",
              f"| 我們的 pain | {_f(v['cos_pain'])} |",
              f"| 我們的 pain_painaxis | {_f(v['cos_pain_painaxis'])} |"]
    lines += [f"| {k}（對照） | {_f(c)} |" for k, c in v["cos_baselines"].items()]
    lines.append("")
    base_max = max(v["cos_baselines"].values()) if v["cos_baselines"] else float("-inf")
    pain = v["cos_pain"]
    if pain > base_max:
        lines.append(f"白話：我們用故事提取的疼痛向量與對方的相似度（{_f(pain)}）高於所有對照情緒"
                     f"（最高 {_f(base_max)}），疼痛比較突出。")
    else:
        lines.append(f"白話：我們用故事提取的疼痛向量與對方的相似度（{_f(pain)}）並不比對照情緒"
                     f"（最高 {_f(base_max)}）高，疼痛沒有特別突出，不能說兩者抓到了同一個東西。")
    lines.append(f"另外，pain_painaxis 的餘弦是 {_f(v['cos_pain_painaxis'])}。它是用 Pain-axis 自己的句子"
                 f"構建的，和對方相似本來就是預期的，所以只是較弱的證據，不用它來判斷疼痛是否突出。")
    cav = []
    if not pc.get("layer_match", True):
        cav.append(f"層不一致（對方 {pc['their_layer']}，我們 {pc['our_layer']}）")
    if pc.get("extraction_mismatch"):
        cav.append(f"取值方式不同（{pc['extraction_mismatch']}）")
    if pc.get("model_mismatch"):
        cav.append(f"模型不同（{pc['model_mismatch']}）")
    if pc.get("category_mismatch"):
        cav.append(f"句子類別不同（{pc['category_mismatch']}）")
    if cav:
        lines.append("注意的差異：" + "；".join(cav) + "，所以這個數字只能當粗略參考。")
    lines.append("")
    return lines


def render_checkpoint1(validation: pd.DataFrame, separability: dict, contamination: dict,
                       axes_info: dict, sae_tops: dict, examples_path: str,
                       pain_compare: dict | None = None, core: list[str] | None = None) -> str:
    """sae_tops: {"anger": [(feature_idx, cos), ...], "fear": [...]}; axes_info: axes.json."""
    base = os.path.basename(examples_path)
    out = ["# 檢查點 1 摘要：情緒向量夠不夠好？", "",
           "這份摘要說明：從模型裡提取出的情緒向量是否可信，決定要不要繼續做後面的調整實驗。"
           "壞消息會直接寫出來，不做美化。", ""]
    out += _auroc_section(validation, core or CORE) + _separability_section(separability)
    out += _contam_section(contamination) + _axes_section(axes_info)
    out += _sae_section(sae_tops) + _pain_section(pain_compare)
    out += ["## 七、範例文字", "",
            f"在第 18 層對每種核心情緒施加不同強度（α = 0、0.1、0.2、0.4）後，模型的實際輸出見："
            f"[{base}]({base})。這些 α 只是暫定值，正式數值會在計劃二校準。"
            f"疼痛只生成了這 12 段範例，沒有做更多。", ""]
    return "\n".join(out)
