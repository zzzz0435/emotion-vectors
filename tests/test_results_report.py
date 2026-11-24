import importlib.util
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from emolab import results_report as rr  # noqa: E402
from emolab.stats import (PRIMARY_COLS, SECONDARY_COLS, exploratory, interpret,  # noqa: E402
                          primary_tests, secondary_tests)
from test_stats import A, CORE, FROZEN, LABELS, MAP, make_scores  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts/6.8_analysis" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------- report text ----------

def _prow(h, **kw):
    base = dict(hypothesis=h, task="h1_rating", contrast=f"{h} contrast", n=30, mean_d=0.2,
                ci_low=0.1, ci_high=0.3, p=0.001, p_holm=0.004, w_plus=400.0, w_minus=60.0,
                gate=True, supported=True, untestable_reason=None, alpha_note=None)
    return {**base, **kw}


def _frame(rows, cols):
    return pd.DataFrame(rows).reindex(columns=cols)


def _tables(changed=False):
    prim = _frame([
        _prow("H1"),
        _prow("H2", task="h2", supported=False, mean_d=-0.01, p_holm=0.8, gate=False),
        _prow("H3", supported=None, untestable_reason="未能誘發", mean_d=np.nan, ci_low=np.nan,
              ci_high=np.nan, p=np.nan, p_holm=np.nan, gate=None),
        _prow("H4", mean_d=np.nan, ci_low=np.nan, ci_high=np.nan, p_holm=np.nan),
    ], PRIMARY_COLS)
    sec = _frame([
        dict(hypothesis="H5", task="h2", predicted="−", mean_d=-0.05, ci_low=-0.1, ci_high=0.0,
             p_holm=0.2, supported=False),
        dict(hypothesis="text_vs_vector", task="h1_rating", contrast="text_anger 對比 anger@+0.6000",
             text_effect=0.1, text_ci_low=0.0, text_ci_high=0.2, vector_effect=0.3,
             vector_ci_low=0.2, vector_ci_high=0.4),
    ], SECONDARY_COLS)
    if changed:
        prim["exploratory_changed_rules"] = True
        sec["exploratory_changed_rules"] = True
    return prim, sec


EXPLO = {
    "emotion_map": {"h1_rating": {"n": 20, "r2": 0.42, "effects": {},
                                  "top_deviations": [{"label": "rage", "effect": 0.5,
                                                      "predicted": 0.1, "residual": 0.4}]},
                    "h2": {"note": "有效情緒少於 5 種，無法做迴歸"}},
    "dose_response": {"h1_rating": {"anger": {"spearman_rho": 0.9}}, "h2": {"pain": {"spearman_rho": np.nan}}},
    "random_directions": {"h1_rating": {"+0.6000": {"n": 20, "p2_5": -0.1, "p97_5": 0.1}}},
    "refusal": {"h1_rating": {"baseline": 0.01, "text_anger": 0.02,
                              "anger@+0.6000": 0.05, "random00@+0.6000": 0.03}},
    "physical_harm": {"label": "探索性", "contrast": "pain − concepts", "n_items": 12, "n": 12,
                      "mean_d": 0.1, "ci_low": 0.0, "ci_high": 0.2, "p": 0.04},
}
FZ = {**FROZEN, "alpha_max": 0.6, "layer": 23,
      "residual_removed_axes": {"anger__resid3": ["V", "A"]}}
SENTENCES = ["H1（調高憤怒 → 懲罰變重）：支持。", "以上結果只描述調整模型內部方向後答案的變化。"]


def render(changed=False, explo=EXPLO, sentences=SENTENCES):
    prim, sec = _tables(changed)
    return rr.render_results(prim, sec, explo, sentences, FZ)


def test_verdict_phrases():
    md = render()
    assert "支持" in md and "不支持" in md and "無法檢驗（未能誘發）" in md


def test_sections_in_order():
    md = render()
    heads = ["## 1. 一句話結論", "## 2.", "## 3. 判讀", "## 4. 次要（單獨校正）", "## 5. 探索性結果",
             "## 6. 限制", "## 7. 圖表"]
    pos = [md.index(h) for h in heads]
    assert pos == sorted(pos)
    assert "figures/dose_response_h1_rating.png" in md


def test_exploratory_labelled():
    md = render()
    sect = md[md.index("## 5."):md.index("## 6.")]
    assert sect.count("探索性") >= 6
    for kw in ("情緒地圖", "劑量反應", "隨機方向", "拒答程度", "致死或致殘"):
        line_start = sect.rindex("###", 0, sect.index(kw))
        assert "探索性" in sect[line_start:sect.index("\n", line_start)]


def test_banner_only_when_rules_changed():
    assert rr.BANNER in render(changed=True)
    assert rr.BANNER not in render(changed=False)
    assert rr.BANNER in rr.render_results(*_tables(), {**EXPLO, "exploratory_changed_rules": True},
                                          SENTENCES, FZ)


def test_no_banned_phrases():
    for md in (render(), render(changed=True)):
        for ph in rr.BANNED_PHRASES:
            assert ph not in md


def test_nan_renders_dash_and_not_nan():
    md = render()
    assert "—" in md
    assert "nan" not in md.lower()
    assert rr.fmt(float("nan")) == "—" and rr.fmt(None) == "—" and rr.fmt(0.12345) == "0.123"


def test_h4_removed_axes_caveat():
    md = render()
    assert "超出已移除的軸之外" in md
    assert "正負面（V）、激動程度（A）" in md


def test_h4_caveat_without_frozen_record():
    prim, sec = _tables()
    md = rr.render_results(prim, sec, EXPLO, SENTENCES, {**FZ, "residual_removed_axes": {}})
    assert "超出已移除的軸之外" in md


def test_limitations_listed():
    md = render()
    for kw in ("一個模型", "英文", "4-bit", "主觀感受", "故事", "心理上的痛苦", "困惑度",
               "回答格式", "喜悅"):
        assert kw in md


def test_accepts_interpretation_markdown():
    text = "# t\n\n## 判讀（依設計）\n\n- 句子一。\n- 句子二。\n"
    assert rr.parse_interpretation(text) == ["句子一。", "句子二。"]
    md = rr.render_results(*_tables(), EXPLO, text, FZ)
    assert "句子一。" in md


def test_end_to_end_with_real_stats_outputs():
    scores = make_scores(shifts={"h1_rating": {f"anger@{A:+.4f}": 0.5}})
    prim, sec = primary_tests(scores, FROZEN, LABELS), secondary_tests(scores, FROZEN, LABELS)
    explo = exploratory(scores, FROZEN, None, None, LABELS)
    md = rr.render_results(prim, sec, explo, interpret(prim, FROZEN), FROZEN)
    assert "## 6. 限制" in md and "nan" not in md.lower()
    assert not any(p in md for p in rr.BANNED_PHRASES)


# ---------- figures ----------

@pytest.fixture(scope="module")
def synth():
    scores = make_scores(shifts={"h1_rating": {f"anger@{A:+.4f}": 0.5, "anger@+0.7000": 0.5}},
                         n=12)
    prim, sec = primary_tests(scores, FROZEN, LABELS), secondary_tests(scores, FROZEN, LABELS)
    rng = np.random.default_rng(1)
    emos = CORE + MAP
    vad = pd.DataFrame({"label": emos, "valence": rng.uniform(0, 1, len(emos)),
                        "arousal": rng.uniform(0, 1, len(emos)),
                        "dominance": rng.uniform(0, 1, len(emos))})
    explo = exploratory(scores, FROZEN, vad, None, LABELS)
    return scores, prim, sec, explo, vad


def _ok(p, name):
    assert p is not None and p.name == name and p.exists() and p.stat().st_size > 1000


@pytest.mark.parametrize("font", ["Microsoft JhengHei", None])
def test_figures_smoke(synth, tmp_path, font):
    fig = _load("02_figures")
    scores, prim, sec, explo, vad = synth
    for task in ("h1_rating", "h2"):
        _ok(fig.fig_dose_response(scores, FROZEN, LABELS, task, tmp_path, font),
            f"dose_response_{task}.png")
        _ok(fig.fig_emotion_map(explo, vad, task, tmp_path, LABELS, font), f"emotion_map_{task}.png")
    _ok(fig.fig_anger_vs_fear(prim, tmp_path, font), "anger_vs_fear_eq.png")
    _ok(fig.fig_text_vs_vector(sec, tmp_path, LABELS, font), "text_vs_vector.png")
    _ok(fig.fig_refusal(explo, FROZEN, LABELS, tmp_path, font), "refusal.png")


def test_make_all_writes_every_linked_figure(synth, tmp_path):
    fig = _load("02_figures")
    scores, prim, sec, explo, vad = synth
    paths = fig.make_all(scores, FROZEN, LABELS, prim, sec, explo, vad, tmp_path / "figures", None)
    assert sorted(p.name for p in paths) == sorted(Path(p).name for _, p in rr.FIGURES)


def test_font_fallback_chain():
    fig = _load("02_figures")
    assert fig.pick_font(["Arial", "Microsoft YaHei", "Microsoft JhengHei"]) == "Microsoft JhengHei"
    assert fig.pick_font(["Arial", "Microsoft YaHei"]) == "Microsoft YaHei"
    assert fig.pick_font(["Arial"]) is None
    assert fig.make_tr(None)("效果", "effect") == "effect"
    assert fig.make_tr("Microsoft YaHei")("效果", "effect") == "效果"


def test_emotion_colours_distinct_and_cover_core():
    fig = _load("02_figures")
    assert set(CORE) <= set(fig.EMO_COLORS)
    assert len(set(fig.EMO_COLORS.values())) == len(fig.EMO_COLORS)


def test_emotion_map_skipped_without_vad(synth, tmp_path):
    fig = _load("02_figures")
    _, _, _, explo, _ = synth
    assert fig.fig_emotion_map(explo, None, "h2", tmp_path, LABELS, None) is None


def test_03_refuses_when_inputs_missing(tmp_path):
    mod = _load("03_results_summary")
    frozen = tmp_path / "frozen.json"
    with pytest.raises(SystemExit, match="凍結檔"):
        mod.main(["--frozen", str(frozen), "--out-dir", str(tmp_path)])
    frozen.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit, match="primary.csv"):
        mod.main(["--frozen", str(frozen), "--out-dir", str(tmp_path)])


def test_03_writes_summary(tmp_path):
    import json
    mod = _load("03_results_summary")
    prim, sec = _tables()
    prim.to_csv(tmp_path / "primary.csv", index=False)
    sec.to_csv(tmp_path / "secondary.csv", index=False)
    clean = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in EXPLO.items()}
    (tmp_path / "exploratory.json").write_text(
        json.dumps(clean, ensure_ascii=False, default=lambda o: None), encoding="utf-8")
    (tmp_path / "interpretation.md").write_text("## 判讀\n\n- 句子一。\n", encoding="utf-8")
    frozen = tmp_path / "frozen.json"
    frozen.write_text(json.dumps({**FZ}), encoding="utf-8")
    mod.main(["--frozen", str(frozen), "--out-dir", str(tmp_path)])
    text = (tmp_path / "results_summary.md").read_text(encoding="utf-8")
    assert "無法檢驗（未能誘發）" in text and "句子一。" in text


# ---------- fix round 1 ----------

def test_banner_switches_headline_and_section_title_to_exploratory():
    on, off = render(changed=True), render(changed=False)
    assert "在預先登記的四個假設中" not in on and "四個主要假設" not in on
    assert "（探索性）四個假設的結果" in on and "## 2. （探索性）四個假設" in on
    assert "在預先登記的四個假設中" in off and "## 2. 四個主要假設" in off


def test_concept_count_not_hardcoded(monkeypatch):
    monkeypatch.setattr(rr, "load_labels", lambda: {**LABELS, "concepts": [f"c{i}" for i in range(7)]})
    md = render()
    assert "調 7 個無關概念向量" in md and "「7 個無關概念向量的平均」" in md
    assert "40 個" not in md


def test_control_ranges_only_where_data_exist(synth):
    fig = _load("02_figures")
    scores = synth[0]
    r = fig.control_ranges(scores, "h1_rating", LABELS, A)
    assert set(r["concept"]) == {-A, A}                       # nothing at +-A/2 or 0
    assert set(r["random"]) == {-A, -A / 2, A / 2, A}         # nothing at 0
    # drop random conditions at +A/2: that alpha must disappear, not be bridged
    from emolab.conditions import condition_name
    cut = scores["h1_rating"].drop(columns=[condition_name(f"random{i:02d}", A / 2) for i in range(10)])
    r2 = fig.control_ranges({**scores, "h1_rating": cut}, "h1_rating", LABELS, A)
    assert A / 2 not in r2["random"] and -A / 2 in r2["random"]


def test_emotion_map_survives_null_effects_roundtrip(synth, tmp_path):
    import json
    fig = _load("02_figures")
    _, _, _, explo, vad = synth
    explo = json.loads(json.dumps(explo, allow_nan=False, default=lambda o: None)) \
        if False else explo
    effects = dict(explo["emotion_map"]["h1_rating"]["effects"])
    effects[CORE[0]] = float("nan")
    effects[CORE[1]] = float("nan")
    bad = {"emotion_map": {"h1_rating": {**explo["emotion_map"]["h1_rating"], "effects": effects}}}
    import math
    clean = {"emotion_map": {"h1_rating": {**bad["emotion_map"]["h1_rating"], "effects": {
        k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in effects.items()}}}}
    loaded = json.loads(json.dumps(clean, allow_nan=False))
    assert loaded["emotion_map"]["h1_rating"]["effects"][CORE[0]] is None
    _ok(fig.fig_emotion_map(loaded, vad, "h1_rating", tmp_path, LABELS, None), "emotion_map_h1_rating.png")
    all_null = {"emotion_map": {"h2": {"effects": {e: None for e in effects}}}}
    assert fig.fig_emotion_map(all_null, vad, "h2", tmp_path, LABELS, None) is None


def test_make_all_with_missing_conditions(synth, tmp_path, capsys):
    fig = _load("02_figures")
    scores, prim, sec, explo, vad = synth
    keep = {t: (df.drop(columns=[c for c in df.columns if c.startswith(("text_fear", "fear@", "joy@"))
                                 or c.startswith("random03")]) if isinstance(df, pd.DataFrame) else df)
            for t, df in scores.items()}
    paths = fig.make_all(keep, FROZEN, LABELS, prim, sec, explo, vad, tmp_path, None)
    assert paths and all(p.exists() for p in paths)
    # no baseline at all: figures skip or draw, never raise
    nobase = {t: df.drop(columns=["baseline"]) for t, df in scores.items()}
    fig.make_all(nobase, FROZEN, LABELS, prim, sec, explo, vad, tmp_path / "nb", None)
    # an unexpected error inside one figure is reported, not fatal
    out = fig._safe("boom", lambda: 1 / 0)
    assert out is None and "略過圖 boom" in capsys.readouterr().out


# ---------- final-review wave ----------

REAL_AXES = {"anger__resid3": ["V", "D"], "anger__resid2": ["V"],
             "pain__resid3": ["V", "A", "D"], "compassion__resid3": ["V", "A", "D"]}


def _sec_with(**kw):
    prim, sec = _tables()
    row = dict(hypothesis="pain_painaxis_vs_pain", task="h2", predicted="two-sided", mean_d=0.02,
               ci_low=-0.01, ci_high=0.05, p_holm=0.3, supported=False)
    return prim, pd.concat([sec, _frame([{**row, **kw}], SECONDARY_COLS)], ignore_index=True)


def test_item_counts_column_and_explanation():
    md = render()
    assert "有效題數" in md
    assert f"30 / {rr.n_items('h1_rating')}" in md and f"30 / {rr.n_items('h2')}" in md
    assert (rr.n_items("h1_rating"), rr.n_items("h2")) == (72, 150)
    assert "沒有任何有效回答的題目會被排除" in md and "差為 0" in md
    h1_line = next(ln for ln in md.splitlines() if ln.startswith("- **H1**"))
    assert "有效題數 30 / 72" in h1_line
    h2_line = next(ln for ln in md.splitlines() if ln.startswith("- **H2**"))
    assert "有效題數 30 / 150" in h2_line


def test_anger_resid_label_comes_from_frozen():
    prim, sec = _tables()
    sec = pd.concat([sec, _frame([dict(hypothesis="anger_resid2_vs_resid3", task="h1_rating",
                                       predicted="+", mean_d=0.1, ci_low=0.0, ci_high=0.2,
                                       p_holm=0.1, supported=False)], SECONDARY_COLS)],
                    ignore_index=True)
    md = rr.render_results(prim, sec, EXPLO, SENTENCES, {**FZ, "residual_removed_axes": REAL_AXES})
    assert "只扣 V（anger__resid2）vs 扣 V、D（anger__resid3）：看掌控感（D）的貢獻" in md
    md2 = rr.render_results(prim, sec, EXPLO, SENTENCES, {**FZ, "residual_removed_axes": {
        "anger__resid3": ["V", "A", "D"], "anger__resid2": ["V", "A"]}})
    assert "只扣 V、A（anger__resid2）vs 扣 V、A、D（anger__resid3）" in md2
    # no record in the frozen file: falls back to the generic name
    md3 = rr.render_results(prim, sec, EXPLO, SENTENCES, {**FZ, "residual_removed_axes": {}})
    assert "憤怒只扣兩個軸 vs 扣三個軸" in md3


def test_two_sided_rows_never_say_support():
    _, sec = _sec_with(supported=True, p_holm=0.001)
    assert rr.verdict(sec.iloc[-1]) == "有顯著差異"
    _, sec = _sec_with(supported=False)
    assert rr.verdict(sec.iloc[-1]) == "沒有偵測到差異"
    prim, sec = _sec_with(supported=True)
    md = rr.render_results(prim, sec, EXPLO, SENTENCES, FZ)
    line = next(ln for ln in md.splitlines() if ln.startswith("| Pain-axis"))
    assert "有顯著差異" in line and "支持" not in line
    # a directional row keeps the old words
    assert rr.verdict(pd.Series({"supported": True, "predicted": "+"})) == "支持"


def test_r31_wording_is_accurate():
    md = render()
    assert "在看到道德題資料之前加入" not in md
    assert ("在正式實驗開始前，依先導試跑（3 個條件、10 題）的無效回答比例加入，只規範回答格式；"
            "沒有參考先導的懲罰或傷害分數。") in md


def test_conclusion_is_one_sentence_and_rules_stay_in_section3():
    md = render()
    sect = md[md.index("## 1."):md.index("## 2.")]
    body = sect.split("\n\n")[1]
    assert body.count("。") == 1 and body.endswith("。") and "）。" in body
    assert "H1 成立" not in body and "（）" not in body
    assert "H1（調高憤怒 → 懲罰變重）：支持。" in md[md.index("## 3."):md.index("## 4.")]


def test_units_next_to_effects():
    md = render()
    assert "分（1–7 分量表）" in md and "機率（0–1，百分點）" in md
    assert rr.fmt_eff(0.2, "h1_rating") == "+0.200 分"
    assert rr.fmt_eff(0.012, "h2") == "+0.012 機率（+1.2 百分點）"
    assert rr.fmt_eff(float("nan"), "h2") == "—"
    assert "（單位：機率（0–1，百分點））" not in md       # random ranges only list h1_rating here
    assert "（單位：分（1–7 分量表））" in md


def test_cross_combinations_and_auxiliary_choice_shown():
    a = FZ["alpha_max"]
    cross = {t: {f"{e}@{s * a:+.4f}": {"effect": 0.05 * (1 if s > 0 else -1), "ci_low": -0.1,
                                      "ci_high": 0.1, "n": 70}
                 for e in CORE for s in (1, -1)} for t in ("h1_rating", "h1_choice", "h2")}
    md = render(explo={**EXPLO, "cross_combinations": cross})
    sect = md[md.index("### 探索性：輔助指標"):md.index("## 6.")]
    assert "輔助指標" in sect and "H1 二選一" in sect
    assert "核心情緒 × 任務的交叉結果" in sect
    for e in CORE:
        assert f"| {e} |" in sect
    assert "+0.050 機率（+5.0 百分點）" in sect and "-0.050 機率（-5.0 百分點）" in sect
    # without the data the subsections still render, with a dash
    md0 = render()
    assert "### 探索性：輔助指標" in md0 and "核心情緒 × 任務的交叉結果" in md0


def test_limitations_come_from_frozen():
    fz = {**FZ, "model": "Qwen/Qwen3-8B", "layer": 31, "alpha_eq_target": 1.5,
          "alpha_eq": {"anger": 0.4, "joy": 0.7}, "coherent_max": {"anger": 0.6, "joy": 0.7}}
    md = rr.render_results(*_tables(), EXPLO, SENTENCES, fz)
    assert "只測了一個模型（Qwen/Qwen3-8B）" in md and "4-bit" in md
    assert "受限於喜悅" in md and "1.50" in md and "層 31" in md
    # unknown model, no coherence data: generic wording, nothing invented
    fz2 = {**FZ, "model": "Some_Other_Model", "alpha_eq_target": 1.5, "coherent_max": 2.0}
    md2 = rr.render_results(*_tables(), EXPLO, SENTENCES, fz2)
    assert "Some_Other_Model" in md2 and "4-bit" not in md2
    assert "受限於喜悅" not in md2 and "誘發增幅最小的那種情緒" in md2
