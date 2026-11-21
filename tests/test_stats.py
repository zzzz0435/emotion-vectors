import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from emolab import integrity
from emolab.conditions import build_conditions, condition_name, load_labels
from emolab.stats import (beats_controls, bootstrap_ci, exploratory, holm, interpret,
                          load_score_table, primary_tests, secondary_tests,
                          signed_rank_sums, wilcoxon_two_sided)

REPO = Path(__file__).resolve().parents[1]
STATS_PY = REPO / "scripts/emolab/stats.py"
CONDITIONS_PY = REPO / "scripts/emolab/conditions.py"
LABELS = load_labels()
CORE, MAP, CONCEPTS = LABELS["core"], LABELS["map"], LABELS["concepts"]
RESIDS = ("anger__resid3", "anger__resid2", "pain__resid3", "compassion__resid3")
A = 1.0
EQ = {"anger": 0.7, "fear": 0.9, "sadness": 0.8, "joy": 0.6, "compassion": 0.75, "pain": 0.85}
FROZEN = {
    "model": "Qwen/Qwen3-8B", "layer": 18, "resid_norm": 100.0, "alpha_max": A,
    "alpha_eq": dict(EQ), "alpha_eq_target": 1.0, "alpha_individual": {},
    "induction": {e: "ok" for e in CORE}, "coherent_max": 2.0, "thresholds": {}, "grid": [],
    "contamination_at_layer": [], "separability_at_layer": {"auroc": 0.9, "passed": True},
    "residual_status_at_layer": {r: "ok" for r in RESIDS},
    "residual_removed_axes": {"anger__resid3": ["V", "A", "D"], "anger__resid2": ["V", "A"],
                              "pain__resid3": ["V", "A", "D"],
                              "compassion__resid3": ["V", "A", "D"]},
    "stats_sha256": "x", "conditions_sha256": "y", "git_commit": "z", "created_at": "t"}
TASKS = ("h1_rating", "h1_choice", "h2")
N = 30


def frozen(**kw):
    return {**FROZEN, **kw}


def make_scores(fz=FROZEN, shifts=None, n=N, seed=0, noise=0.05, items=None):
    """Synthetic ScoreTable: every condition of `fz` on every task; value = item base
    + shifts[task][cond] (scalar or per-item array) + independent noise."""
    rng = np.random.default_rng(seed)
    shifts = shifts or {}
    conds = [c.name for c in build_conditions(fz, LABELS)]
    out = {}
    for task in TASKS:
        idx = items if items is not None else [f"{task}_{i:03d}" for i in range(n)]
        base = rng.normal(4.0, 1.0, len(idx))
        sh = shifts.get(task, {})
        out[task] = pd.DataFrame(
            {c: base + np.broadcast_to(sh.get(c, 0.0), len(idx))
             + (0.0 if c == "baseline" else rng.normal(0, noise, len(idx))) for c in conds},
            index=pd.Index(idx, name="item_id"))
        out[f"invalid_{task}"] = pd.DataFrame(
            {c: rng.uniform(0, 0.1, len(idx)) for c in conds}, index=out[task].index)
    return out


def row(df, h):
    return df.set_index("hypothesis").loc[h]


def nm(label, alpha=A, eq=False):
    return condition_name(label, alpha, eq)


PLANT = {"h1_rating": {nm("anger"): 1.0, nm("anger__resid3"): 1.0,
                       nm("anger", EQ["anger"], True): 1.0},
         "h2": {nm("pain"): 1.0}}


# ---------- primitives ----------

def test_holm_matches_reference():
    assert holm({"a": 0.01, "b": 0.04, "c": 0.03}) == pytest.approx(
        {"a": 0.03, "b": 0.06, "c": 0.06})


def test_holm_caps_at_one_and_keeps_nan():
    out = holm({"a": 0.6, "b": 0.7, "c": float("nan")})
    assert out["a"] == pytest.approx(1.0) and out["b"] == pytest.approx(1.0)
    assert np.isnan(out["c"])


def test_wilcoxon_drops_nan_pairs():
    p, n = wilcoxon_two_sided(np.array([1.0, 2.0, np.nan, 3.0, 4.0, 5.0, 6.0]))
    assert n == 6 and p < 0.05


def test_wilcoxon_degenerate_is_nan():
    p, n = wilcoxon_two_sided(np.zeros(5))
    assert np.isnan(p) and n == 0
    p, n = wilcoxon_two_sided(np.array([np.nan, np.nan]))
    assert np.isnan(p) and n == 0


def test_wilcoxon_n_excludes_zero_differences():
    p, n = wilcoxon_two_sided(np.array([0, 0, 0, 1, 2, 3, 4, 5, 6.0]))
    assert n == 6 and p < 0.05


SKEW = np.r_[-0.01 - 0.0001 * np.arange(60), 0.5 + 0.01 * np.arange(10)]  # mean > 0, W+ < W−


def test_signed_rank_sums_on_skewed_differences():
    p, n = wilcoxon_two_sided(SKEW)
    assert SKEW.mean() > 0.05 and p < 0.01 and n == 70
    assert signed_rank_sums(np.append(SKEW, [0.0, np.nan])) == (655.0, 1830.0)


def test_bootstrap_ci_brackets_mean_and_is_deterministic():
    d = np.random.default_rng(1).normal(0.5, 1.0, 30)
    lo, hi = bootstrap_ci(np.append(d, np.nan))
    assert lo < d.mean() < hi
    assert (lo, hi) == bootstrap_ci(d)


def test_gate_needs_38_of_40():
    # Spec §6.8 "至少贏過其中 38 個": 0.375 > 0.00..0.37 = 38 controls -> passes;
    # 0.365 beats 37 -> fails. (The brief's example used 0.375 as the failing case:
    # off by one against its own interface and the spec.)
    c = np.arange(40) / 100.0
    assert beats_controls(0.385, c) and beats_controls(0.375, c)
    assert not beats_controls(0.365, c) and not beats_controls(0.375, c, need=39)
    assert not beats_controls(float("nan"), c)


# ---------- score loading ----------

def _write(path, rows):
    pd.DataFrame(rows, columns=["item_id", "variant", "p_target", "invalid"]).to_csv(
        path, index=False)


def test_load_score_table_nan_if_any_variant_nan(tmp_path):
    _write(tmp_path / "h1_rating__baseline.csv", [
        ("i1", "light1", 3.0, 0.1), ("i1", "harsh1", 5.0, 0.3),
        ("i2", "light1", 4.0, 0.1), ("i2", "harsh1", np.nan, 0.1),
        ("i3", "light1", 4.0, 0.2)])                      # harsh1 row missing
    _write(tmp_path / "h1_rating__anger__resid3@+1.0000.csv", [
        ("i1", "light1", 1.0, 0.0), ("i1", "harsh1", 2.0, 0.0)])
    _write(tmp_path / "h2__pain@+1.0000.csv", [("G_1", 0, 0.2, 0.0), ("G_1", 1, 0.4, 0.2)])
    st = load_score_table(tmp_path)
    t = st["h1_rating"]
    assert t.loc["i1", "baseline"] == pytest.approx(4.0)
    assert np.isnan(t.loc["i2", "baseline"]) and np.isnan(t.loc["i3", "baseline"])
    assert t.loc["i1", "anger__resid3@+1.0000"] == pytest.approx(1.5)
    assert np.isnan(t.loc["i2", "anger__resid3@+1.0000"])
    assert st["invalid_h1_rating"].loc["i1", "baseline"] == pytest.approx(0.2)
    assert st["h2"].loc["G_1", "pain@+1.0000"] == pytest.approx(0.3)


def test_load_score_table_rejects_unknown_variant(tmp_path):
    _write(tmp_path / "h2__baseline.csv", [("G_1", 0, 0.2, 0.0), ("G_1", 2, 0.4, 0.0)])
    with pytest.raises(ValueError):
        load_score_table(tmp_path)


# ---------- primary ----------

def test_primary_h1_supported_on_planted_effect():
    p = primary_tests(make_scores(shifts=PLANT), FROZEN)
    assert list(p["hypothesis"]) == ["H1", "H2", "H3", "H4"]
    for h in ("H1", "H2", "H4"):
        r = row(p, h)
        assert r["supported"] is True and r["gate"] is True
        assert r["n"] == N and r["mean_d"] == pytest.approx(1.0, abs=0.1)
        assert r["ci_low"] < r["mean_d"] < r["ci_high"] and r["p_holm"] < 0.05
        assert r["w_plus"] > r["w_minus"]
    h3 = row(p, "H3")
    assert h3["supported"] is True and h3["gate"] is None
    assert h3["x_effect"] > 0.5 and h3["x_ci_low"] > 0.5 and abs(h3["y_effect"]) < 0.1
    assert h3["y_ci_low"] <= h3["y_effect"] <= h3["y_ci_high"]
    for col in ("hypothesis", "task", "contrast", "n", "mean_d", "ci_low", "ci_high", "p",
                "p_holm", "w_plus", "w_minus", "gate", "supported", "untestable_reason",
                "alpha_note", "clean_n", "clean_mean_d", "clean_p", "clean_w_plus",
                "clean_w_minus", "clean_gate", "clean_pass"):
        assert col in p.columns


def test_supported_requires_rank_direction():
    # R22: mean_d > 0 and Holm p < .05, but the signed ranks point the other way (W+ < W−)
    st = make_scores(shifts={"h1_rating": {nm("anger"): SKEW}}, n=len(SKEW), noise=0.0)
    p = primary_tests(st, FROZEN)
    h1 = row(p, "H1")
    assert h1["mean_d"] > 0 and h1["p_holm"] < 0.05 and h1["gate"] is True
    assert (h1["w_plus"], h1["w_minus"]) == (655.0, 1830.0)
    assert h1["supported"] is False
    assert "W+ ≤ W−" in "\n".join(interpret(p, FROZEN))


def test_gate_alone_flips_supported():
    shifts = {"h1_rating": {nm("anger"): 1.0, **{nm(c): 10.0 for c in CONCEPTS[:3]}}}
    h1 = row(primary_tests(make_scores(shifts=shifts), FROZEN), "H1")
    assert h1["p_holm"] < 0.05 and h1["mean_d"] > 0 and h1["w_plus"] > h1["w_minus"]
    assert h1["gate"] is False and h1["supported"] is False   # beats only 37 concepts


def test_nan_items_reduce_n():
    st = make_scores(shifts=PLANT)
    items = st["h1_rating"].index
    st["h1_rating"].loc[items[:5], nm("anger")] = np.nan
    st["h1_rating"].loc[items[5], [nm(c) for c in CONCEPTS[:10]]] = np.nan  # item kept
    h1 = row(primary_tests(st, FROZEN), "H1")
    assert h1["n"] == N - 5 and h1["supported"] is True


def test_primary_h1_not_supported_when_concepts_shift_equally():
    st = make_scores(FROZEN)
    for task in TASKS:
        cols = [c for c in st[task].columns if c != "baseline"]
        st[task][cols] += 1.0
    p = primary_tests(st, FROZEN)
    h1 = row(p, "H1")
    assert abs(h1["mean_d"]) < 0.1
    assert h1["supported"] is False and h1["gate"] is False


def test_untestable_when_induction_failed():
    fz = frozen(induction={**FROZEN["induction"], "pain": "failed"})
    p = primary_tests(make_scores(fz, shifts=PLANT), fz)
    h2 = row(p, "H2")
    assert h2["supported"] is None and h2["untestable_reason"] == "未能誘發"
    assert np.isnan(h2["p_holm"])
    assert row(p, "H1")["supported"] is True


def test_untestable_when_column_missing():
    st = make_scores(shifts=PLANT)
    st["h1_rating"] = st["h1_rating"].drop(columns=[nm(CONCEPTS[0])])
    p = primary_tests(st, FROZEN)
    assert row(p, "H1")["untestable_reason"] == "缺少條件資料"
    assert row(p, "H1")["supported"] is None


def test_h3_untestable_when_not_separable():
    fz = frozen(separability_at_layer={"auroc": 0.6, "passed": False})
    p = primary_tests(make_scores(fz, shifts=PLANT), fz)
    h3 = row(p, "H3")
    assert h3["supported"] is None
    assert h3["untestable_reason"] == "模型內部無法區分憤怒與恐懼"
    expect = holm({h: row(p, h)["p"] for h in ("H1", "H2", "H4")})
    for h in ("H1", "H2", "H4"):
        assert row(p, h)["p_holm"] == pytest.approx(expect[h])


def test_h3_untestable_when_alpha_eq_missing():
    fz = frozen(alpha_eq={**EQ, "fear": None})
    p = primary_tests(make_scores(fz, shifts=PLANT), fz)
    assert row(p, "H3")["untestable_reason"] == "未能誘發"


def test_h4_untestable_when_residual_not_ok():
    fz = frozen(residual_status_at_layer={**FROZEN["residual_status_at_layer"],
                                          "anger__resid3": "untestable"})
    p = primary_tests(make_scores(fz, shifts=PLANT), fz)
    h4 = row(p, "H4")
    assert h4["supported"] is None and h4["untestable_reason"] == "正負面軸不可靠"


def test_fallback_uses_individual_alpha():
    fz = frozen(induction={**FROZEN["induction"], "pain": "fallback"},
                alpha_individual={"pain": 1.6})
    st = make_scores(fz, shifts={"h2": {nm("pain", 1.6): 1.0}})
    p = primary_tests(st, fz)
    h2 = row(p, "H2")
    assert "+1.6000" in h2["contrast"]
    assert h2["alpha_note"] == "強度不同（備案強度）"
    assert h2["supported"] is True and h2["mean_d"] == pytest.approx(1.0, abs=0.1)
    assert row(p, "H1")["alpha_note"] is None


def test_clean_version_must_also_pass():
    fz = frozen(contamination_at_layer=["anger"])
    p = primary_tests(make_scores(fz, shifts=PLANT), fz)   # anger__clean has no effect
    h1 = row(p, "H1")
    assert h1["p_holm"] < 0.05 and h1["gate"] is True and h1["mean_d"] > 0.5
    assert h1["clean_pass"] is False and h1["supported"] is False
    assert abs(h1["clean_mean_d"]) < 0.1 and h1["clean_n"] == N
    assert row(p, "H2")["supported"] is True and row(p, "H2")["clean_pass"] is None

    shifts = {**PLANT, "h1_rating": {**PLANT["h1_rating"], nm("anger__clean"): 1.0,
                                     nm("anger__clean__resid3"): 1.0,
                                     nm("anger__clean", EQ["anger"], True): 1.0}}
    p = primary_tests(make_scores(fz, shifts=shifts), fz)
    for h in ("H1", "H3", "H4"):
        assert row(p, h)["clean_pass"] is True and row(p, h)["supported"] is True
        assert row(p, h)["clean_gate"] is (None if h == "H3" else True)
        assert row(p, h)["clean_w_plus"] > row(p, h)["clean_w_minus"]


def test_fallback_contaminated_clean_uses_individual_alpha():
    # R20: anger fallback + contaminated -> clean is anger__clean@α_ind vs concepts@α_ind
    fz = frozen(induction={**FROZEN["induction"], "anger": "fallback"},
                alpha_individual={"anger": 1.6}, contamination_at_layer=["anger"])
    shifts = {"h1_rating": {nm("anger", 1.6): 1.0, nm("anger__clean", 1.6): 1.0,
                            **{nm(c): 5.0 for c in CONCEPTS}}}   # concepts@α_max would swamp it
    h1 = row(primary_tests(make_scores(fz, shifts=shifts), fz), "H1")
    assert h1["untestable_reason"] is None and "+1.6000" in h1["contrast"]
    assert h1["clean_n"] == N and h1["clean_mean_d"] == pytest.approx(1.0, abs=0.1)
    assert h1["clean_pass"] is True and h1["supported"] is True


# ---------- secondary ----------

def test_secondary_rows_and_text_vs_vector():
    s = secondary_tests(make_scores(), FROZEN)
    hyps = list(s["hypothesis"])
    for h in ("H5", "compassion_resid3", "anger_vs_sadness", "pain_resid3",
              "anger_resid2_vs_resid3", "pain_painaxis_vs_pain"):
        assert hyps.count(h) == 1
    tv = s[s["hypothesis"] == "text_vs_vector"]
    assert len(tv) == len(CORE) * 2
    assert tv["p"].isna().all() and tv["p_holm"].isna().all()
    assert {"text_effect", "vector_effect", "text_ci_low", "vector_ci_high"} <= set(s.columns)
    assert row(s, "pain_painaxis_vs_pain")["predicted"] == "two-sided"


def test_secondary_negative_prediction_gate():
    r = nm("compassion__resid3")
    s = secondary_tests(make_scores(shifts={"h2": {r: -1.0}}), FROZEN)
    c = row(s, "compassion_resid3")
    assert c["predicted"] == "−" and c["gate"] is True and c["supported"] is True
    s = secondary_tests(make_scores(shifts={"h2": {r: 1.0}}), FROZEN)
    c = row(s, "compassion_resid3")
    assert c["gate"] is False and c["supported"] is False and c["p_holm"] < 0.05


def test_secondary_untestable_rules():
    fz = frozen(induction={**FROZEN["induction"], "joy": "failed"},
                residual_status_at_layer={**FROZEN["residual_status_at_layer"],
                                          "pain__resid3": "untestable"})
    s = secondary_tests(make_scores(fz), fz)
    assert row(s, "H5")["untestable_reason"] == "未能誘發"
    assert row(s, "pain_resid3")["untestable_reason"] == "正負面軸不可靠"
    assert row(s, "H5")["supported"] is None and np.isnan(row(s, "H5")["p_holm"])


def test_secondary_sign_conventions():
    shifts = {"h2": {nm("compassion", EQ["compassion"], True): -1.0, nm("pain__resid3"): 1.0},
              "h1_rating": {nm("anger", EQ["anger"], True): 1.0, nm("anger__resid2"): 1.0}}
    s = secondary_tests(make_scores(shifts=shifts), FROZEN)
    for h, sign in (("H5", -1), ("anger_vs_sadness", 1), ("pain_resid3", 1),
                    ("anger_resid2_vs_resid3", 1)):
        r = row(s, h)
        assert np.sign(r["mean_d"]) == sign and r["supported"] is True
        assert np.sign(r["w_plus"] - r["w_minus"]) == sign
    assert row(s, "pain_resid3")["gate"] is True
    flipped = {"h2": {nm("compassion", EQ["compassion"], True): 1.0},
               "h1_rating": {nm("anger", EQ["anger"], True): -1.0}}
    s = secondary_tests(make_scores(shifts=flipped), FROZEN)
    for h in ("H5", "anger_vs_sadness"):
        assert row(s, h)["p_holm"] < 0.05 and row(s, h)["supported"] is False


def test_secondary_clean_variants():
    # R21: clean variant where it can be built; resid rows of a contaminated emotion untestable
    fz = frozen(contamination_at_layer=["anger", "compassion", "pain"])
    shifts = {"h2": {nm("compassion", EQ["compassion"], True): -1.0,
                     nm("compassion__clean", EQ["compassion"], True): -1.0,
                     nm("pain_painaxis"): 1.0},
              "h1_rating": {nm("anger", EQ["anger"], True): 1.0}}   # anger__clean: no effect
    s = secondary_tests(make_scores(fz, shifts=shifts), fz)
    h5 = row(s, "H5")
    assert h5["clean_n"] == N and h5["clean_mean_d"] < -0.5
    assert h5["clean_pass"] is True and h5["supported"] is True
    avs = row(s, "anger_vs_sadness")
    assert avs["p_holm"] < 0.05 and avs["clean_pass"] is False and avs["supported"] is False
    pp = row(s, "pain_painaxis_vs_pain")       # two-sided: Bonferroni p × 3 only
    assert pp["clean_mean_d"] > 0.5 and pp["clean_p"] * 3 < 0.05 and pp["clean_pass"] is True
    for h in ("compassion_resid3", "pain_resid3", "anger_resid2_vs_resid3"):
        r = row(s, h)
        assert r["untestable_reason"] == "受汙染，未建立乾淨版" and r["supported"] is None
        assert np.isnan(r["p_holm"])
    testable = s[(s["hypothesis"] != "text_vs_vector") & s["untestable_reason"].isna()]
    assert len(testable) == 3
    assert row(s, "H5")["clean_pass"] is True and row(secondary_tests(
        make_scores(fz, shifts=shifts), frozen()), "H5")["clean_pass"] is None


# ---------- exploratory ----------

def _vad():
    rng = np.random.default_rng(3)
    labels = CORE + MAP
    v = rng.uniform(-1, 1, 20)
    v[labels.index("anger")] = 0.5          # anger@+α_max effect = 2·0.5 = 1 (dose top)
    return pd.DataFrame({"label": labels, "valence": v, "arousal": rng.uniform(-1, 1, 20),
                         "dominance": rng.uniform(-1, 1, 20)})


def test_exploratory_map_dose_random_refusal():
    vad = _vad()
    v = vad.set_index("label")["valence"]
    shifts = {"h1_rating": {nm(e): 2.0 * v[e] for e in CORE + MAP}}
    shifts["h1_rating"].update({nm("anger", s * A): s for s in (-1, -0.5, 0.5, 1)})
    ex = exploratory(make_scores(shifts=shifts), FROZEN, vad)
    m = ex["emotion_map"]["h1_rating"]
    assert m["r2"] > 0.9 and len(m["top_deviations"]) == 3
    assert ex["dose_response"]["h1_rating"]["anger"]["spearman_rho"] == pytest.approx(1.0)
    rnd = ex["random_directions"]["h1_rating"]
    assert set(rnd) == {f"{a:+.4f}" for a in (-A, -A / 2, A / 2, A)}
    assert rnd[f"{A:+.4f}"]["n"] == 20
    assert ex["refusal"]["h2"]["baseline"] == pytest.approx(0.05, abs=0.03)
    assert nm("fear", -A) in ex["cross_combinations"]["h2"]
    assert "note" in exploratory(make_scores(), FROZEN, None)["emotion_map"]


def test_physical_harm_subset():
    ids = [f"G_{i}" for i in range(N)]
    phys = np.array([i % 3 == 0 for i in range(N)])
    harm = [["death", "pain"] if p else ["pain"] for p in phys]
    harm[3] = ["disable"]
    h2_items = pd.DataFrame({"id": ids, "harm_types": harm})
    st = make_scores(shifts={"h2": {nm("pain"): phys * 1.0}}, items=ids)
    ex = exploratory(st, FROZEN, None, h2_items=h2_items)
    ph = ex["physical_harm"]
    assert ph["n"] == phys.sum() and ph["mean_d"] == pytest.approx(1.0, abs=0.1)
    assert ph["p"] < 0.05 and ph["label"] == "探索性"
    assert "note" in exploratory(st, FROZEN, None)["physical_harm"]


# ---------- interpretation ----------

def _prim(rows=None, **status):
    """Synthetic primary table. status: H1..H4 -> True/False/None, f"{h}_reason";
    rows: {h: {column: value}} overrides."""
    out = []
    for h in ("H1", "H2", "H3", "H4"):
        s = status.get(h, False)
        r = {"hypothesis": h, "task": "h1_rating", "contrast": "", "n": 30,
             "mean_d": 0.5 if s else -0.1, "ci_low": 0.1, "ci_high": 0.9,
             "p": 0.001 if s else 0.5, "p_holm": 0.004 if s else 0.9,
             "w_plus": 400.0 if s else 100.0, "w_minus": 65.0 if s else 365.0,
             "gate": None if h == "H3" else bool(s), "supported": s,
             "untestable_reason": status.get(h + "_reason"), "alpha_note": None,
             **{f"clean_{k}": np.nan for k in ("n", "mean_d", "p", "w_plus", "w_minus")},
             "clean_gate": None, "clean_pass": None,
             "x_effect": 0.6, "x_ci_low": 0.4, "x_ci_high": 0.8,
             "y_effect": 0.1, "y_ci_low": -0.1, "y_ci_high": 0.3}
        r.update((rows or {}).get(h, {}))
        out.append(r)
    return pd.DataFrame(out)


def J(prim, fz=FROZEN):
    return "\n".join(interpret(prim, fz))


FORBIDDEN = ("AI 會生氣", "AI 會痛", "模型會生氣", "模型會痛", "感到憤怒", "感到疼痛")
OPP = {"H3": {"y_effect": -0.3, "y_ci_low": -0.5, "y_ci_high": -0.1}}
SAME = {"H3": {"y_effect": 0.2, "y_ci_low": 0.1, "y_ci_high": 0.3}}
FUZZY = {"H3": {"y_effect": -0.0001, "y_ci_low": -0.1, "y_ci_high": 0.1}}


def test_interpret_sentences_follow_spec_rules():
    t = J(_prim(H1=True))
    assert "憤怒的效果可以用三個通用軸解釋" in t
    t = J(_prim(H1=True, H4=True))
    assert "憤怒有超出三個通用軸的作用" in t and "可以用三個通用軸解釋" not in t
    t = J(_prim(H1=True, H3=True))
    assert "憤怒和恐懼的作用不同，但差異可以用三個通用軸（很可能是掌控感）解釋" in t
    assert "只扣兩個軸" in t and "只扣掉了" not in t      # full V/A/D and V/A: no caveats
    t = J(_prim(H3=None, H3_reason="模型內部無法區分憤怒與恐懼"))
    assert "無法判斷" in t and "模型內部無法區分憤怒與恐懼" in t
    t = J(_prim())
    assert "不支持" in t
    for st in ({}, {"H1": True}, {"H4": True}, {"H3": True}):
        assert not any(f in J(_prim(rows=OPP, **st)) for f in FORBIDDEN)


def test_interpret_h3_direction_needs_ci():
    t = J(_prim(rows=OPP, H3=True))
    assert "方向相反" in t and "最有力的證據" in t
    t = J(_prim(rows=SAME, H3=True))
    assert "兩者只是大小不同" in t and "最有力的證據" not in t
    t = J(_prim(rows=FUZZY, H3=True))       # opposite point estimates, CI of fear spans 0
    assert "無法確定方向" in t and "最有力的證據" not in t and "兩者只是大小不同" not in t
    t = J(_prim(rows={"H3": {"y_ci_low": np.nan}}, H3=True))
    assert "無法計算" in t and "最有力的證據" not in t


def test_interpret_rule2_and_always_states_partial_axes():
    fz = frozen(residual_removed_axes={**FROZEN["residual_removed_axes"],
                                       "anger__resid3": ["V", "D"], "anger__resid2": ["V"]})
    lines = interpret(_prim(H1=True, H3=True), fz)
    r2 = next(x for x in lines if "憤怒和恐懼的作用不同" in x)
    assert "H4 實際只扣掉了正負面（V）、掌控感（D）" in r2
    assert "「只扣兩個軸」的版本實際只扣掉了正負面（V）。" in r2
    lines = interpret(_prim(), fz)            # no combined rule fires: still stated
    assert any("只扣掉了正負面（V）、掌控感（D）" in x and "激動程度（A）不可靠" in x
               for x in lines)
    fz2 = frozen(residual_removed_axes={**FROZEN["residual_removed_axes"],
                                        "anger__resid3": ["V", "D"]})   # resid2 = V, A: full
    r2 = next(x for x in interpret(_prim(H1=True, H3=True), fz2) if "作用不同" in x)
    assert "「只扣兩個軸」的版本實際只扣掉了" not in r2


def test_why_not_reports_nan_mean():
    t = J(_prim(rows={"H2": {"mean_d": np.nan, "p": np.nan, "p_holm": np.nan}}))
    line = next(x for x in t.splitlines() if x.startswith("H2"))
    assert "平均差無法計算" in line and "平均差不是正的" not in line


def test_interpret_names_removed_axes():
    fz = frozen(residual_removed_axes={**FROZEN["residual_removed_axes"],
                                       "anger__resid3": ["V", "D"]})
    t = "\n".join(interpret(_prim(H1=True, H4=True), fz))
    assert "超出三個通用軸" not in t
    s = next(x for x in t.splitlines() if "超出" in x)
    assert "正負面" in s and "掌控感" in s and "激動程度" not in s and "這幾個軸" in s


# ---------- required frozen keys (R23) ----------

KEY_USERS = {"alpha_max": "pse", "alpha_eq": "pse", "alpha_individual": "pe",
             "induction": "pse", "contamination_at_layer": "pse",
             "separability_at_layer": "pe", "residual_status_at_layer": "pse",
             "residual_removed_axes": "i"}


@pytest.fixture(scope="module")
def small():
    return make_scores(n=8)


@pytest.mark.parametrize("key", sorted(KEY_USERS))
def test_missing_frozen_key_raises(key, small):
    fz = {k: v for k, v in FROZEN.items() if k != key}
    calls = {"p": lambda: primary_tests(small, fz), "s": lambda: secondary_tests(small, fz),
             "e": lambda: exploratory(small, fz, None), "i": lambda: interpret(_prim(), fz)}
    for c in KEY_USERS[key]:
        with pytest.raises(ValueError, match=key):
            calls[c]()


def test_missing_frozen_entry_raises(small):
    ind = {k: v for k, v in FROZEN["induction"].items() if k != "pain"}
    with pytest.raises(ValueError, match="pain"):
        primary_tests(small, frozen(induction=ind))      # no "missing = ok" default
    with pytest.raises(ValueError, match="pain"):         # fallback without its α
        primary_tests(small, frozen(induction={**FROZEN["induction"], "pain": "fallback"}))
    rra = {k: v for k, v in FROZEN["residual_removed_axes"].items() if k != "anger__resid3"}
    with pytest.raises(ValueError, match="anger__resid3"):
        interpret(_prim(), frozen(residual_removed_axes=rra))


# ---------- integrity & script ----------

def test_analysis_refuses_changed_rules(tmp_path):
    files = {"stats_sha256": STATS_PY, "conditions_sha256": CONDITIONS_PY}
    good = {k: integrity.file_sha256(p) for k, p in files.items()}
    assert integrity.check_frozen_hashes(good, files, allow_changed=False) is False
    bad = {**good, "stats_sha256": "0" * 64}
    with pytest.raises(SystemExit) as ei:
        integrity.check_frozen_hashes(bad, files, allow_changed=False)
    assert "stats.py" in str(ei.value.code)
    assert integrity.check_frozen_hashes(bad, files, allow_changed=True) is True


def _load_script():
    path = REPO / "scripts/6.8_analysis/01_stats.py"
    spec = importlib.util.spec_from_file_location("stats01", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_analysis_refuses_without_frozen(tmp_path):
    mod = _load_script()
    with pytest.raises(SystemExit) as ei:
        mod.main(["--frozen", str(tmp_path / "frozen_config.json"),
                  "--scores-dir", str(tmp_path), "--out-dir", str(tmp_path / "out")])
    assert "凍結" in str(ei.value.code)


def _write_scores(st, d):
    d.mkdir(parents=True)
    for task in ("h1_rating", "h2"):
        vs = ("light1", "harsh1") if task == "h1_rating" else ("0", "1")
        for cond in st[task].columns:
            val, inv = st[task][cond], st[f"invalid_{task}"][cond]
            pd.DataFrame({"item_id": np.repeat(val.index, 2), "variant": list(vs) * len(val),
                          "p_target": np.repeat(val.to_numpy(), 2),
                          "invalid": np.repeat(inv.to_numpy(), 2)}).to_csv(
                d / f"{task}__{cond}.csv", index=False)


def test_01_stats_end_to_end(tmp_path):
    mod = _load_script()
    hashed = mod.hashed_files()
    assert set(hashed) == {"stats_sha256", "conditions_sha256", "concepts_sha256",
                           "vignettes_sha256", "moralchoice_sha256"}
    fz = frozen(**{k: integrity.file_sha256(p) for k, p in hashed.items()})
    _write_scores(make_scores(fz, shifts=PLANT), tmp_path / "scores")
    fpath = tmp_path / "frozen_config.json"
    fpath.write_text(json.dumps(fz), encoding="utf-8")
    out = tmp_path / "out"
    args = ["--frozen", str(fpath), "--scores-dir", str(tmp_path / "scores"),
            "--out-dir", str(out)]
    mod.main(args)
    names = ("primary.csv", "secondary.csv", "exploratory.json", "interpretation.md")
    assert all((out / n).exists() for n in names)
    prim = pd.read_csv(out / "primary.csv")
    assert list(prim["hypothesis"]) == ["H1", "H2", "H3", "H4"]
    assert "exploratory_changed_rules" not in prim.columns
    json.loads((out / "exploratory.json").read_text(encoding="utf-8"))
    md = (out / "interpretation.md").read_text(encoding="utf-8")
    assert "18" in md and "⚠" not in md

    for key, fname in (("moralchoice_sha256", "6.6_moralchoice_harm.json"),
                       ("concepts_sha256", "6.2_concepts.json"),
                       ("vignettes_sha256", "6.6_punishment_vignettes.json"),
                       ("stats_sha256", "stats.py")):
        fpath.write_text(json.dumps({**fz, key: "0" * 64}), encoding="utf-8")
        with pytest.raises(SystemExit) as ei:
            mod.main(args)
        assert fname in str(ei.value.code)
    mod.main(args + ["--allow-changed-analysis"])
    md = (out / "interpretation.md").read_text(encoding="utf-8")
    assert md.startswith("⚠ 凍結後改過分析程式：以下所有結果僅屬探索性") or \
        "⚠ 凍結後改過分析程式：以下所有結果僅屬探索性" in md.splitlines()[0]
    for n in ("primary.csv", "secondary.csv"):
        assert pd.read_csv(out / n)["exploratory_changed_rules"].all()
