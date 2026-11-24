import importlib.util
import json
import math
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from emolab import axes as axes_mod
from emolab import calibration as cal
from emolab import freeze, stats
from emolab.conditions import build_conditions, load_labels

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "6.5_calibration"
CORE = load_labels()["core"]
NAME = "Qwen_3_8B_instruct"
LAYERS = [9, 14, 18, 23]
GRID = cal.ALPHA_GRID
BANNED = ("AI 會生氣", "AI 會痛", "模型感到")


def _script(fname, modname):
    spec = importlib.util.spec_from_file_location(modname, SCRIPTS / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------- synthetic tables ----------

def _row(layer, alpha, label, gain, ok=True, base=1.0):
    return dict(layer=layer, alpha=alpha, label=label, rating_mean=base + gain, ppl_ratio=1.0,
                fluency=4.0, format_valid=1.0, coherent=ok, gain=gain)


def make_metrics(spec):
    """spec {(layer, emotion): [(alpha, gain, coherent), ...]}; alpha = 0 rows are added."""
    rows = []
    for (layer, e), pts in spec.items():
        rows.append(_row(layer, 0.0, e, 0.0))
        rows += [_row(layer, a, e, g, ok) for a, g, ok in pts]
    return pd.DataFrame(rows, columns=cal.METRICS_COLS)


def ramp(slope, cm):
    """Grid points up to cm, coherent, gain = slope * alpha; then one incoherent point."""
    pts = [(a, slope * a, True) for a in GRID if a <= cm + 1e-9]
    nxt = [a for a in GRID if a > cm + 1e-9]
    return pts + ([(nxt[0], slope * nxt[0], False)] if nxt else [])


def layer_spec(layer, cms, slopes):
    return {(layer, e): ramp(slopes[e], cms[e]) for e in CORE}


def make_controls(alpha, value=1.5, n=60, per_emotion=None):
    rows = []
    for e in CORE:
        vals = (per_emotion or {}).get(e)
        for i in range(n):
            rows.append(dict(alpha=alpha, control=f"c{i:02d}", emotion=e,
                             rating=float(vals[i]) if vals is not None else value))
    return pd.DataFrame(rows, columns=cal.CONTROL_COLS)


def contam(flagged=None, layers=LAYERS):
    """flagged {layer: {probe: {emotions}}}."""
    flagged = flagged or {}
    labels = [*CORE, "disgust"]
    return {str(L): {p: {e: {"cos": 0.0, "flagged": e in flagged.get(L, {}).get(p, set())}
                         for e in labels}
                     for p in ("punishment", "harming_others")} for L in layers}


# ---------- coherent_max / alpha_max ----------

def test_coherent_max_stops_before_first_incoherent():
    m = make_metrics({(9, "anger"): [(0.1, .5, True), (0.2, 1., True), (0.3, 1.5, False),
                                     (0.4, 2., True)]})
    assert freeze.coherent_max(m, 9, "anger") == 0.2
    assert freeze.coherent_max(m.iloc[::-1], 9, "anger") == 0.2   # row order irrelevant


def test_coherent_max_zero_and_truncated_series():
    m = make_metrics({(9, "anger"): [(0.1, .5, False)],
                      (9, "fear"): [(0.1, .5, True), (0.2, .7, True)]})
    assert freeze.coherent_max(m, 9, "anger") == 0.0     # smallest alpha incoherent
    assert freeze.coherent_max(m, 9, "fear") == 0.2      # coherent_max alone takes the tail
    assert freeze.coherent_max(m, 9, "joy") == 0.0       # no rows at all
    assert freeze.coherent_max(m, 14, "fear") == 0.0
    # ...but a series that stops while still coherent is an incomplete sweep: rejected
    with pytest.raises(ValueError, match="layer 9 fear"):
        freeze.validate_metrics(m, [9], ["anger", "fear"])


def test_coherent_max_nan_flag_is_incoherent():
    m = make_metrics({(9, "anger"): [(0.1, .5, True), (0.2, 1., True), (0.3, 1.5, True)]})
    m["coherent"] = m["coherent"].astype(object)
    m.loc[m["alpha"] == 0.2, "coherent"] = float("nan")
    assert freeze.coherent_max(m, 9, "anger") == 0.1     # fail closed


def test_coherent_max_reads_string_flags_from_csv(tmp_path):
    m = make_metrics({(9, "anger"): [(0.1, .5, True), (0.2, 1., False)]})
    m.to_csv(tmp_path / "metrics.csv", index=False)
    back = freeze.load_metrics(tmp_path / "metrics.csv")
    assert freeze.coherent_max(back, 9, "anger") == 0.1


# ---------- validate_metrics ----------

FULL = [(a, a, True) for a in GRID]                      # never incoherent: all 12 points


def _valid_spec():
    """Layers 9/14: early-stopped series (last row incoherent) and full series."""
    spec = {}
    for i, e in enumerate(CORE):
        spec[(9, e)] = ramp(2.0, GRID[i])                # grid[:i+2], last incoherent
        spec[(14, e)] = list(FULL)
    spec[(9, "pain")] = list(FULL)
    return spec


def test_validate_metrics_accepts_early_stopped_and_full_series():
    freeze.validate_metrics(make_metrics(_valid_spec()), [9, 14], CORE)
    freeze.validate_metrics(make_metrics(_valid_spec()).iloc[::-1], [9, 14], CORE)


def _bad(spec_fix, match, layers=(9, 14)):
    spec = _valid_spec()
    spec_fix(spec)
    with pytest.raises(ValueError, match=match):
        freeze.validate_metrics(make_metrics(spec), list(layers), CORE)


def test_validate_metrics_truncated_tail():
    _bad(lambda s: s.__setitem__((14, "joy"), FULL[:3]), "layer 14 joy")


def test_validate_metrics_missing_middle():
    _bad(lambda s: s.__setitem__((14, "joy"), FULL[:4] + FULL[5:]), r"layer 14 joy.*missing")


def test_validate_metrics_off_grid_alpha():
    _bad(lambda s: s[(14, "joy")].append((0.25, 0.25, True)), r"layer 14 joy.*off-grid")


def test_validate_metrics_duplicate_rows():
    _bad(lambda s: s[(14, "joy")].append(FULL[2]), r"layer 14 joy.*duplicate")


def test_validate_metrics_missing_series_and_unexpected_layer():
    _bad(lambda s: s.pop((14, "joy")), r"layer 14 joy")
    _bad(lambda s: s.__setitem__((23, "joy"), list(FULL)), r"unexpected layers \[23\]")
    _bad(lambda s: None, r"layer 18 anger", layers=(9, 14, 18))


def test_validate_metrics_needs_alpha0_row():
    m = make_metrics(_valid_spec())
    m = m[~((m["layer"] == 9) & (m["label"] == "fear") & (m["alpha"] == 0.0))]
    with pytest.raises(ValueError, match=r"layer 9 fear.*alpha = 0"):
        freeze.validate_metrics(m, [9, 14], CORE)


def test_validate_metrics_names_every_bad_series():
    spec = _valid_spec()
    spec[(14, "joy")] = FULL[:3]
    spec[(9, "pain")] = FULL[:4] + FULL[5:]
    with pytest.raises(ValueError) as ei:
        freeze.validate_metrics(make_metrics(spec), [9, 14], CORE)
    assert "layer 14 joy" in str(ei.value) and "layer 9 pain" in str(ei.value)


@pytest.mark.parametrize("cell", ["", "nan", "maybe"])
def test_validate_metrics_rejects_blank_or_invalid_coherent(tmp_path, cell):
    m = make_metrics(_valid_spec())
    m["coherent"] = m["coherent"].map(str)
    m.loc[(m["layer"] == 14) & (m["label"] == "joy") & (m["alpha"] == 0.3), "coherent"] = cell
    m.to_csv(tmp_path / "metrics.csv", index=False)
    back = freeze.load_metrics(tmp_path / "metrics.csv")
    with pytest.raises(ValueError, match=r"layer 14 joy.*coherent"):
        freeze.validate_metrics(back, [9, 14], CORE)
    assert freeze.coherent_max(back, 14, "joy") == 0.2       # fail closed


def test_validate_metrics_catches_truncated_layer_that_flips_choice():
    """Layer 23 cut off after 0.3 (all coherent) would silently lose to layer 14."""
    spec = {**_uniform(14, 0.4, 3.0), **_uniform(23, 1.3, 2.0)}
    m = make_metrics(spec)
    assert freeze.choose_layer(m, contam()) == 23
    cut = m[~((m["layer"] == 23) & (m["alpha"] > 0.3))]
    assert freeze.choose_layer(cut, contam()) == 14           # what the guard prevents
    freeze.validate_metrics(m, [14, 23], CORE)
    with pytest.raises(ValueError, match="layer 23"):
        freeze.validate_metrics(cut, [14, 23], CORE)


def test_alpha_max_is_min_over_core():
    cms = dict(zip(CORE, [0.8, 0.5, 1.3, 0.6, 0.7, 0.4]))
    m = make_metrics(layer_spec(9, cms, {e: 3.0 for e in CORE}))
    assert freeze.alpha_max(m, 9) == 0.4
    assert freeze.alpha_max(m, 9, CORE) == 0.4


# ---------- choose_layer ----------

def _uniform(layer, cm, slope):
    return layer_spec(layer, {e: cm for e in CORE}, {e: slope for e in CORE})


def test_choose_layer_mean_gain_at_own_alpha_max():
    spec = {**_uniform(9, 0.5, 2.0),      # gain 1.0 at its alpha_max 0.5
            **_uniform(14, 0.3, 4.0),     # gain 1.2 at 0.3  <- best
            **_uniform(18, 0.8, 1.0),     # gain 0.8 at 0.8
            **_uniform(23, 0.0, 100.0)}   # nothing coherent: skipped despite huge gains
    m = make_metrics(spec)
    assert freeze.choose_layer(m, contam()) == 14
    table = {r["layer"]: r for r in freeze.layer_table(m, contam(), CORE)}
    assert table[14]["alpha_max"] == 0.3 and table[14]["mean_gain"] == pytest.approx(1.2)
    assert table[23]["alpha_max"] == 0.0 and table[23]["mean_gain"] is None


@pytest.mark.parametrize("reverse", [False, True])
def test_choose_layer_tie_fewer_flagged_then_smaller_layer(reverse):
    m = make_metrics({**_uniform(9, 0.5, 2.0), **_uniform(14, 0.5, 2.0),
                      **_uniform(18, 0.4, 2.0)})
    if reverse:
        m = m.iloc[::-1].reset_index(drop=True)
    # 14 has fewer flagged emotions than 9 -> 14, although 9 is the smaller layer
    c = contam({9: {"punishment": {"anger"}, "harming_others": {"pain"}},
                14: {"punishment": {"anger"}}})
    assert freeze.choose_layer(m, c) == 14
    # 9 has fewer flagged emotions -> 9
    c = contam({9: {"punishment": {"anger"}}, 14: {"harming_others": {"anger", "pain"}}})
    assert freeze.choose_layer(m, c) == 9
    # equal flag counts -> smaller layer
    c = contam({9: {"punishment": {"fear"}}, 14: {"harming_others": {"anger"}}})
    assert freeze.choose_layer(m, c) == 9


def test_choose_layer_tie_uses_isclose():
    m = make_metrics({**_uniform(9, 0.5, 2.0), **_uniform(14, 0.5, 2.0)})
    m.loc[(m["layer"] == 9) & (m["alpha"] == 0.5), "gain"] += 1e-12
    c = contam({9: {"punishment": {"anger"}}})
    # layer 9 is 1e-12 better but has more flags: counts as a tie, so flags decide -> 14
    assert freeze.choose_layer(m, c) == 14
    m.loc[(m["layer"] == 9) & (m["alpha"] == 0.5), "gain"] += 1e-3
    assert freeze.choose_layer(m, c) == 9      # a real difference wins over flags


def test_choose_layer_tie_tolerance_is_absolute_only():
    """rel_tol = 0: at large gains a 5e-7 difference is not a tie (default rel_tol would be)."""
    m = make_metrics({**_uniform(9, 0.5, 2000.0), **_uniform(14, 0.5, 2000.0)})
    m.loc[(m["layer"] == 9) & (m["alpha"] == 0.5), "gain"] += 5e-7
    c = contam({9: {"punishment": {"anger"}}})
    assert freeze.choose_layer(m, c) == 9


def test_choose_layer_raises_without_coherent_layer():
    m = make_metrics({**_uniform(9, 0.0, 1.0), **_uniform(14, 0.0, 1.0)})
    with pytest.raises(ValueError):
        freeze.choose_layer(m, contam())


def test_flagged_at_any_probe_all_labels():
    c = contam({9: {"punishment": {"anger"}, "harming_others": {"pain", "disgust", "anger"}}})
    assert freeze.flagged_at(c, 9) == ["anger", "disgust", "pain"]
    assert freeze.flagged_at(c, 14) == []
    with pytest.raises(ValueError):
        freeze.flagged_at(c, 99)


# ---------- induction (R13) ----------

AMAX = 0.4


def _ind_metrics(**override):
    """Layer 9; every emotion gain 5*alpha up to 0.4 (rating 3.0 at 0.4), incoherent at 0.5;
    override[e] = explicit points."""
    spec = {(9, e): [(a, 5 * a, True) for a in GRID if a <= AMAX + 1e-9] + [(0.5, 2.5, False)]
            for e in CORE}
    for e, pts in override.items():
        spec[(9, e)] = pts
    m = make_metrics(spec)
    return m, {e: freeze.coherent_max(m, 9, e) for e in CORE}


def test_induction_all_ok():
    m, cm = _ind_metrics()
    assert freeze.alpha_max(m, 9) == AMAX
    res = freeze.induction(m, make_controls(AMAX, 1.5), 9, AMAX, cm)
    assert list(res) == CORE
    for d in res.values():
        assert d == {"status": "ok", "alpha": AMAX, "gain": 2.0, "rating": 3.0, "p95": 1.5}


def test_induction_p95_is_linear_percentile_of_60_controls():
    vals = np.arange(60) / 20.0
    m, cm = _ind_metrics()
    res = freeze.induction(m, make_controls(AMAX, per_emotion={e: vals for e in CORE}),
                           9, AMAX, cm)
    assert res["anger"]["p95"] == pytest.approx(np.percentile(vals, 95))
    assert res["anger"]["status"] == "ok"


def test_induction_rating_equal_to_p95_is_not_ok():
    m, cm = _ind_metrics()           # coherent_max == alpha_max: no fallback alpha exists
    res = freeze.induction(m, make_controls(AMAX, 3.0), 9, AMAX, cm)
    for d in res.values():
        assert d["status"] == "failed" and d["alpha"] is None
        assert d["rating"] == d["p95"] == 3.0


def test_induction_gain_below_one_is_not_ok():
    pts = [(0.1, .2, True), (0.2, .4, True), (0.3, .6, True), (0.4, .99, True), (0.5, 3., False)]
    m, cm = _ind_metrics(anger=pts)
    res = freeze.induction(m, make_controls(AMAX, 0.0), 9, AMAX, cm)
    assert res["anger"]["status"] == "failed"
    assert res["fear"]["status"] == "ok"


@pytest.mark.parametrize("n", [59, 61])
def test_induction_requires_exactly_60_controls(n):
    m, cm = _ind_metrics()
    with pytest.raises(ValueError):
        freeze.induction(m, make_controls(AMAX, 1.5, n=n), 9, AMAX, cm)


def test_induction_rejects_nan_control_rating():
    m, cm = _ind_metrics()
    ctrl = make_controls(AMAX, 1.5)
    ctrl.loc[0, "rating"] = float("nan")
    with pytest.raises(ValueError):
        freeze.induction(m, ctrl, 9, AMAX, cm)


ANGER_SLOW = [(0.1, .1, True), (0.2, .2, True), (0.3, .3, True), (0.4, .5, True),
              (0.5, .8, True), (0.6, 1.2, True), (0.7, 1.5, True), (0.8, 1.7, False)]


def test_induction_need_controls_carries_alpha():
    m, cm = _ind_metrics(anger=ANGER_SLOW)
    assert cm["anger"] == 0.7 and freeze.alpha_max(m, 9) == AMAX
    with pytest.raises(freeze.NeedControls) as ei:
        freeze.induction(m, make_controls(AMAX, 1.5), 9, AMAX, cm)
    assert ei.value.alpha == 0.6       # first grid alpha above alpha_max with gain >= 1


def test_induction_fallback_and_continue_walk():
    m, cm = _ind_metrics(anger=ANGER_SLOW)
    base = make_controls(AMAX, 1.5)
    # rating at 0.6 = 2.2 > 1.5 -> fallback at 0.6
    res = freeze.induction(m, pd.concat([base, make_controls(0.6, 1.5)]), 9, AMAX, cm)
    assert res["anger"] == {"status": "fallback", "alpha": 0.6, "gain": 1.2, "rating": 2.2,
                            "p95": 1.5}
    assert res["fear"]["status"] == "ok"
    # rating 2.2 <= 2.5 at 0.6 -> walk on; 0.7 needs controls
    ctrl = pd.concat([base, make_controls(0.6, 2.5)])
    with pytest.raises(freeze.NeedControls) as ei:
        freeze.induction(m, ctrl, 9, AMAX, cm)
    assert ei.value.alpha == 0.7
    res = freeze.induction(m, pd.concat([ctrl, make_controls(0.7, 2.0)]), 9, AMAX, cm)
    assert res["anger"]["status"] == "fallback" and res["anger"]["alpha"] == 0.7
    # nothing passes up to coherent_max (0.7) -> failed
    res = freeze.induction(m, pd.concat([ctrl, make_controls(0.7, 3.0)]), 9, AMAX, cm)
    assert res["anger"]["status"] == "failed" and res["anger"]["alpha"] is None


def test_induction_nan_gain_never_requests_controls():
    nan = float("nan")
    pts = [(0.1, .1, True), (0.2, .2, True), (0.3, .3, True), (0.4, .5, True),
           (0.5, nan, True), (0.6, nan, True), (0.7, nan, True), (0.8, 1.7, False)]
    m, cm = _ind_metrics(anger=pts)
    res = freeze.induction(m, make_controls(AMAX, 1.5), 9, AMAX, cm)
    assert res["anger"]["status"] == "failed"
    pts[5] = (0.6, 1.2, True)
    m, cm = _ind_metrics(anger=pts)
    with pytest.raises(freeze.NeedControls) as ei:
        freeze.induction(m, make_controls(AMAX, 1.5), 9, AMAX, cm)
    assert ei.value.alpha == 0.6                      # NaN at 0.5 skipped, not requested


def test_pain_fallback_never_exceeds_its_coherent_max():
    # pain: too weak up to its coherent_max 0.5; strong (and even "coherent") beyond it
    pain = [(0.1, .2, True), (0.2, .4, True), (0.3, .6, True), (0.4, .8, True),
            (0.5, .9, True), (0.6, 3.0, False), (0.7, 3.5, True), (0.8, 4.0, True)]
    m, cm = _ind_metrics(pain=pain)
    assert cm["pain"] == 0.5
    ctrl = pd.concat([make_controls(a, 1.0) for a in (AMAX, 0.5, 0.6, 0.7, 0.8)])
    res = freeze.induction(m, ctrl, 9, AMAX, cm)
    assert res["pain"]["status"] == "failed" and res["pain"]["alpha"] is None
    # passes exactly at its own coherent_max: allowed
    pain[4] = (0.5, 1.1, True)
    m, cm = _ind_metrics(pain=pain)
    res = freeze.induction(m, ctrl, 9, AMAX, cm)
    assert res["pain"]["status"] == "fallback" and res["pain"]["alpha"] == 0.5
    for e, d in res.items():
        assert d["alpha"] is None or d["alpha"] <= cm[e]


# ---------- alpha_eq (R17) ----------

def _eq_case():
    spec = {
        (9, "anger"): [(0.1, 1.0, True), (0.2, 3.0, True), (0.3, 3.5, False)],
        (9, "fear"): [(0.1, 0.5, True), (0.2, 1.5, True), (0.3, 1.8, True), (0.4, 1.0, False)],
        (9, "sadness"): [(0.1, 3.6, True), (0.2, 4.0, False)],
        (9, "joy"): [(0.1, 1.0, True), (0.2, 1.0, True), (0.3, 2.6, True)],
        (9, "compassion"): [(0.1, 1.8, True)],
        (9, "pain"): [(0.1, 0.9, True), (0.2, 2.0, True), (0.3, 1.0, True)],
    }
    m = make_metrics(spec)
    return m, {e: freeze.coherent_max(m, 9, e) for e in CORE}


def test_alpha_eq_interpolation_including_first_segment():
    m, cm = _eq_case()
    t, eq = freeze.alpha_eq(m, 9, cm, {e: {"status": "ok"} for e in CORE})
    assert t == pytest.approx(1.8)   # fear's and compassion's best gain within coherent_max
    assert eq == pytest.approx({"anger": 0.14, "fear": 0.3, "sadness": 0.05, "joy": 0.25,
                                "compassion": 0.1, "pain": 0.1818})
    assert eq["pain"] == 0.1818      # rounded to 4 decimals (0.18181...)


def test_alpha_eq_target_capped_at_two():
    m = make_metrics(_uniform(9, 0.8, 10.0))
    cm = {e: 0.8 for e in CORE}
    t, eq = freeze.alpha_eq(m, 9, cm, {e: "ok" for e in CORE})
    assert t == 2.0
    assert all(v == pytest.approx(0.2) for v in eq.values())


def test_alpha_eq_ignores_gain_beyond_coherent_max():
    spec = _uniform(9, 0.8, 10.0)
    spec[(9, "anger")] = [(0.1, 1.2, True), (0.2, 5.0, False), (0.3, 6.0, True)]
    m = make_metrics(spec)
    cm = {e: freeze.coherent_max(m, 9, e) for e in CORE}
    t, eq = freeze.alpha_eq(m, 9, cm, {e: "ok" for e in CORE})
    assert t == pytest.approx(1.2)
    assert eq["anger"] == pytest.approx(0.1)


def test_alpha_eq_failed_excluded_from_target_and_none():
    spec = _uniform(9, 0.8, 10.0)
    spec[(9, "pain")] = [(0.1, 0.1, True), (0.2, 0.3, True), (0.3, 0.2, False)]
    m = make_metrics(spec)
    cm = {e: freeze.coherent_max(m, 9, e) for e in CORE}
    ind = {e: {"status": "ok"} for e in CORE}
    ind["pain"] = {"status": "failed"}
    ind["joy"] = {"status": "fallback"}
    t, eq = freeze.alpha_eq(m, 9, cm, ind)
    assert t == 2.0                       # pain's 0.3 does not set the target
    assert eq["pain"] is None and eq["joy"] == pytest.approx(0.2)


def test_alpha_eq_all_failed():
    m, cm = _eq_case()
    t, eq = freeze.alpha_eq(m, 9, cm, {e: "failed" for e in CORE})
    assert math.isnan(t) and eq == {e: None for e in CORE}


# ---------- write_frozen ----------

def test_write_frozen_refuses_and_force_keeps_old(tmp_path):
    p = tmp_path / "frozen_config.json"
    freeze.write_frozen(p, {"layer": 9, "note": "通順"})
    raw = p.read_bytes()
    assert raw.endswith(b"}\n") and b"\r\n" not in raw and b'\n  "layer": 9' in raw
    assert json.loads(raw.decode("utf-8")) == {"layer": 9, "note": "通順"}
    with pytest.raises(FileExistsError):
        freeze.write_frozen(p, {"layer": 14})
    assert json.loads(p.read_text(encoding="utf-8"))["layer"] == 9
    payload = {"layer": 14}
    freeze.write_frozen(p, payload, force=True)
    old = list(tmp_path.glob("frozen_config.superseded-*.json"))
    assert len(old) == 1
    assert re.fullmatch(r"frozen_config\.superseded-\d{8}T\d{6}\.json", old[0].name)
    assert json.loads(old[0].read_text(encoding="utf-8"))["layer"] == 9
    new = json.loads(p.read_text(encoding="utf-8"))
    assert new == {"layer": 14, "supersedes": old[0].name}
    assert payload["supersedes"] == old[0].name
    assert not list(tmp_path.glob("*.tmp"))


# ---------- build_payload ----------

def _axes_meta(reliable):
    """reliable {fit_set: "VD"} -> axes.json-like dict at layer 18."""
    return {"fits": {fs: {"18": {k: {"loo_r": 0.9 if k in reliable.get(fs, "VAD") else 0.1,
                                     "reliable": k in reliable.get(fs, "VAD")} for k in "VAD"}}
                     for fs in axes_mod.FIT_SETS}}


def _payload(**kw):
    ind = {e: {"status": "ok", "alpha": 0.4, "gain": 2.0, "rating": 3.0, "p95": 1.5}
           for e in CORE}
    ind["pain"] = {"status": "fallback", "alpha": 0.6, "gain": 1.2, "rating": 2.2, "p95": 1.5}
    ind["joy"] = {"status": "failed", "alpha": None, "gain": 0.3, "rating": 1.3, "p95": 1.5}
    args = dict(
        model=NAME, layer=18, alpha_max=0.4,
        coherent={e: 0.8 for e in CORE}, induction=ind, eq_target=1.5,
        alpha_eq={**{e: 0.3 for e in CORE}, "joy": None},
        layer_table=[{"layer": 18, "alpha_max": 0.4, "mean_gain": 1.6, "n_flagged": 2}],
        contamination=contam({18: {"punishment": {"pain"}, "harming_others": {"anger"}}}),
        separability={"18": {"auroc": 0.98, "separable": True}},
        residual_status={"anger__resid3": {18: "ok"}, "anger__resid2": {18: "ok"},
                         "pain__resid3": {18: "untestable"}, "compassion__resid3": {18: "ok"}},
        axes_meta=_axes_meta({"for_anger": "VD"}),
        resid_norm={9: 54.7, 18: 94.2},
        hashes={k: f"h_{k}" for k in freeze.HASHED},
        git_commit="abc123", created_at="2026-11-22T00:00:00+00:00")
    args.update(kw)
    return freeze.build_payload(**args)


def test_build_payload_keys_and_values():
    p = _payload()
    for k in (*stats.KEYS_PRIMARY, *stats.KEYS_SECONDARY, *stats.KEYS_INTERPRET):
        assert k in p
    for k in ("model", "layer", "resid_norm", "alpha_eq_target", "induction_detail",
              "coherent_max", "thresholds", "grid", "layer_table", "stats_sha256",
              "conditions_sha256", "concepts_sha256", "vignettes_sha256", "moralchoice_sha256",
              "git_commit", "created_at", "coherence_rule"):
        assert k in p
    assert p["layer"] == 18 and p["resid_norm"] == 94.2 and p["alpha_max"] == 0.4
    assert p["induction"] == {**{e: "ok" for e in CORE}, "pain": "fallback", "joy": "failed"}
    assert p["induction_detail"]["pain"]["alpha"] == 0.6
    assert p["alpha_individual"] == {"pain": 0.6}
    assert set(p["alpha_eq"]) == set(CORE) and p["alpha_eq"]["joy"] is None
    assert p["contamination_at_layer"] == ["anger", "pain"]
    assert p["separability_at_layer"] == {"auroc": 0.98, "passed": True}
    assert p["residual_status_at_layer"]["pain__resid3"] == "untestable"
    assert p["residual_removed_axes"] == {"anger__resid3": ["V", "D"], "anger__resid2": ["V"],
                                          "pain__resid3": [],
                                          "compassion__resid3": ["V", "A", "D"]}
    assert p["thresholds"] == {"format_min": 0.9, "fluency_drop": 0.5, "induction_gain": 1.0,
                               "control_percentile": 95, "eq_target_max": 2.0,
                               "holm_alpha": 0.05, "gate_need": 38, "bootstrap_n": 10000,
                               "seed": 42}
    assert p["grid"] == GRID and p["stats_sha256"] == "h_stats_sha256"
    assert "0.9" in p["coherence_rule"] and "0.5" in p["coherence_rule"]
    json.dumps(p, allow_nan=False)
    assert len(build_conditions(p, load_labels())) > 100


def test_build_payload_all_failed_target_is_null():
    p = _payload(eq_target=float("nan"))
    assert p["alpha_eq_target"] is None


def test_build_payload_asserts_required_stats_keys(monkeypatch):
    monkeypatch.setattr(stats, "KEYS_INTERPRET", ("residual_removed_axes", "not_a_key"))
    with pytest.raises(AssertionError):
        _payload()


def test_hashed_files_match_analysis_script():
    an = importlib.util.spec_from_file_location(
        "stats01", ROOT / "scripts" / "6.8_analysis" / "01_stats.py")
    mod = importlib.util.module_from_spec(an)
    an.loader.exec_module(mod)
    expect = {k: Path(v).resolve() for k, v in mod.hashed_files().items()}
    assert {k: (ROOT / rel).resolve() for k, rel in freeze.HASHED.items()} == expect


def test_passes_per_condition_matches_items():
    v = json.loads((ROOT / "datasets/6.6_punishment_vignettes.json").read_text(encoding="utf-8"))
    m = json.loads((ROOT / "datasets/6.6_moralchoice_harm.json").read_text(encoding="utf-8"))
    assert freeze.PASSES_PER_CONDITION == 4 * len(v) + 2 * len(m) == 588


# ---------- git helpers ----------

def test_git_commit_of_this_repo():
    assert re.fullmatch(r"[0-9a-f]{40}", freeze.git_commit(ROOT))


def test_uncommitted_lists_changed_hashed_files(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    assert freeze.uncommitted(tmp_path) == []          # files absent: nothing to report
    (tmp_path / "datasets").mkdir()
    (tmp_path / "datasets" / "6.2_concepts.json").write_text("[]", encoding="utf-8")
    (tmp_path / "other.txt").write_text("x", encoding="utf-8")
    lines = freeze.uncommitted(tmp_path)
    assert len(lines) == 1 and "datasets/6.2_concepts.json" in lines[0]


# ---------- render_checkpoint2 ----------

def _texts(layer, alpha):
    rows = [dict(layer=None, alpha=0.0, label="baseline", prompt_idx=i, text=f"BASE TEXT {i}")
            for i in range(3)]
    for e in CORE:
        rows += [dict(layer=layer, alpha=alpha, label=e, prompt_idx=i, text=f"{e} sample {i}")
                 for i in range(3)]
        rows.append(dict(layer=layer, alpha=0.1, label=e, prompt_idx=0, text=f"{e} WRONG ALPHA"))
    return rows


def test_render_checkpoint2_sections_and_wording():
    frozen = _payload()
    m = make_metrics({**_uniform(9, 0.5, 2.0), **_uniform(18, 0.4, 4.0)})
    md = freeze.render_checkpoint2(frozen, m, _texts(18, 0.4), n_conditions=240, rate=6.0)
    for s in ("選哪一層", "通順", "2026-11-22", "α_max", "α_eq", "誘發", "範例", "汙染",
              "扣掉", "工作量", "item_samples.md", "心理", "身體傷害", "請你確認"):
        assert s in md, s
    assert "強度不同" in md and "未能誘發" in md
    assert "anger sample 0" in md and "anger sample 1" in md and "anger sample 2" not in md
    assert "WRONG ALPHA" not in md and "BASE TEXT 0" in md
    assert "141,120" in md and "6.5" in md         # 240 x 588 passes at 6 per second
    assert "H1" in md and "H2" in md               # what fallback / failure means
    assert "| 9 |" in md and "| 18 |" in md       # per-layer tables
    for b in BANNED:
        assert b not in md


def _all_ok_frozen():
    fz = json.loads(json.dumps(_payload()))
    fz["induction"] = {e: "ok" for e in CORE}
    fz["alpha_individual"] = {}
    fz["alpha_eq"] = {e: 0.3 for e in CORE}
    fz["contamination_at_layer"] = []
    fz["residual_status_at_layer"] = {n: "ok" for n in axes_mod.RESIDUALS}
    return fz


def _hyps(frozen, pred):
    labels = load_labels()
    specs = stats._primary_specs(frozen, labels) + stats._secondary_specs(frozen, labels)
    return {s["hypothesis"] for s in specs if pred(s)}


@pytest.mark.parametrize("e", CORE)
def test_hypothesis_maps_match_stats(e):
    fz = _all_ok_frozen()
    assert _hyps(fz, lambda s: s["reason"] == stats.R_FAILED) == set()
    failed = {**fz, "induction": {**fz["induction"], e: "failed"},
              "alpha_eq": {**fz["alpha_eq"], e: None}}
    assert _hyps(failed, lambda s: s["reason"] == stats.R_FAILED) == set(freeze.HYP_IF_FAILED[e])
    fb = {**fz, "induction": {**fz["induction"], e: "fallback"}, "alpha_individual": {e: 0.6}}
    assert (_hyps(fb, lambda s: s.get("alpha_note") == stats.FALLBACK_NOTE)
            == set(freeze.HYP_IF_FALLBACK.get(e, ())))
    for h in (*freeze.HYP_IF_FAILED[e], *freeze.HYP_IF_FALLBACK.get(e, ())):
        assert h in freeze.HYP_ZH


def test_render_failed_anger_lists_every_affected_test():
    fz = _all_ok_frozen()
    fz["induction"]["anger"] = "failed"
    fz["induction_detail"]["anger"] = {"status": "failed", "alpha": None, "gain": 0.2,
                                       "rating": 1.2, "p95": 1.5}
    fz["alpha_eq"]["anger"] = None
    for e in CORE:
        if e != "anger":
            fz["induction_detail"][e] = {"status": "ok", "alpha": 0.4, "gain": 2.0,
                                         "rating": 3.0, "p95": 1.5}
    md = freeze.render_checkpoint2(fz, make_metrics(_uniform(18, 0.4, 4.0)), _texts(18, 0.4),
                                   n_conditions=240)
    note = next(ln for ln in md.splitlines() if ln.startswith("- **憤怒"))
    for h in freeze.HYP_IF_FAILED["anger"]:
        assert freeze.HYP_ZH[h] in note
    assert "anger_resid2_vs_resid3" in freeze.HYP_IF_FAILED["anger"]


def _contam_section(md):
    return md.split("## 六、")[1].split("## 七、")[0]


def test_render_contamination_core_vs_other_labels():
    p = _payload(contamination=contam({18: {"punishment": {"anger", "disgust"}}}))
    assert p["contamination_at_layer"] == ["anger", "disgust"]
    names = {c.name for c in build_conditions(p, load_labels())}
    assert any(n.startswith("anger__clean@") for n in names)          # core: clean run built
    assert not any(n.startswith("disgust__clean") for n in names)     # other: no clean run
    sec = _contam_section(freeze.render_checkpoint2(
        p, make_metrics(_uniform(18, 0.4, 4.0)), _texts(18, 0.4), n_conditions=240))
    core_line = next(ln for ln in sec.splitlines() if "anger" in ln)
    other_line = next(ln for ln in sec.splitlines() if "disgust" in ln)
    assert "disgust" not in core_line and "anger" not in other_line
    assert "不另外" in other_line and "乾淨版" in other_line
    assert "必須乾淨版也通過" in sec
    # only a non-core label flagged: no core clean-version rule claimed
    p = _payload(contamination=contam({18: {"punishment": {"disgust"}}}))
    sec = _contam_section(freeze.render_checkpoint2(
        p, make_metrics(_uniform(18, 0.4, 4.0)), _texts(18, 0.4), n_conditions=240))
    assert "沒有核心情緒被標示" in sec and "必須乾淨版也通過" not in sec and "disgust" in sec


# ---------- 03 / 04 end to end on synthetic files ----------

def _tree(tmp_path, anger18=None):
    res = tmp_path / "results"
    cal_dir = res / "6.5_calibration" / NAME
    vec_dir = res / "6.2_emotion_vectors" / NAME
    ax_dir = res / "6.3_axes" / NAME
    for d in (cal_dir, vec_dir, ax_dir):
        d.mkdir(parents=True)
    spec = {**_uniform(9, 0.5, 1.0), **_uniform(14, 0.4, 2.0), **_uniform(18, 0.4, 5.0),
            **_uniform(23, 0.6, 1.0)}
    if anger18 is not None:
        spec[(18, "anger")] = anger18
    make_metrics(spec).to_csv(cal_dir / "metrics.csv", index=False, lineterminator="\n")
    with (cal_dir / "texts.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for r in _texts(18, 0.4):
            f.write(json.dumps(r) + "\n")
    (vec_dir / "contamination.json").write_text(
        json.dumps(contam({18: {"punishment": {"anger"}}})), encoding="utf-8")
    (vec_dir / "separability.json").write_text(
        json.dumps({str(L): {"auroc": 0.98, "separable": True} for L in LAYERS}), encoding="utf-8")
    torch.save({"layers": LAYERS, "resid_norm": {9: 54.7, 14: 74.2, 18: 94.3, 23: 197.0}},
               vec_dir / "vectors.pt")
    torch.save({"status": {n: {L: "ok" for L in LAYERS} for n in axes_mod.RESIDUALS}},
               ax_dir / "residuals.pt")
    (ax_dir / "axes.json").write_text(json.dumps({"fits": {
        fs: {str(L): {k: {"reliable": True} for k in "VAD"} for L in LAYERS}
        for fs in axes_mod.FIT_SETS}}), encoding="utf-8")
    repo = tmp_path / "repo"
    for rel in freeze.HASHED.values():
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text(f"content of {rel}\n", encoding="utf-8")
    return res, cal_dir, repo


def _controls(cal_dir, *alphas, layer=18):
    pd.concat([make_controls(a, 1.5) for a in alphas]).to_csv(
        cal_dir / "control_ratings.csv", index=False, lineterminator="\n")
    (cal_dir / "control_meta.json").write_text(json.dumps({"layer": layer}), encoding="utf-8")


@pytest.fixture
def fake_git(monkeypatch):
    state = {"dirty": []}
    monkeypatch.setattr(freeze, "git_commit", lambda root: "f" * 40)
    monkeypatch.setattr(freeze, "uncommitted", lambda root, rel=None: list(state["dirty"]))
    return state


def _run03(res, repo, *extra):
    mod = _script("03_select_and_freeze.py", "select03")
    mod.main(["--model", "qwen3-8b", "--results-dir", str(res), "--repo-root", str(repo),
              *extra])


CMD = "python scripts/6.5_calibration/02_measure_calibration.py --model qwen3-8b --controls"


def test_03_phase1_exit3_when_controls_missing(tmp_path, capsys, fake_git):
    res, cal_dir, repo = _tree(tmp_path)
    with pytest.raises(SystemExit) as ei:
        _run03(res, repo)
    assert ei.value.code == 3
    assert f"{CMD} --layer 18 --alpha 0.4" in capsys.readouterr().out
    sel = json.loads((cal_dir / "selection.json").read_text(encoding="utf-8"))
    assert sel["layer"] == 18 and sel["alpha_max"] == 0.4
    assert not (cal_dir / "frozen_config.json").exists()


def test_03_rejects_incomplete_metrics(tmp_path, fake_git):
    res, cal_dir, repo = _tree(tmp_path)
    path = cal_dir / "metrics.csv"
    m = pd.read_csv(path)
    m[~((m["layer"] == 23) & (m["alpha"] > 0.3))].to_csv(path, index=False, lineterminator="\n")
    with pytest.raises(SystemExit) as ei:
        _run03(res, repo)
    msg = str(ei.value.code)
    assert ei.value.code not in (0, 3, None) and "layer 23" in msg and "拒絕選層" in msg
    assert not (cal_dir / "selection.json").exists()


def test_03_control_layer_mismatch_is_an_error(tmp_path, fake_git):
    res, cal_dir, repo = _tree(tmp_path)
    _controls(cal_dir, 0.4, layer=9)
    with pytest.raises(SystemExit) as ei:
        _run03(res, repo)
    assert ei.value.code not in (0, 3, None) and "9" in str(ei.value.code)


def test_03_phase2_need_controls_then_freeze(tmp_path, capsys, fake_git):
    anger = [(0.1, .1, True), (0.2, .2, True), (0.3, .3, True), (0.4, .5, True),
             (0.5, 1.2, True), (0.6, 1.5, False)]
    res, cal_dir, repo = _tree(tmp_path, anger18=anger)
    _controls(cal_dir, 0.4)
    with pytest.raises(SystemExit) as ei:
        _run03(res, repo)
    assert ei.value.code == 3
    assert f"{CMD} --layer 18 --alpha 0.5" in capsys.readouterr().out
    _controls(cal_dir, 0.4, 0.5)
    _run03(res, repo)
    frozen = json.loads((cal_dir / "frozen_config.json").read_text(encoding="utf-8"))
    assert frozen["induction"]["anger"] == "fallback"
    assert frozen["alpha_individual"] == {"anger": 0.5}


def test_03_freezes_refuses_rerun_and_force_supersedes(tmp_path, capsys, fake_git):
    res, cal_dir, repo = _tree(tmp_path)
    _controls(cal_dir, 0.4)
    _run03(res, repo)
    out = capsys.readouterr().out
    path = cal_dir / "frozen_config.json"
    frozen = json.loads(path.read_text(encoding="utf-8"))
    n = len(build_conditions(frozen, load_labels()))
    assert str(n) in out
    assert frozen["layer"] == 18 and frozen["alpha_max"] == 0.4
    assert frozen["resid_norm"] == 94.3 and frozen["git_commit"] == "f" * 40
    assert frozen["contamination_at_layer"] == ["anger"]
    assert frozen["induction"] == {e: "ok" for e in CORE}
    from emolab.integrity import file_sha256
    assert frozen["stats_sha256"] == file_sha256(repo / freeze.HASHED["stats_sha256"])
    before = path.read_bytes()
    with pytest.raises(SystemExit) as ei:
        _run03(res, repo)
    assert ei.value.code not in (0, None) and path.read_bytes() == before
    _run03(res, repo, "--force")
    assert len(list(cal_dir.glob("frozen_config.superseded-*.json"))) == 1
    assert json.loads(path.read_text(encoding="utf-8"))["supersedes"].startswith(
        "frozen_config.superseded-")


def test_03_refuses_with_uncommitted_hashed_files(tmp_path, fake_git):
    res, cal_dir, repo = _tree(tmp_path)
    _controls(cal_dir, 0.4)
    fake_git["dirty"] = [" M scripts/emolab/stats.py"]
    with pytest.raises(SystemExit) as ei:
        _run03(res, repo)
    assert "stats.py" in str(ei.value.code) and "提交" in str(ei.value.code)
    assert not (cal_dir / "frozen_config.json").exists()


def test_04_writes_summary(tmp_path, capsys, fake_git):
    res, cal_dir, repo = _tree(tmp_path)
    _controls(cal_dir, 0.4)
    _run03(res, repo)
    mod = _script("04_checkpoint2.py", "checkpoint04")
    mod.main(["--model", "qwen3-8b", "--results-dir", str(res), "--rate", "3"])
    md = (cal_dir / "checkpoint2_summary.md").read_text(encoding="utf-8")
    frozen = json.loads((cal_dir / "frozen_config.json").read_text(encoding="utf-8"))
    n = len(build_conditions(frozen, load_labels()))
    assert f"{n * 588:,}" in md and "anger sample 0" in md
    for b in BANNED:
        assert b not in md
