import json

import numpy as np
import pytest
import torch

from emolab.conditions import (NAME_RE, Condition, build_conditions, condition_name,
                               load_labels, load_stores, random_directions,
                               resolve_direction)

CORE = ["anger", "fear", "disgust", "desperation", "pain", "compassion"]
LABELS = {"core": CORE, "map": [f"map{i}" for i in range(14)],
          "concepts": [f"concept{i}" for i in range(40)],
          "induce": {e: f"You feel {e}." for e in CORE}}
EQ = {"anger": 0.7312, "fear": 0.3, "disgust": 0.81, "desperation": 0.66, "pain": 0.4,
      "compassion": 0.9}
FROZEN_L18 = {
    "model": "Qwen/Qwen3-8B", "layer": 18, "resid_norm": 100.0, "alpha_max": 1.0,
    "alpha_eq": dict(EQ), "alpha_eq_target": 0.1, "alpha_individual": {},
    "induction": {e: "ok" for e in CORE}, "coherent_max": 2.0, "thresholds": {}, "grid": [],
    "contamination_at_layer": [], "separability_at_layer": {"auroc": 0.9, "passed": True},
    "residual_status_at_layer": {n: "ok" for n in ("anger__resid3", "anger__resid2",
                                                   "pain__resid3", "compassion__resid3")},
    "stats_sha256": "x", "conditions_sha256": "y", "git_commit": "z", "created_at": "t"}
FROZEN_L14_PAIN_FALLBACK = {
    **FROZEN_L18, "layer": 14,
    "contamination_at_layer": ["anger", "fear", "disgust", "desperation"],
    "alpha_individual": {"pain": 1.6}, "induction": {**FROZEN_L18["induction"], "pain": "fallback"}}


def test_random_directions_unit_and_in_activation_span():
    rng = np.random.default_rng(0)
    acts = rng.normal(size=(50, 8)) @ np.diag([5, 1, 1, 1, 0, 0, 0, 0])
    r = random_directions(acts, 20)
    assert np.allclose(np.linalg.norm(r, axis=1), 1)
    assert np.abs(r[:, 4:]).max() < 1e-9


def test_random_directions_degenerate_raises():
    with pytest.raises(ValueError):
        random_directions(np.ones((5, 4)), 3)


def test_build_conditions_counts_without_fallback_or_clean():
    conds = build_conditions(FROZEN_L18, LABELS)
    assert len(conds) == 1+24+6+4+6+2+6+28+6+80+40+20
    assert len({c.name for c in conds}) == len(conds)
    assert conds[0].name == "baseline"


def test_clean_and_fallback_conditions_added():
    conds = build_conditions(FROZEN_L14_PAIN_FALLBACK, LABELS)
    assert any(c.name.startswith("anger__clean") for c in conds)
    assert sum(c.kind == "concept" and c.alpha == FROZEN_L14_PAIN_FALLBACK["alpha_individual"]["pain"]
               for c in conds) == 40
    assert len({c.name for c in conds}) == len(conds)
    assert condition_name("pain", 1.6) in {c.name for c in conds}


def test_names_filename_safe_and_examples():
    for f in (FROZEN_L18, FROZEN_L14_PAIN_FALLBACK):
        assert all(NAME_RE.match(c.name) for c in build_conditions(f, LABELS))
    assert condition_name("anger", 1.0) == "anger@+1.0000"
    assert condition_name("anger", -0.5) == "anger@-0.5000"
    assert condition_name("anger", 0.7312, eq=True) == "anger@eq+0.7312"
    with pytest.raises(ValueError):
        condition_name("bad name", 1.0)


def test_residual_omitted_when_untestable():
    f = {**FROZEN_L18, "residual_status_at_layer": {**FROZEN_L18["residual_status_at_layer"],
                                                    "pain__resid3": "untestable"}}
    f["residual_status_at_layer"].pop("compassion__resid3")
    names = {c.name for c in build_conditions(f, LABELS)}
    assert not any(n.startswith(("pain__resid3", "compassion__resid3")) for n in names)
    assert "anger__resid3@+0.5000" in names


def test_eq_omitted_when_alpha_eq_none():
    f = {**FROZEN_L14_PAIN_FALLBACK, "alpha_eq": {**EQ, "anger": None, "pain": None}}
    names = {c.name for c in build_conditions(f, LABELS)}
    assert not any(n.startswith(("anger@eq", "pain@eq", "anger__clean@eq")) for n in names)
    assert "fear@eq+0.3000" in names and "fear__clean@eq+0.3000" in names


def test_clean_only_core_and_anger_resid_pair():
    names = {c.name for c in build_conditions(FROZEN_L14_PAIN_FALLBACK, LABELS)}
    clean = {n.split("@")[0] for n in names if "__clean" in n}
    assert clean == {"anger__clean", "fear__clean", "disgust__clean", "desperation__clean",
                     "anger__clean__resid3"}
    assert {"anger__clean__resid3@+1.0000", "anger__clean__resid3@-1.0000"} <= names
    for e in ("anger", "fear"):
        assert {f"{e}__clean@+1.0000", f"{e}__clean@-1.0000", f"{e}__clean@eq{EQ[e]:+.4f}"} <= names
    assert not any(n.startswith(("pain__clean", "compassion__clean")) for n in names)


FROZEN_L14_ANGER_FALLBACK = {
    **FROZEN_L14_PAIN_FALLBACK, "alpha_individual": {"pain": 1.6, "anger": 1.4},
    "induction": {**FROZEN_L14_PAIN_FALLBACK["induction"], "anger": "fallback"}}


def test_fallback_and_contaminated_gets_clean_at_individual_alpha():
    # pain is fallback but NOT contaminated in the L14 fixture -> no pain__clean, count unchanged
    names = {c.name for c in build_conditions(FROZEN_L14_PAIN_FALLBACK, LABELS)}
    assert len(names) == 298 and not any(n.startswith("pain__clean") for n in names)
    # anger is fallback AND contaminated -> anger__clean@+α_ind (R20)
    conds = {c.name: c for c in build_conditions(FROZEN_L14_ANGER_FALLBACK, LABELS)}
    assert len(conds) == 298 + 2 + 40 + 20      # anger, anger__clean, concepts, randoms @1.4
    c = conds["anger__clean@+1.4000"]
    assert (c.kind, c.source, c.alpha) == ("clean", "clean:anger__clean", 1.4)
    assert "anger@+1.4000" in conds and not any(n.startswith("pain__clean") for n in conds)


def test_text_and_baseline():
    conds = {c.name: c for c in build_conditions(FROZEN_L18, LABELS)}
    assert conds["text_anger"].induce == "You feel anger." and conds["text_anger"].alpha == 0.0
    assert conds["baseline"].source is None


def _stores():
    g = torch.Generator().manual_seed(0)
    d, layers = 8, [14, 18]

    def t(*s):
        return torch.randn(*s, generator=g, dtype=torch.float64)

    meta = {"V": {"reliable": True}, "A": {"reliable": False}, "D": {"reliable": True}}
    return {
        "vectors": {"layers": layers, "unit": {k: t(2, d) * 3 for k in ("anger", "pain_painaxis", "c0")}},
        "clean": {"layers": layers, "unit": {"anger__clean": t(2, d)}},
        "residuals": {"layers": layers, "unit": {"anger__resid3": t(2, d)},
                      "status": {"anger__resid3": {14: "ok", 18: "ok"}}},
        "axes": {"fits": {"all": {L: {k: t(d) for k in "VAD"} for L in layers}},
                 "raw_fits": {"for_anger": {L: {k: t(d) for k in "VAD"} for L in layers}}},
        "axes_meta": {"fits": {"for_anger": {str(L): meta for L in layers}}},
        "random": np.random.default_rng(0).normal(size=(20, d))}


def _cond(src):
    return Condition("x", "k", src, 1.0)


def test_resolve_direction_all_sources():
    s = _stores()
    for src in ("vectors:unit:anger", "vectors:unit:pain_painaxis", "vectors:unit:c0",
                "clean:anger__clean", "residual:anger__resid3", "axis:all:V", "random:7",
                "clean_residual:anger__resid3"):
        v = resolve_direction(_cond(src), 18, s)
        assert v.dtype == np.float64 and v.shape == (8,) and abs(np.linalg.norm(v) - 1) < 1e-9
    v = resolve_direction(_cond("clean_residual:anger__resid3"), 14, s)
    for k in "VD":
        assert abs(v @ s["axes"]["raw_fits"]["for_anger"][14][k].numpy()) < 1e-9
    assert resolve_direction(Condition("baseline", "baseline", None, 0.0), 18, s) is None
    assert resolve_direction(Condition("t", "text", None, 0.0, "s"), 18, s) is None


def test_resolve_direction_errors():
    s = _stores()
    for src, layer in (("bogus:x", 18), ("vectors:unit:nope", 18), ("vectors:unit:anger", 99)):
        with pytest.raises(ValueError):
            resolve_direction(_cond(src), layer, s)
    with pytest.raises(ValueError):
        resolve_direction(_cond("random:1"), 18, {**s, "random": None})
    s["axes_meta"]["fits"]["for_anger"]["18"] = {"V": {"reliable": False}}
    with pytest.raises(ValueError):
        resolve_direction(_cond("clean_residual:anger__resid3"), 18, s)


def test_load_labels_real_datasets():
    lab = load_labels()
    assert len(lab["core"]) == 6 and len(lab["map"]) == 14 and len(lab["concepts"]) == 40
    assert not {"punishment", "harming_others"} & set(lab["concepts"])
    assert set(lab["induce"]) == set(lab["core"])


def test_load_stores_tmp(tmp_path, monkeypatch):
    import emolab.conditions as cm
    monkeypatch.setattr(cm, "RESULTS", tmp_path)
    m = "M"
    for sub, files in (("6.2_emotion_vectors", ("vectors.pt", "vectors_clean.pt")),
                       ("6.3_axes", ("residuals.pt", "axes.pt"))):
        (tmp_path / sub / m).mkdir(parents=True)
        for f in files:
            torch.save({"f": f}, tmp_path / sub / m / f)
    (tmp_path / "6.3_axes" / m / "axes.json").write_text(json.dumps({"fits": {}}))
    s = load_stores(m, 18)
    assert s["random"] is None and s["vectors"] == {"f": "vectors.pt"} and s["axes_meta"] == {"fits": {}}
    (tmp_path / "6.7_runs" / m).mkdir(parents=True)
    np.save(tmp_path / "6.7_runs" / m / "random_directions_L18.npy", np.zeros((2, 3)))
    assert load_stores(m, 18)["random"].shape == (2, 3)
