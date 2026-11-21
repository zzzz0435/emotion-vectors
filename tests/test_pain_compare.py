import importlib.util
from pathlib import Path

import numpy as np

_p = Path(__file__).parents[1] / "scripts" / "6.9_crosscheck" / "02_compare_pain_axis.py"
_spec = importlib.util.spec_from_file_location("compare02", _p)
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)


def test_compare_fields_and_planted_ranking():
    rng = np.random.default_rng(0)
    d, n = 32, 12
    labels = np.repeat(["pain", "neutral", "sadness", "fear", "anger", "joy"], n)
    acts = rng.normal(size=(len(labels), d))
    s2 = rng.normal(size=d)
    acts[labels == "pain"] += 3 * s2 / np.linalg.norm(s2)  # planted: pain - neutral ~ s2
    ours = m.label_vectors(acts, labels, ["pain", "sadness", "fear", "anger", "joy"])
    assert set(ours) == {"pain", "sadness", "fear", "anger", "joy"}
    ours["pain_painaxis"] = rng.normal(size=d)
    out = m.compare(34, {"s2": s2, "s1": rng.normal(size=d)}, ours)
    assert out["their_layer"] == out["our_layer"] == 34 and out["layer_match"] is True
    assert out["extraction_mismatch"] == "theirs final_token, ours mean"
    assert out["model_mismatch"] == "theirs base, ours chat"
    v = out["vectors"]["s2"]
    assert v["ranking"][0] == "pain" and v["cos_pain"] > 0.8
    assert abs(v["cos_pain"] - m.cosine(ours["pain"], s2)) < 1e-12
    assert set(v["cos_baselines"]) == {"sadness", "fear", "anger", "joy"}
    assert "cos_pain_painaxis" in out["vectors"]["s1"]
