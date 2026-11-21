import importlib.util
from pathlib import Path

import numpy as np
import pytest

from emolab.vectors import contamination_flags, remove_component, remove_components

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts/6.2_emotion_vectors"
D = 16


def _e(i):
    v = np.zeros(D)
    v[i] = 1.0
    return v


def _load():
    spec = importlib.util.spec_from_file_location("contam07", SCRIPTS / "07_contamination.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _unit_set():
    """p = e0, q = e1; anger = 0.8 p + 0.6 u; sadness/joy orthogonal to p; 40 low-cos concepts."""
    rng = np.random.default_rng(0)
    p, q, u = _e(0), _e(1), _e(2)
    unit = {"punishment": p, "harming_others": q,
            "anger": 0.8 * p + 0.6 * u, "sadness": _e(3), "joy": _e(4)}
    concepts = []
    for i in range(40):
        c0 = 0.09 * i / 39
        v = rng.normal(size=D)
        v[0] = 0.0
        v = v / np.linalg.norm(v)
        v = np.sqrt(1 - c0 * c0) * v + c0 * p  # cos with p = c0, spread over 0..0.09
        unit[f"c{i}"] = v
        concepts.append(f"c{i}")
    return unit, concepts


def test_flags_contaminated_and_cleans():
    unit, concepts = _unit_set()
    res = contamination_flags(unit, ["anger", "joy", "sadness"], concepts, "punishment")
    assert res["anger"]["flagged"] and not res["joy"]["flagged"]
    assert not res["sadness"]["flagged"]  # never flagged against itself
    assert res["anger"]["cos"] == pytest.approx(0.8)
    assert all(isinstance(x, (float, bool)) for x in res["anger"].values())
    assert res["anger"]["p95_concepts"] < 0.1
    c = remove_component(unit["anger"], unit["punishment"])
    assert abs(c @ unit["punishment"]) < 1e-6
    assert np.linalg.norm(c) == pytest.approx(1.0)


def test_requires_exceeding_both_thresholds():
    unit, concepts = _unit_set()
    unit["sadness"] = 0.9 * _e(0) + np.sqrt(1 - 0.81) * _e(3)  # reference above anger
    assert not contamination_flags(unit, ["anger"], concepts, "punishment")["anger"]["flagged"]


def test_two_probes_removed_orthogonally():
    p = _e(0)
    q = (_e(0) + _e(1)) / np.sqrt(2)  # correlated probes
    v = 0.5 * _e(0) + 0.5 * _e(1) + 0.7 * _e(2)
    c = remove_components(v, [p, q])
    assert abs(c @ p) < 1e-6 and abs(c @ q) < 1e-6
    assert np.linalg.norm(c) == pytest.approx(1.0)


def test_script_cleans_per_layer(tmp_path):
    mod = _load()
    unit, concepts = _unit_set()
    layers = [9, 14]
    # anger contaminated by punishment at layer 9 only
    stack = {k: np.stack([v, v]) for k, v in unit.items()}
    stack["anger"][1] = _e(5)
    flags, clean, meta = mod.run_check(stack, layers, ["anger", "joy", "sadness"], concepts,
                                       ["punishment", "harming_others"])
    assert flags["9"]["punishment"]["anger"]["flagged"]
    assert not flags["14"]["punishment"]["anger"]["flagged"]
    assert list(clean) == ["anger__clean"]
    assert meta["cleaned_layers"] == {"anger": [9]}
    assert abs(clean["anger__clean"][0] @ unit["punishment"]) < 1e-6
    np.testing.assert_allclose(clean["anger__clean"][1], stack["anger"][1])  # untouched layer


def _with_cos(c, axis):
    """Unit vector with cosine c to e0, remainder along e<axis>."""
    return c * _e(0) + np.sqrt(1 - c * c) * _e(axis)


def test_p95_is_exact_and_gates_flag():
    unit, concepts = _unit_set()
    unit["sadness"] = _with_cos(-0.1, 3)
    unit["mild"] = _with_cos(0.02, 5)  # beats sadness, but below the control p95 (~0.09)
    res = contamination_flags(unit, ["mild"], concepts, "punishment")["mild"]
    ctrl = [unit[c] @ _e(0) / np.linalg.norm(unit[c]) for c in concepts]
    assert res["p95_concepts"] == pytest.approx(float(np.percentile(ctrl, 95)), abs=1e-12)
    assert res["p95_concepts"] != pytest.approx(float(np.median(ctrl)), abs=1e-6)
    assert res["cos"] > res["ref_cos"] and res["cos"] < res["p95_concepts"]
    assert not res["flagged"]


def test_tie_with_reference_is_not_flagged():
    unit, concepts = _unit_set()
    unit["sadness"] = _with_cos(0.3, 3)
    unit["twin"] = unit["sadness"].copy()  # exactly sadness's cosine, above p95
    res = contamination_flags(unit, ["twin"], concepts, "punishment")["twin"]
    assert res["cos"] == res["ref_cos"] and res["cos"] > res["p95_concepts"]
    assert not res["flagged"]


def test_nothing_flagged_deletes_stale_vectors_clean(tmp_path):
    m = _load()
    stale = tmp_path / "vectors_clean.pt"
    stale.write_bytes(b"old")
    m.write_clean(tmp_path, [18], {}, {})
    assert not stale.exists()
    m.write_clean(tmp_path, [18], {}, {})  # no file: still fine
    m.write_clean(tmp_path, [18], {"anger": np.ones((1, 4), dtype=np.float32)}, {})
    assert stale.exists()
