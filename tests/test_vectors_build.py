import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from emolab.vectors import diff_of_means, pain_axis_vector

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts/6.2_emotion_vectors"


def _load(name):
    spec = importlib.util.spec_from_file_location(name.replace(".", "_"), SCRIPTS / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


build = _load("04_build_vectors.py")
pain = _load("05_pain_axis_vector.py")


def test_diff_of_means_simple():
    acts = np.array([[1., 0], [1, 0], [0, 0], [0, 0]])
    labels = np.array(["a", "a", "n", "n"])
    np.testing.assert_allclose(diff_of_means(acts, labels, "a", "n", np.ones(4, bool)), [1, 0])


def test_mask_excludes_heldout():
    acts = np.array([[1., 0], [1, 0], [1e6, 1e6], [0, 0], [0, 0], [-1e6, 5e5]])
    labels = np.array(["a", "a", "a", "n", "n", "n"])
    mask = np.array([True, True, False, True, True, False])
    np.testing.assert_allclose(diff_of_means(acts, labels, "a", "n", mask), [1, 0])


def test_pain_axis_denoise_removes_dominant_control_direction():
    rng = np.random.default_rng(0)
    ctrl = rng.normal(size=(200, 4)) * [10, 1, 1, 1]
    pain_acts = ctrl[:50] + [3, 2, 0, 0]
    v = pain_axis_vector(pain_acts, ctrl, 0.5)
    assert abs(v[0]) < 0.1 * abs(v[1])


def test_pain_axis_no_denoise_when_variance_zero_keeps_most_signal():
    rng = np.random.default_rng(1)
    ctrl = rng.normal(size=(100, 3))
    p = ctrl[:20] + [0, 0, 5]
    v = pain_axis_vector(p, ctrl, 0.0)  # one component removed at minimum (searchsorted + 1)
    assert v.shape == (3,) and np.isfinite(v).all()


# ---- 04 -------------------------------------------------------------------

def _index(labels, heldout):
    return pd.DataFrame({"row": range(len(labels)), "id": [f"i{i}" for i in range(len(labels))],
                         "label": labels, "kind": "x", "topic_idx": 0, "subtype": None,
                         "heldout": heldout})


def test_build_vectors_values_and_unit():
    labels = ["a", "a", "a", "n", "n", "n"]
    heldout = [False, False, True, False, False, True]
    index = _index(labels, heldout)
    acts = {3: np.array([[2., 0], [2, 0], [99, 99], [0, 0], [0, 0], [-99, 7]], np.float32),
            5: np.array([[0., 3], [0, 3], [99, 99], [0, 0], [0, 0], [-99, 7]], np.float32)}
    out = build.build_vectors(acts, index, ["a"], "n", {3: 1.0, 5: 2.0}, "m")
    assert out["layers"] == [3, 5]
    assert out["resid_norm"] == {3: 1.0, 5: 2.0}
    raw = out["raw"]["a"]
    assert raw.dtype == torch.float32 and raw.shape == (2, 2)
    torch.testing.assert_close(raw, torch.tensor([[2., 0], [0, 3]]))
    torch.testing.assert_close(out["unit"]["a"], torch.tensor([[1., 0], [0, 1]]))
    assert out["meta"] == {"model": "m", "n_train": {"a": 2, "n": 2}, "heldout_topics": [0]}
    assert set(out["raw"]) == set(out["unit"])


def test_build_vectors_names_label_with_no_training_rows():
    index = _index(["a", "n", "n", "b"], [False, False, False, True])  # b only heldout
    acts = {0: np.zeros((4, 2), np.float32)}
    with pytest.raises(ValueError, match="'b'"):
        build.build_vectors(acts, index, ["a", "b"], "n", {0: 1.0}, "m")


def test_build_vectors_missing_neutral_fails():
    index = _index(["a", "a"], [False, False])
    with pytest.raises(ValueError, match="'n'"):
        build.build_vectors({0: np.zeros((2, 2), np.float32)}, index, ["a"], "n", {0: 1.0}, "m")


def test_normalize_heldout_strings():
    s = pd.Series(["True", "False", "true", "False"])
    assert build.as_bool(s).tolist() == [True, False, True, False]
    assert build.as_bool(pd.Series([True, False])).tolist() == [True, False]


def test_expected_labels_counts():
    emo, conc, neutral = build.expected_labels()
    assert len(emo) == 20 and len(conc) == 42 and neutral == "neutral"
    assert "punishment" in conc and "pain" in emo


# ---- 05 -------------------------------------------------------------------

def test_select_pain_rows_categories():
    data = {"metadata": {}, "datasets": {"S2_1P": {"total": 7, "sentences": [
        {"category": c, "set": 1, "prompt": f"{c} text"}
        for c in ["A1", "A2", "A3", "A4", "A5", "B", "C1", "C2", "D", "E"]]},
        "S2_3P": {"sentences": [{"category": "A1", "set": 1, "prompt": "other"}]}}}
    pain_t, ctrl_t = pain.select_pain_rows(data)
    assert pain_t == ["A1 text", "A2 text", "A3 text", "A5 text"]  # A4 excluded
    assert ctrl_t == ["B text", "C1 text", "C2 text", "D text", "E text"]


def test_add_pain_vector_keeps_other_keys(tmp_path):
    p = tmp_path / "vectors.pt"
    base = {"layers": [1, 2], "resid_norm": {1: 1.0, 2: 1.0},
            "raw": {"a": torch.ones(2, 3)}, "unit": {"a": torch.ones(2, 3) / 3 ** .5},
            "meta": {"model": "m"}}
    torch.save(base, p)
    vec = {1: np.array([3., 0, 4]), 2: np.array([0., 2, 0])}
    pain.add_pain_vector(p, vec)
    out = torch.load(p)
    assert set(out["raw"]) == {"a", "pain_painaxis"} == set(out["unit"])
    assert out["meta"] == {"model": "m"} and out["layers"] == [1, 2]
    torch.testing.assert_close(out["raw"]["pain_painaxis"], torch.tensor([[3., 0, 4], [0, 2, 0]]))
    torch.testing.assert_close(out["unit"]["pain_painaxis"], torch.tensor([[.6, 0, .8], [0, 1, 0]]))
    assert out["raw"]["pain_painaxis"].dtype == torch.float32
