import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from emolab.vectors import SEPARABLE_THRESHOLD, projection_auroc, separability

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts/6.2_emotion_vectors"


def _load(name):
    spec = importlib.util.spec_from_file_location(name.replace(".", "_"), SCRIPTS / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _two_groups(shift, n=200, d=8, seed=0):
    rng = np.random.default_rng(seed)
    acts = rng.normal(size=(2 * n, d))
    acts[:n, 0] += shift
    labels = np.array(["anger"] * n + ["fear"] * n)
    train = np.tile(np.arange(n) % 2 == 0, 2)
    return acts, labels, train, ~train


def test_separability_high_when_distinct():
    acts, labels, train, test = _two_groups(5.0)
    assert separability(acts, labels, train, test) > 0.95


def test_separability_chance_when_identical():
    acts, labels, train, test = _two_groups(0.0)
    auc = separability(acts, labels, train, test)
    assert 0.35 < auc < 0.65 and auc < SEPARABLE_THRESHOLD


def test_projection_auroc_direction():
    acts, labels, _, _ = _two_groups(5.0)
    e0 = np.eye(8)[0]
    mask = np.ones(len(labels), bool)
    assert projection_auroc(acts, labels, e0, "anger", "fear", mask) > 0.9
    assert projection_auroc(acts, labels, -e0, "anger", "fear", mask) < 0.1


def test_projection_auroc_respects_mask():
    acts, labels, _, _ = _two_groups(5.0)
    acts[:100, 0] = -50  # first 100 anger rows flipped; masking them out must hide the damage
    mask = np.ones(len(labels), bool)
    mask[:100] = False
    assert projection_auroc(acts, labels, np.eye(8)[0], "anger", "fear", mask) > 0.9


def test_validate_script_tables(tmp_path):
    val = _load("06_validate_vectors.py")
    rng = np.random.default_rng(1)
    d, n = 6, 40
    labels = np.array(["neutral"] * n + ["anger"] * n + ["pain"] * n)
    acts = rng.normal(size=(len(labels), d))
    acts[n:2 * n, 0] += 6
    acts[2 * n:, 1] += 6
    held = np.tile(np.arange(n) % 2 == 0, 3)
    index = pd.DataFrame({"label": labels, "heldout": np.where(held, "True", "False")})
    unit = {"anger": np.eye(d)[[0]], "pain_painaxis": np.eye(d)[[1]]}  # [n_layers=1, d]
    df = val.validation_table({3: acts}, index, unit, [3], neutral="neutral")
    assert list(df.columns) == ["label", "layer", "auroc_vs_neutral"]
    assert set(df.label) == {"anger", "pain_painaxis"}
    assert (df.auroc_vs_neutral > 0.95).all()

    sim = val.similarity_matrix({"a": np.array([1., 0]), "b": np.array([0., 1]), "c": np.array([1., 1])})
    np.testing.assert_allclose(np.diag(sim.to_numpy()), 1.0)
    assert abs(sim.loc["a", "b"]) < 1e-9
    assert abs(sim.loc["a", "c"] - 2 ** -0.5) < 1e-9
    out = tmp_path / "sim.png"
    val.plot_similarity(sim, 3, out)
    assert out.stat().st_size > 0
