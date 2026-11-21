import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from emolab import hooks

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/6.2_emotion_vectors/03_extract_activations.py"
spec = importlib.util.spec_from_file_location("extract_script", SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

TEXTS = ["abcdefgh", "ab", "abcdefghijkl", "abcd", "abcdef"]
ROWS_5 = [{"id": f"s{i}", "label": "x", "kind": "neutral", "topic_idx": i, "subtype": None,
           "heldout": i == 4, "text": t} for i, t in enumerate(TEXTS)]


def fake_encode(texts):
    n = max(len(t) for t in texts)
    ids = torch.zeros(len(texts), n, dtype=torch.long)
    mask = torch.zeros(len(texts), n, dtype=torch.long)
    for i, t in enumerate(texts):  # left padding
        ids[i, n - len(t):] = torch.tensor([1 + ord(c) % 100 for c in t])
        mask[i, n - len(t):] = 1
    return {"input_ids": ids, "attention_mask": mask}


def test_extract_shapes_and_order(tiny_qwen3, tmp_path):
    idx = mod.extract(tiny_qwen3, fake_encode, ROWS_5, [1, 2], batch_size=2, out_dir=tmp_path)
    a = np.load(tmp_path / "acts_L1.npy")
    assert a.shape == (5, 32) and a.dtype == np.float32
    assert list(idx.id) == [r["id"] for r in ROWS_5]
    # row order matches input despite length-sorted batching
    enc = fake_encode([TEXTS[2]])
    ref = hooks.capture_mean_activations(tiny_qwen3, enc["input_ids"], enc["attention_mask"], [1])[1]
    np.testing.assert_allclose(a[2], ref[0].numpy(), atol=1e-5)
    assert list(idx.heldout) == [False] * 4 + [True]


def test_extract_skips_when_done(tiny_qwen3, tmp_path, monkeypatch):
    mod.extract(tiny_qwen3, fake_encode, ROWS_5, [1, 2], batch_size=2, out_dir=tmp_path)

    def boom(*a, **k):
        raise AssertionError("capture called")
    monkeypatch.setattr(hooks, "capture_mean_activations", boom)
    idx = mod.extract(tiny_qwen3, fake_encode, ROWS_5, [1, 2], batch_size=2, out_dir=tmp_path)
    assert list(idx.id) == [r["id"] for r in ROWS_5]
    with pytest.raises(AssertionError):  # force and changed ids both recompute
        mod.extract(tiny_qwen3, fake_encode, ROWS_5, [1, 2], 2, tmp_path, force=True)
    with pytest.raises(AssertionError):
        mod.extract(tiny_qwen3, fake_encode, ROWS_5[:4], [1, 2], 2, tmp_path)


def test_resid_norms_matches_direct(tiny_qwen3):
    got = mod.resid_norms(tiny_qwen3, fake_encode, ROWS_5, [1, 3], batch_size=2)
    per_tok = []
    for r in ROWS_5:
        e = fake_encode([r["text"]])
        per_tok.append((hooks.mean_token_norm(tiny_qwen3, e["input_ids"], e["attention_mask"], [3])[3],
                        len(r["text"]) - 1))
    want = sum(v * n for v, n in per_tok) / sum(n for _, n in per_tok)
    assert got[3] == pytest.approx(want, rel=1e-5)
