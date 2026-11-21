import numpy as np
import torch

from emolab.sae import load_decoder, nearest_features, top_stories_per_feature


def test_load_decoder_transposes_and_normalizes(tmp_path):
    torch.save({"W_dec": torch.randn(16, 100)}, tmp_path / "s.pt")  # stored as [d, n]
    W = load_decoder(tmp_path / "s.pt", d_model=16)
    assert W.shape == (100, 16)
    np.testing.assert_allclose(np.linalg.norm(W, axis=1), 1, atol=1e-6)


def test_load_decoder_keeps_n_by_d_layout(tmp_path):
    torch.save({"W_dec": torch.randn(100, 16), "W_enc": torch.randn(16, 100)}, tmp_path / "s.pt")
    W = load_decoder(tmp_path / "s.pt", d_model=16)
    assert W.shape == (100, 16)


def test_nearest_features_finds_planted():
    rng = np.random.default_rng(0)
    W = rng.normal(size=(100, 16))
    W /= np.linalg.norm(W, axis=1, keepdims=True)
    vec = rng.normal(size=16)
    W[37] = vec / np.linalg.norm(vec)
    top = nearest_features(vec * 5, W, k=5)
    assert top[0][0] == 37 and abs(top[0][1] - 1.0) < 1e-6
    assert len(top) == 5 and [c for _, c in top] == sorted((c for _, c in top), reverse=True)


def test_top_stories_per_feature():
    rng = np.random.default_rng(1)
    W = np.eye(4, 8)  # features 0..3 = axes 0..3
    acts = rng.normal(scale=0.1, size=(20, 8))
    acts[[3, 7, 11, 15, 19, 2], 1] += [9, 8, 7, 6, 5, 4]  # story rows strongest on feature 1
    out = top_stories_per_feature(acts, W, [1, 2], n=5)
    assert out[1] == [3, 7, 11, 15, 19]
    assert len(out[2]) == 5
