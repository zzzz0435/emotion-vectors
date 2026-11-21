"""Sparse-autoencoder (SAE) decoder helpers for the cross-check against an official dictionary."""
from pathlib import Path

import numpy as np
import torch

DECODER_KEYS = ("W_dec", "decoder.weight", "decoder", "dec.weight")


def _find_decoder(obj, d_model: int):
    """The decoder tensor: a known key first, else the only 2-D tensor with a d_model side."""
    if not isinstance(obj, dict):
        raise ValueError(f"SAE file holds a {type(obj).__name__}, expected a dict")
    for nested in ("state_dict", "model"):
        if nested in obj and isinstance(obj[nested], dict):
            obj = obj[nested]
    for key in DECODER_KEYS:
        if key in obj:
            return key, obj[key]
    cands = [(k, v) for k, v in obj.items() if isinstance(v, torch.Tensor) and v.ndim == 2
             and d_model in v.shape and "dec" in k.lower()]
    if len(cands) == 1:
        return cands[0]
    raise ValueError(f"cannot find the decoder; keys: {sorted(obj)}")


def load_decoder(path: Path, d_model: int) -> np.ndarray:
    """[n_features, d_model] float32 decoder with every feature direction unit-normalised.

    A [d_model, n] matrix is transposed. Only the decoder is kept (the file is memory-mapped
    when possible); rows are normalised in float32.
    """
    try:
        obj = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    except Exception:  # legacy (non-zipfile) files cannot be mmapped
        obj = torch.load(path, map_location="cpu", weights_only=True)
    key, W = _find_decoder(obj, d_model)
    W = W.detach().to(torch.float32)
    if W.shape[1] == d_model and W.shape[0] != d_model:
        pass
    elif W.shape[0] == d_model:
        W = W.T
    else:
        raise ValueError(f"decoder {key!r} has shape {tuple(W.shape)}, no side equals d_model={d_model}")
    W = W.contiguous().clone()
    del obj
    return (W / W.norm(dim=1, keepdim=True).clamp_min(1e-12)).numpy()


def nearest_features(vec: np.ndarray, W_dec: np.ndarray, k: int = 10) -> list[tuple[int, float]]:
    """(feature index, cosine) of the k decoder rows closest to vec, highest cosine first.

    W_dec rows are assumed unit-norm (load_decoder guarantees it)."""
    v = np.asarray(vec, dtype=np.float32)
    v = v / np.linalg.norm(v)
    cos = W_dec @ v
    top = np.argsort(-cos)[:k]
    return [(int(i), float(cos[i])) for i in top]


def top_stories_per_feature(acts: np.ndarray, W_dec: np.ndarray, features: list[int],
                            n: int = 5) -> dict[int, list[int]]:
    """For each feature, the row indices of `acts` ([n_stories, d]) with the largest projection
    on that decoder direction, largest first."""
    out = {}
    for f in features:
        proj = np.asarray(acts, dtype=np.float32) @ W_dec[f]
        out[int(f)] = [int(i) for i in np.argsort(-proj)[:n]]
    return out
