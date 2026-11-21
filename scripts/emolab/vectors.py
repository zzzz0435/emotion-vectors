"""Direction vectors from mean activations (difference of means, Pain-axis denoised vector)."""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

SEPARABLE_THRESHOLD = 0.75  # anger vs fear held-out AUROC needed to call them separable


def as_bool(s: pd.Series) -> pd.Series:
    """index.csv may round-trip `heldout` as bool or as 'True'/'False' strings."""
    return s.astype(str).str.strip().str.lower().eq("true")


def diff_of_means(acts: np.ndarray, labels: np.ndarray, target: str, baseline: str,
                  mask: np.ndarray) -> np.ndarray:
    """mean(acts[label == target]) - mean(acts[label == baseline]), over rows where mask is True."""
    labels = np.asarray(labels)
    mask = np.asarray(mask, dtype=bool)
    tgt = acts[(labels == target) & mask]
    base = acts[(labels == baseline) & mask]
    if len(tgt) == 0 or len(base) == 0:
        raise ValueError(f"no masked rows for {target!r} ({len(tgt)}) or {baseline!r} ({len(base)})")
    return tgt.mean(axis=0) - base.mean(axis=0)


def pain_axis_vector(pain_acts: np.ndarray, control_acts: np.ndarray,
                     denoise_variance: float = 0.5) -> np.ndarray:
    """Pain-axis vector: mean(pain) - mean(controls), with the leading control PCs projected out.

    Ported from `compute_pain_vector` (baseline="all_controls", denoise=True) in
    scripts/3.2_pain_vectors/01_extract_activations_and_pain_vectors.py of
    https://github.com/valen-research/Pain-axis at commit
    213fb6c89c96004dbe93d1704c636ad8c9d801f3 (MIT License). The original takes all
    activations plus category codes; here the pain / control split is done by the caller.
    PCA is fit on the centred control activations and components are removed until
    `denoise_variance` of the control variance is covered (the first component that crosses
    the threshold is included, as in the original).
    """
    pain_acts = np.asarray(pain_acts, dtype=np.float64)
    control_acts = np.asarray(control_acts, dtype=np.float64)
    denoise = True
    if not (np.isfinite(pain_acts).all() and np.isfinite(control_acts).all()):
        denoise = False  # original: skip denoising, treat inf as nan, nan-aware means
        pain_acts = np.where(np.isinf(pain_acts), np.nan, pain_acts)
        control_acts = np.where(np.isinf(control_acts), np.nan, control_acts)

    pain_mean = np.nanmean(pain_acts, axis=0)
    control_mean = np.nanmean(control_acts, axis=0)
    vec = np.nan_to_num(pain_mean - control_mean, nan=0.0, posinf=0.0, neginf=0.0)

    if denoise and len(control_acts) > 1:
        pca = PCA()
        pca.fit(control_acts - control_mean)
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        n_comp = min(np.searchsorted(cumvar, denoise_variance) + 1, len(pca.components_))
        for d in pca.components_[:n_comp]:
            vec = vec - np.dot(vec, d) * d
    return vec


def projection_auroc(acts: np.ndarray, labels: np.ndarray, unit_vec: np.ndarray,
                     pos: str, neg: str, mask: np.ndarray) -> float:
    """AUROC of the projection onto `unit_vec` for `pos` (positive class) vs `neg`, over masked rows."""
    labels = np.asarray(labels)
    mask = np.asarray(mask, dtype=bool)
    sel = mask & ((labels == pos) | (labels == neg))
    if not ((labels[sel] == pos).any() and (labels[sel] == neg).any()):
        raise ValueError(f"need masked rows for both {pos!r} and {neg!r}")
    return float(roc_auc_score(labels[sel] == pos, acts[sel] @ unit_vec))


def separability(acts: np.ndarray, labels: np.ndarray, train_mask: np.ndarray,
                 test_mask: np.ndarray, a: str = "anger", b: str = "fear") -> float:
    """Direction unit(mean_a - mean_b) from training rows; AUROC of a vs b on test rows."""
    d = diff_of_means(acts, labels, a, b, train_mask).astype(np.float64)
    return projection_auroc(acts, labels, d / np.linalg.norm(d), a, b, test_mask)
