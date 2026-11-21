"""General valence / arousal / dominance axes and residual emotion vectors (design 6.3)."""
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

ALPHAS = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
RELIABLE_R = 0.5
RATING_COLS = {"V": "valence", "A": "arousal", "D": "dominance"}

# Fit sets: emotions left out of the fit so a residual is not circular with its own axes.
FIT_SETS = {
    "all": set(),
    "for_anger": {"anger", "fear"},
    "for_pain": {"pain"},
    "for_compassion": {"compassion"},
}
# residual name -> (emotion, fit set, axes to remove)
RESIDUALS = {
    "anger__resid3": ("anger", "for_anger", "VAD"),
    "anger__resid2": ("anger", "for_anger", "VA"),
    "pain__resid3": ("pain", "for_pain", "VAD"),
    "compassion__resid3": ("compassion", "for_compassion", "VAD"),
}


def _fit(X: np.ndarray, y: np.ndarray) -> RidgeCV:
    """RidgeCV with its default efficient leave-one-out for alpha selection.

    That inner LOO only picks alpha (only ~18-20 samples); the number to report as axis
    validation is the outer `loo_r`, which refits this whole procedure per held-out sample.
    """
    return RidgeCV(alphas=ALPHAS, fit_intercept=True).fit(X, y)


def fit_axis(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    model = _fit(X, y)
    coef = np.asarray(model.coef_, dtype=np.float64)
    return coef / np.linalg.norm(coef), float(model.alpha_)


def loo_r(X: np.ndarray, y: np.ndarray) -> float:
    """Pearson r between true y and predictions made without the sample (alpha re-chosen
    inside every fold; prediction = intercept + coef @ x)."""
    n = len(y)
    preds = np.empty(n)
    for i in range(n):
        keep = np.arange(n) != i
        preds[i] = _fit(X[keep], y[keep]).predict(X[i:i + 1])[0]
    if np.std(preds) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(preds, y)[0, 1])


def fit_axes(unit: dict[str, np.ndarray], vad: pd.DataFrame, exclude: set[str]) -> dict:
    """Fit V, A, D axes on unit emotion vectors (labels in `exclude` or without a vector are
    skipped), then Gram-Schmidt V -> A -> D. loo_r / reliable come from each axis's own raw
    fit, before orthogonalisation."""
    rows = vad[vad["label"].isin(unit) & ~vad["label"].isin(exclude)]
    X = np.stack([np.asarray(unit[lab], dtype=np.float64) for lab in rows["label"]])
    out, basis = {}, []
    for key, col in RATING_COLS.items():
        y = rows[col].to_numpy(dtype=np.float64)
        axis, alpha = fit_axis(X, y)
        r = loo_r(X, y)
        for b in basis:
            axis = axis - (axis @ b) * b
        axis = axis / np.linalg.norm(axis)
        basis.append(axis)
        out[key] = {"axis": axis, "alpha": alpha, "loo_r": r, "reliable": bool(r >= RELIABLE_R)}
    return out


def residual(v: np.ndarray, axes: list[np.ndarray]) -> np.ndarray:
    """Unit vector with every given axis component removed (axes are re-orthonormalised
    first, so the result is exactly orthogonal to each, even if they overlap)."""
    v = np.asarray(v, dtype=np.float64)
    basis = []
    for a in axes:
        a = np.asarray(a, dtype=np.float64)
        for b in basis:
            a = a - (a @ b) * b
        basis.append(a / np.linalg.norm(a))
    for b in basis:
        v = v - (v @ b) * b
    return v / np.linalg.norm(v)


def residual_vectors(emotion: str, v: np.ndarray, fit: dict) -> dict[str, np.ndarray | None]:
    """Residuals for one emotion from its fit set's `fit_axes` output. Only reliable axes
    are removed; None means untestable (V unreliable)."""
    out = {}
    for name, (emo, _, which) in RESIDUALS.items():
        if emo != emotion:
            continue
        if not fit["V"]["reliable"]:
            out[name] = None
            continue
        out[name] = residual(v, [fit[k]["axis"] for k in which if fit[k]["reliable"]])
    return out
