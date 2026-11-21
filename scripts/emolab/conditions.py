"""Experiment-condition registry: the single source of truth for which conditions exist
and what they are called. Names are built only by `condition_name`."""
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from emolab import axes as axes_mod
from emolab.paths import DATASETS, RESULTS

NAME_RE = re.compile(r"^[A-Za-z0-9_@+.\-]+$")
RESIDUALS_FOUR = ("anger__resid3", "anger__resid2", "pain__resid3", "compassion__resid3")
CLEAN_RESIDUAL = "anger__clean__resid3"


@dataclass(frozen=True)
class Condition:
    name: str
    kind: str   # baseline emotion emotion_eq clean residual painaxis axis map text concept random
    source: str | None
    alpha: float
    induce: str | None = None


def _check(name: str) -> str:
    if not NAME_RE.match(name):
        raise ValueError(f"condition name not filename-safe: {name!r}")
    return name


def condition_name(label: str, alpha: float, eq: bool = False) -> str:
    return _check(f"{label}@{'eq' if eq else ''}{alpha:+.4f}")


def random_directions(acts: np.ndarray, n: int, seed: int = 42) -> np.ndarray:
    acts = np.asarray(acts, dtype=np.float64)
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, acts.shape[0]))
    r = z @ (acts - acts.mean(axis=0))
    norms = np.linalg.norm(r, axis=1)
    if np.any(norms == 0):
        raise ValueError("random direction with zero norm (degenerate activations)")
    return r / norms[:, None]


def load_labels() -> dict:
    emotions = json.loads((DATASETS / "6.2_emotions.json").read_text(encoding="utf-8"))
    concepts = json.loads((DATASETS / "6.2_concepts.json").read_text(encoding="utf-8"))
    phrases = json.loads((DATASETS / "6.5_emotion_phrases.json").read_text(encoding="utf-8"))
    return {
        "core": [e["label"] for e in emotions if e["group"] == "core"],
        "map": [e["label"] for e in emotions if e["group"] == "map"],
        "concepts": [c["label"] for c in concepts if c.get("role") == "control"],
        "induce": dict(phrases["induce"]),
    }


def build_conditions(frozen: dict, labels: dict) -> list[Condition]:
    a = float(frozen["alpha_max"])
    half = a / 2
    alpha_eq = frozen["alpha_eq"]
    alpha_ind = frozen.get("alpha_individual", {})
    contaminated = set(frozen.get("contamination_at_layer", []))
    resid_status = frozen.get("residual_status_at_layer", {})
    out: dict[str, Condition] = {}

    def add(label, kind, source, alpha, eq=False, induce=None):
        name = condition_name(label, alpha, eq)
        if name not in out:
            out[name] = Condition(name, kind, source, float(alpha), induce)

    def both(label, kind, source, alphas):
        for al in alphas:
            add(label, kind, source, al)

    out["baseline"] = Condition("baseline", "baseline", None, 0.0, None)
    core = labels["core"]
    for e in core:
        both(e, "emotion", f"vectors:unit:{e}", (-a, -half, half, a))
    for e in core:
        if alpha_eq.get(e) is not None:
            add(e, "emotion_eq", f"vectors:unit:{e}", alpha_eq[e], eq=True)
    for r in RESIDUALS_FOUR:
        if resid_status.get(r) != "ok":
            continue
        both(r, "residual", f"residual:{r}",
             (-a, -half, half, a) if r == "anger__resid3" else (-a, a))
    both("pain_painaxis", "painaxis", "vectors:unit:pain_painaxis", (-a, a))
    for k in "VAD":
        both(f"axis_{k}", "axis", f"axis:all:{k}", (-a, a))
    for m in labels["map"]:
        both(m, "map", f"vectors:unit:{m}", (-a, a))
    for e in core:
        s = labels["induce"][e]
        out[f"text_{_check(e)}"] = Condition(f"text_{e}", "text", None, 0.0, s)
    for c in labels["concepts"]:
        both(c, "concept", f"vectors:unit:{c}", (-a, a))
    for i in range(20):
        both(f"random{i:02d}", "random", f"random:{i}", (-a, a))
    for i in range(10):
        both(f"random{i:02d}", "random", f"random:{i}", (-half, half))
    for e in core:
        if e not in contaminated:
            continue
        src = f"clean:{e}__clean"
        both(f"{e}__clean", "clean", src, (a, -a))
        if alpha_eq.get(e) is not None:
            add(f"{e}__clean", "clean", src, alpha_eq[e], eq=True)
    if "anger" in contaminated:
        both(CLEAN_RESIDUAL, "clean", f"clean_residual:{CLEAN_RESIDUAL.replace('__clean', '')}",
             (-a, a))
    for e, ai in alpha_ind.items():
        add(e, "emotion", f"vectors:unit:{e}", ai)
        if e in core and e in contaminated:   # clean check of a fallback H1/H2 (R20)
            add(f"{e}__clean", "clean", f"clean:{e}__clean", ai)
        for c in labels["concepts"]:
            add(c, "concept", f"vectors:unit:{c}", ai)
        for i in range(20):
            add(f"random{i:02d}", "random", f"random:{i}", ai)
    return list(out.values())


def _row(payload: dict, label: str, layer: int, what: str) -> np.ndarray:
    if layer not in payload["layers"]:
        raise ValueError(f"layer {layer} not in {what} payload")
    if label not in payload["unit"]:
        raise ValueError(f"label {label!r} missing from {what} payload")
    return payload["unit"][label][payload["layers"].index(layer)].double().numpy()


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


def resolve_direction(cond: Condition, layer: int, stores: dict) -> np.ndarray | None:
    if cond.source is None:
        return None
    kind, _, rest = cond.source.partition(":")
    if kind == "vectors":
        _, _, label = rest.partition(":")
        return _unit(_row(stores["vectors"], label, layer, "vectors"))
    if kind == "clean":
        return _unit(_row(stores["clean"], rest, layer, "clean"))
    if kind == "residual":
        return _unit(axes_mod.get_residual(stores["residuals"], rest, layer))
    if kind == "axis":
        fit_set, _, k = rest.partition(":")
        try:
            return _unit(stores["axes"]["fits"][fit_set][layer][k].double().numpy())
        except (KeyError, IndexError) as exc:
            raise ValueError(f"axis {rest!r} not found at layer {layer}") from exc
    if kind == "random":
        r = stores.get("random")
        if r is None:
            raise ValueError("random directions not available (stores['random'] is None)")
        return _unit(np.asarray(r)[int(rest)])
    if kind == "clean_residual":
        v = _row(stores["clean"], "anger__clean", layer, "clean")
        meta = stores["axes_meta"]["fits"]["for_anger"][str(layer)]
        if not meta["V"]["reliable"]:
            raise ValueError("V axis unreliable: anger__clean__resid3 is untestable")
        raw = stores["axes"]["raw_fits"]["for_anger"][layer]
        removed = [raw[k].double().numpy() for k in "VAD" if meta[k]["reliable"]]
        return axes_mod.residual(v, removed)
    raise ValueError(f"unknown direction source: {cond.source!r}")


def load_stores(model_name: str, layer: int) -> dict:
    import torch
    vec_dir = RESULTS / "6.2_emotion_vectors" / model_name
    ax_dir = RESULTS / "6.3_axes" / model_name
    rnd = RESULTS / "6.7_runs" / model_name / f"random_directions_L{layer}.npy"
    load = lambda p: torch.load(p, weights_only=False)  # noqa: E731
    return {
        "vectors": load(vec_dir / "vectors.pt"),
        "clean": load(vec_dir / "vectors_clean.pt"),
        "residuals": load(ax_dir / "residuals.pt"),
        "axes": load(ax_dir / "axes.pt"),
        "axes_meta": json.loads((ax_dir / "axes.json").read_text(encoding="utf-8")),
        "random": np.load(rnd) if rnd.exists() else None,
    }
