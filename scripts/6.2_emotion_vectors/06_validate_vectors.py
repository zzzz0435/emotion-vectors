"""Validate emotion vectors on held-out stories; check anger-vs-fear separability."""
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emolab import models  # noqa: E402
from emolab.paths import DATASETS, cache_dir, results_dir  # noqa: E402
from emolab.vectors import SEPARABLE_THRESHOLD, as_bool, projection_auroc, separability  # noqa: E402

PAIN_VEC = "pain_painaxis"  # validated against the stories' `pain` concept, same as `pain`


def validation_table(acts: dict[int, np.ndarray], index: pd.DataFrame,
                     unit: dict[str, np.ndarray], layers: list[int], neutral: str) -> pd.DataFrame:
    """auroc_vs_neutral per (label, layer): heldout rows only, projection on the unit vector.

    unit[label] is [n_layers, d] aligned with `layers`.
    """
    row_labels = index["label"].to_numpy()
    test = as_bool(index["heldout"]).to_numpy()
    rows = []
    for lab, vecs in unit.items():
        data_lab = "pain" if lab == PAIN_VEC else lab
        for i, L in enumerate(layers):
            v = np.asarray(vecs[i], dtype=np.float64)
            auc = projection_auroc(acts[L], row_labels, v / np.linalg.norm(v), data_lab, neutral, test)
            rows.append({"label": lab, "layer": L, "auroc_vs_neutral": auc})
    return pd.DataFrame(rows, columns=["label", "layer", "auroc_vs_neutral"])


def similarity_matrix(unit_at_layer: dict[str, np.ndarray]) -> pd.DataFrame:
    """Pairwise cosine similarity between vectors."""
    labs = list(unit_at_layer)
    m = np.stack([np.asarray(unit_at_layer[k], dtype=np.float64) for k in labs])
    m = m / np.linalg.norm(m, axis=1, keepdims=True)
    return pd.DataFrame(m @ m.T, index=labs, columns=labs)


def plot_similarity(sim: pd.DataFrame, layer: int, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(sim.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(sim)), sim.columns, rotation=90)
    ax.set_yticks(range(len(sim)), sim.index)
    ax.set_title(f"Cosine similarity of emotion vectors, layer {layer}")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(models.MODELS))
    args = ap.parse_args()
    name = models.MODELS[args.model].name
    cdir = cache_dir() / "6.2" / name
    rdir = results_dir("6.2_emotion_vectors", name)

    payload = torch.load(rdir / "vectors.pt")
    layers = payload["layers"]
    emotions = json.loads((DATASETS / "6.2_emotions.json").read_text(encoding="utf-8"))
    emo = [e["label"] for e in emotions if e["group"] != "neutral"]
    neutral = next(e["label"] for e in emotions if e["group"] == "neutral")
    labels = [*emo, PAIN_VEC]
    unit = {lab: payload["unit"][lab].numpy() for lab in labels}

    index = pd.read_csv(cdir / "index.csv")
    acts = {L: np.load(cdir / f"acts_L{L}.npy") for L in layers}
    row_labels = index["label"].to_numpy()
    test = as_bool(index["heldout"]).to_numpy()

    df = validation_table(acts, index, unit, layers, neutral)
    df.to_csv(rdir / "validation.csv", index=False)

    sep = {}
    for L in layers:
        auc = separability(acts[L], row_labels, ~test, test)
        sep[str(L)] = {"auroc": auc, "separable": bool(auc >= SEPARABLE_THRESHOLD)}
    (rdir / "separability.json").write_text(json.dumps(sep, indent=2), encoding="utf-8")

    for i, L in enumerate(layers):
        sim = similarity_matrix({lab: unit[lab][i] for lab in labels})
        sim.to_csv(rdir / f"similarity_L{L}.csv")
        plot_similarity(sim, L, rdir / f"similarity_L{L}.png")

    for lab in emo:
        best = df[df.label == lab].auroc_vs_neutral.max()
        if best < 0.7:
            print(f"WARNING: {lab} auroc_vs_neutral < 0.7 at every layer (max {best:.3f})")
    print("anger vs fear AUROC per layer:", {int(k): round(v["auroc"], 3) for k, v in sep.items()})
    print(f"wrote validation.csv, separability.json, similarity_L*.csv/png to {rdir}")


if __name__ == "__main__":
    main()
