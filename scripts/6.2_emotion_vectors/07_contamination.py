"""Contamination check: do emotion vectors already contain a punishment / harming_others component?"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emolab import models  # noqa: E402
from emolab.paths import DATASETS, results_dir  # noqa: E402
from emolab.vectors import contamination_flags, remove_components  # noqa: E402

PAIN_VEC = "pain_painaxis"
PROBES = ["punishment", "harming_others"]


def run_check(unit: dict[str, np.ndarray], layers: list[int], checked: list[str],
              concepts: list[str], probes: list[str], reference: str = "sadness"):
    """unit[label] is [n_layers, d] aligned with `layers`.

    Returns (flags, clean, meta): flags {"<L>": {probe: {label: {...}}}}; clean {"<label>__clean":
    [n_layers, d]} for labels flagged at any layer (unflagged layers keep the original vector);
    meta records the cleaned layers and the probes removed there.
    """
    flags: dict = {}
    clean: dict[str, np.ndarray] = {}
    cleaned_layers: dict[str, list[int]] = {}
    removed: dict[str, dict[str, list[str]]] = {}
    for i, L in enumerate(layers):
        at = {k: np.asarray(v[i], dtype=np.float64) for k, v in unit.items()}
        flags[str(L)] = {p: contamination_flags(at, checked, concepts, p, reference) for p in probes}
        for lab in checked:
            hit = [p for p in probes if flags[str(L)][p][lab]["flagged"]]
            if not hit:
                continue
            if lab not in clean:
                clean[lab] = np.array(unit[lab], dtype=np.float32)
            clean[lab][i] = remove_components(at[lab], [at[p] for p in hit])
            cleaned_layers.setdefault(lab, []).append(L)
            removed.setdefault(lab, {})[str(L)] = hit
    clean = {f"{lab}__clean": v for lab, v in clean.items()}
    return flags, clean, {"cleaned_layers": cleaned_layers, "removed_probes": removed}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(models.MODELS))
    args = ap.parse_args()
    name = models.MODELS[args.model].name
    rdir = results_dir("6.2_emotion_vectors", name)

    payload = torch.load(rdir / "vectors.pt")
    layers = payload["layers"]
    emotions = json.loads((DATASETS / "6.2_emotions.json").read_text(encoding="utf-8"))
    emo = [e["label"] for e in emotions if e["group"] != "neutral"]
    concepts = [c["label"] for c in json.loads((DATASETS / "6.2_concepts.json").read_text(encoding="utf-8"))
                if c["role"] == "control"]
    unit = {k: v.numpy() for k, v in payload["unit"].items()}

    flags, clean, meta = run_check(unit, layers, [*emo, PAIN_VEC], concepts, PROBES)
    (rdir / "contamination.json").write_text(json.dumps(flags, indent=2), encoding="utf-8")
    for L, by_probe in flags.items():
        for probe, by_lab in by_probe.items():
            hit = [lab for lab, r in by_lab.items() if r["flagged"]]
            print(f"layer {L} {probe}: flagged {hit or 'none'}")
    if clean:
        meta["layers"] = layers
        torch.save({"layers": layers, "unit": {k: torch.from_numpy(v) for k, v in clean.items()},
                    "meta": meta}, rdir / "vectors_clean.pt")
        print(f"wrote vectors_clean.pt ({len(clean)} vectors)")
    else:
        print("nothing flagged; no vectors_clean.pt")


if __name__ == "__main__":
    main()
