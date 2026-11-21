"""Fit the V / A / D axes per fit set and layer; write axes.pt and axes.json.

axes.pt: `fits` = orthogonalised axes (for steering), `raw_fits` = the unit axes before
orthogonalisation (residuals are built from these); both {fit_set: {layer: {V,A,D}}}.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emolab import axes, models  # noqa: E402
from emolab.paths import DATASETS, results_dir  # noqa: E402

VAD_CSV = DATASETS / "6.3_nrc_vad_emotions.csv"


def run(vectors_path: Path, vad_csv: Path, out: Path) -> None:
    vec = torch.load(vectors_path)
    layers = vec["layers"]
    vad = pd.read_csv(vad_csv)
    # pain_painaxis (and any non-VAD label) is not in the csv, so never enters a fit
    labels = [lab for lab in vad["label"] if lab in vec["unit"]]
    vad = vad[vad["label"].isin(labels)]
    pt = {"layers": layers, "fits": {}, "raw_fits": {}}
    meta = {"layers": layers, "fits": {}}
    for fit_set, exclude in axes.FIT_SETS.items():
        pt["fits"][fit_set], pt["raw_fits"][fit_set], meta["fits"][fit_set] = {}, {}, {}
        for i, layer in enumerate(layers):
            unit = {lab: vec["unit"][lab][i].double().numpy() for lab in labels}
            fit = axes.fit_axes(unit, vad, exclude)
            pt["fits"][fit_set][layer] = {k: torch.tensor(f["axis"], dtype=torch.float32)
                                          for k, f in fit.items()}
            pt["raw_fits"][fit_set][layer] = {k: torch.tensor(f["raw_axis"], dtype=torch.float32)
                                              for k, f in fit.items()}
            meta["fits"][fit_set][str(layer)] = {
                k: {"alpha": f["alpha"], "loo_r": f["loo_r"], "reliable": f["reliable"]}
                for k, f in fit.items()}
            print(f"{fit_set} L{layer}: " + "  ".join(
                f"{k} r={f['loo_r']:.2f}{'' if f['reliable'] else ' (unreliable)'}"
                for k, f in fit.items()), flush=True)
    torch.save(pt, out / "axes.pt")
    (out / "axes.json").write_text(json.dumps(meta, indent=2), encoding="utf-8", newline="\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(models.MODELS))
    args = ap.parse_args()
    name = models.MODELS[args.model].name
    run(results_dir("6.2_emotion_vectors", name) / "vectors.pt", VAD_CSV,
        results_dir("6.3_axes", name))


if __name__ == "__main__":
    main()
