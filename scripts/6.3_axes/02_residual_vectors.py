"""Build residual emotion vectors (axes removed) from axes.pt; write residuals.pt.

Same format as vectors.pt["unit"] ({name: Tensor[n_layers, d]}). Layers where the fit set's
V axis is unreliable are "untestable": recorded in axes.json and left as NaN rows;
axes.json `residual_axes` {name: {layer: [axes removed]}} lists the reliable axes (raw
directions) removed at each ok layer; residuals.pt carries `status` {name: {layer: ok|untestable}} -- read rows via emolab.axes.get_residual.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emolab import axes, models  # noqa: E402
from emolab.paths import results_dir  # noqa: E402


def run(vectors_path: Path, out: Path) -> None:
    vec = torch.load(vectors_path)
    pt = torch.load(out / "axes.pt")
    if "raw_fits" not in pt:
        raise ValueError("axes.pt has no raw_fits (written by an older 01_fit_axes.py); "
                         "rerun 6.3_axes/01_fit_axes.py first")
    meta = json.loads((out / "axes.json").read_text(encoding="utf-8"))
    layers = vec["layers"]
    unit_out = {name: torch.full((len(layers), vec["unit"][emo].shape[1]), float("nan"))
                for name, (emo, _, _) in axes.RESIDUALS.items()}
    removed = {name: {} for name in axes.RESIDUALS}  # {name: {str(layer): [axes removed]}}
    status = {name: {} for name in axes.RESIDUALS}  # keyed by str(layer) for json
    pt_status = {name: {} for name in axes.RESIDUALS}  # keyed by int layer, inside residuals.pt
    for i, layer in enumerate(layers):
        for emotion in sorted({emo for emo, _, _ in axes.RESIDUALS.values()}):
            fit_set = next(fs for emo, fs, _ in axes.RESIDUALS.values() if emo == emotion)
            info = meta["fits"][fit_set][str(layer)]
            fit = {k: {"raw_axis": pt["raw_fits"][fit_set][layer][k].double().numpy(),
                       "reliable": info[k]["reliable"]} for k in "VAD"}
            res = axes.residual_vectors(emotion, vec["unit"][emotion][i].double().numpy(), fit)
            for name, r in res.items():
                if r is None:
                    status[name][str(layer)] = pt_status[name][layer] = "untestable"
                else:
                    unit_out[name][i] = torch.tensor(r, dtype=torch.float32)
                    status[name][str(layer)] = pt_status[name][layer] = "ok"
                    removed[name][str(layer)] = axes.removed_axes(fit, axes.RESIDUALS[name][2])
    torch.save({"layers": layers, "unit": unit_out, "status": pt_status}, out / "residuals.pt")
    meta["residuals"] = status
    meta["residual_axes"] = removed
    (out / "axes.json").write_text(json.dumps(meta, indent=2), encoding="utf-8", newline="\n")
    for name, st in status.items():
        print(name, st, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(models.MODELS))
    args = ap.parse_args()
    name = models.MODELS[args.model].name
    run(results_dir("6.2_emotion_vectors", name) / "vectors.pt", results_dir("6.3_axes", name))


if __name__ == "__main__":
    main()
