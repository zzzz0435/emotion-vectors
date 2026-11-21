"""Calibration sweep: steered texts for 6 core emotions x layers x alpha, with early stopping.

Per (layer, alpha, emotion) the steered model writes 20 short texts; coherence (perplexity ratio,
format-validity) is measured right after, and once an emotion is incoherent at a layer, larger
alphas for that (layer, emotion) are not generated. Resumable.

Writes results/6.5_calibration/<model>/texts.jsonl and coherence.csv.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emolab import calibration as cal  # noqa: E402
from emolab import models  # noqa: E402
from emolab.conditions import load_labels  # noqa: E402
from emolab.paths import DATASETS, results_dir  # noqa: E402
from emolab.scoring import SteerSpec  # noqa: E402
from emolab.stories import repair_torn_tail  # noqa: E402

PROMPTS = DATASETS / "6.5_calibration_prompts.json"
FORMAT = DATASETS / "6.5_format_check.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(models.MODELS))
    ap.add_argument("--layers", type=int, nargs="*", default=None,
                    help="candidate layers (default: all layers in vectors.pt)")
    ap.add_argument("--batch-size", type=int, default=cal.N_PROMPTS)
    args = ap.parse_args()
    name = models.MODELS[args.model].name

    vec_path = Path(results_dir("6.2_emotion_vectors", name)) / "vectors.pt"
    for p in (vec_path, PROMPTS, FORMAT):  # check inputs before loading the model
        if not p.exists():
            raise FileNotFoundError(f"missing required input {p}"
                                    + (" (produced by 6.2_emotion_vectors/04_build_vectors.py)"
                                       if p == vec_path else ""))
    import torch
    vec = torch.load(vec_path, weights_only=False)
    layers = args.layers or list(vec["layers"])
    bad = [L for L in layers if L not in vec["layers"]]
    if bad:
        raise ValueError(f"layers {bad} not in vectors.pt layers {vec['layers']}")
    emotions = load_labels()["core"]
    prompts = json.loads(PROMPTS.read_text(encoding="utf-8"))
    fmt = json.loads(FORMAT.read_text(encoding="utf-8"))
    assert len(prompts) == cal.N_PROMPTS
    out_dir = results_dir("6.5_calibration", name)
    texts_path, coh_path = out_dir / "texts.jsonl", out_dir / "coherence.csv"

    model, tok = models.load(args.model)
    bs = args.batch_size

    def spec(layer, alpha, label):
        return SteerSpec(layer, vec["unit"][label][vec["layers"].index(layer)], alpha,
                         vec["resid_norm"][layer])

    # alpha = 0 reference: texts once (shared by all layers), median ppl, unsteered format_valid
    rows = repair_torn_tail(texts_path)
    base = cal.complete_groups(rows).get((None, 0.0, cal.BASELINE))
    if base is None:
        rows = cal.drop_partial_groups(texts_path, rows, lambda r: cal._gkey(r))
        base = cal.generate_batch(model, tok, prompts, None, cal.MAX_NEW_TOKENS, bs)
        with texts_path.open("a", encoding="utf-8", newline="\n") as f:
            for i, t in enumerate(base):
                f.write(json.dumps(dict(layer=None, alpha=0.0, label=cal.BASELINE,
                                        prompt_idx=i, text=t), ensure_ascii=False) + "\n")
        print("baseline texts generated", flush=True)
    base_median = cal.median_ppl(model, tok, prompts, base)
    base_fluency = float(np.nanmean(cal.fluency_ratings(model, tok, base, bs)))
    cal.repair_csv_tail(coh_path)
    coh = cal.read_coherence(coh_path)
    if not (coh["label"] == cal.BASELINE).any():
        fv0 = cal.format_valid(model, tok, fmt, None)
        header = "" if coh_path.exists() and coh_path.stat().st_size else ",".join(
            cal.COHERENCE_COLS) + "\n"
        with coh_path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(f"{header},0.0,{cal.BASELINE},1.0,{base_fluency!r},{fv0!r},True\n")
        print(f"baseline: median ppl {base_median:.3f}, fluency {base_fluency:.2f}, format_valid {fv0:.3f}", flush=True)

    def generate_fn(layer, alpha, label):
        return cal.generate_batch(model, tok, prompts, spec(layer, alpha, label),
                                  cal.MAX_NEW_TOKENS, bs)

    def measure_fn(layer, alpha, label, texts):
        s = spec(layer, alpha, label)
        return (cal.ppl_ratio(model, tok, prompts, texts, base_median),
                float(np.nanmean(cal.fluency_ratings(model, tok, texts, bs))),
                cal.format_valid(model, tok, fmt, s))

    cal.sweep_to_disk(out_dir, cal.sweep_plan(emotions, layers, cal.ALPHA_GRID),
                      generate_fn, measure_fn, base_fluency=base_fluency)
    print("done", flush=True)


if __name__ == "__main__":
    main()
