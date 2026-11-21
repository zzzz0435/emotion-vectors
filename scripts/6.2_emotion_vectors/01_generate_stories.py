"""Generate the emotion / neutral / concept stories (resumable; rerun to continue)."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emolab import models, stories  # noqa: E402
from emolab.paths import DATASETS, results_dir  # noqa: E402


def _load(name: str):
    return json.loads((DATASETS / name).read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(models.MODELS))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--top-up", action="store_true",
                    help="generate extra stories for the cells listed in shortfall.json")
    args = ap.parse_args()

    spec = models.MODELS[args.model]
    rdir = results_dir("6.2_emotion_vectors", spec.name)
    out = args.out or rdir / "stories_raw.jsonl"
    emotions, topics, concepts = (_load("6.2_emotions.json"), _load("6.2_story_topics.json"),
                                  _load("6.2_concepts.json"))

    if args.top_up:
        raw = json.loads((rdir / "shortfall.json").read_text(encoding="utf-8"))
        shortfall = {(r["label"], r["topic_idx"], r["subtype_idx"]): r["missing"] for r in raw}
        existing = stories.read_existing(out)
        offset = max((r["replicate"] for r in existing), default=-1) + 1
        jobs = stories.build_topup_jobs(shortfall, emotions, topics, concepts, offset)
    else:
        jobs = stories.build_jobs(emotions, topics, concepts)
    if args.limit is not None:
        jobs = stories.round_robin_limit(jobs, args.limit)
    print(f"{len(jobs)} jobs -> {out}", flush=True)

    model, tok = models.load(args.model)
    n = stories.run_generation(jobs, out, stories.make_generate_fn(model, tok),
                               args.batch_size, spec.name)
    print(f"done: {n} new stories written", flush=True)


if __name__ == "__main__":
    main()
