"""Filter raw stories, keep each (label, topic, subtype) cell balanced, report shortfalls."""
import argparse
import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emolab import models, stories  # noqa: E402
from emolab.paths import DATASETS, results_dir  # noqa: E402

REASONS = ["thinking_leak", "names_emotion", "contamination", "non_english", "too_short", "too_long"]
FINAL_MODEL = "qwen3-8b"


def _load(name: str):
    return json.loads((DATASETS / name).read_text(encoding="utf-8"))


def _write_lf(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(models.MODELS))
    ap.add_argument("--stories", type=Path, default=None,
                    help="raw stories JSONL (default: results/.../<model>/stories_raw.jsonl)")
    args = ap.parse_args()

    spec = models.MODELS[args.model]
    rdir = results_dir("6.2_emotion_vectors", spec.name)
    raw_path = args.stories or rdir / "stories_raw.jsonl"
    emotions, topics, concepts = (_load("6.2_emotions.json"), _load("6.2_story_topics.json"),
                                  _load("6.2_concepts.json"))
    banned = {e["label"]: e["banned"] for e in emotions}
    raw = stories.read_rows(raw_path)
    targets = stories.targets_from_jobs(stories.build_jobs(emotions, topics, concepts))
    kept, shortfall = stories.select_balanced(raw, targets, banned)

    # report: per label, how many raw stories hit each rejection reason
    labels = list(dict.fromkeys(k[0] for k in targets))
    stat = {l: {"label": l, "n_raw": 0, **{r: 0 for r in REASONS}, "n_kept": 0, "shortfall": 0}
            for l in labels}
    for r in raw:
        s = stat[r["label"]]
        s["n_raw"] += 1
        for reason in stories.check_story(r["text"], banned.get(r["label"], []), r["kind"]):
            s[reason] += 1
    for r in kept:
        stat[r["label"]]["n_kept"] += 1
    for (label, _, _), n in shortfall.items():
        stat[label]["shortfall"] += n
    with (rdir / "filter_report.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["label", "n_raw", *REASONS, "n_kept", "shortfall"],
                           lineterminator="\n")
        w.writeheader()
        w.writerows(stat.values())

    _write_lf(rdir / "shortfall.json", json.dumps(
        [{"label": l, "topic_idx": t, "subtype_idx": s, "missing": m}
         for (l, t, s), m in sorted(shortfall.items())], indent=2) + "\n")

    # spot-check sample (Chinese headings, seed 42)
    rng = random.Random(42)
    core = [e["label"] for e in emotions if e["group"] == "core"]
    contam = [c["label"] for c in concepts if c["role"] == "contamination"]
    lines = ["# 故事抽查样本", "", "固定随机种子 42；核心情绪每种 5 篇，污染概念每个 2 篇。", ""]
    for label, n in [(l, 5) for l in core] + [(l, 2) for l in contam]:
        pool = [r for r in kept if r["label"] == label]
        lines += [f"## {label}", ""]
        for r in rng.sample(pool, min(n, len(pool))):
            lines += [f"### {r['id']}（主题：{r['topic']}）", "", r["text"], ""]
    _write_lf(rdir / "review_samples.md", "\n".join(lines))

    if args.model == FINAL_MODEL:
        heldout = {t["idx"]: t["heldout"] for t in topics}
        out = DATASETS / "6.2_stories.jsonl"
        _write_lf(out, "".join(
            json.dumps({**r, "heldout": heldout[r["topic_idx"]]}, ensure_ascii=False) + "\n"
            for r in kept))
        print(f"wrote {out}")
    print(f"raw {len(raw)}, kept {len(kept)}, shortfall cells {len(shortfall)} "
          f"({sum(shortfall.values())} stories)")


if __name__ == "__main__":
    main()
