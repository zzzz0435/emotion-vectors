"""Download MoralChoice high-ambiguity scenarios, keep harm items, sample 150, write the dataset.

Source: https://huggingface.co/datasets/ninoscherrer/moralchoice (CC-BY-4.0).
"""
import json
import sys
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emolab.moral_items import harm_rows, sample_items  # noqa: E402
from emolab.paths import DATASETS, cache_dir  # noqa: E402

URL = ("https://huggingface.co/datasets/ninoscherrer/moralchoice/resolve/main/"
       "scenarios/moralchoice_high_ambiguity.csv")
OUT = DATASETS / "6.6_moralchoice_harm.json"


def main() -> None:
    csv = cache_dir() / "moralchoice" / "moralchoice_high_ambiguity.csv"
    if not csv.exists():
        csv.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, csv)
    df = pd.read_csv(csv)
    assert len(df) == 680, f"expected 680 rows, got {len(df)}"
    harm = harm_rows(df)
    assert len(harm) == 347, f"expected 347 harm rows, got {len(harm)}"
    sample = sample_items(harm, n=150, seed=42)
    items = [{"id": r.scenario_id, "context": r.context, "action1": r.action1,
              "action2": r.action2, "harmful": int(r.harmful)} for r in sample.itertuples()]
    OUT.write_text(json.dumps(items, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(f"rows={len(df)} harm_rows={len(harm)} sampled={len(items)} -> {OUT}")


if __name__ == "__main__":
    main()
