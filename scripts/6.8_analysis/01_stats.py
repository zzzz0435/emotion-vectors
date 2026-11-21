"""6.8 預先登記分析：讀分數 → 主要、次要、探索性檢驗 → primary.csv、secondary.csv、
exploratory.json、interpretation.md（results/6.8_analysis/<模型名>/）。

- 凍結檔 results/6.5_calibration/<模型名>/frozen_config.json 不存在就拒絕執行。
- scripts/emolab/stats.py、conditions.py，或 datasets/6.2_concepts.json、
  6.6_punishment_vignettes.json、6.6_moralchoice_harm.json 的 sha256 和凍結檔記錄的不同，
  也拒絕執行；加上 --allow-changed-analysis 才會繼續，但所有結果都標示為探索性。
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emolab import conditions, integrity, stats  # noqa: E402
from emolab.paths import DATASETS, RESULTS  # noqa: E402

DEFAULT_MODEL = "Qwen_3_8B_instruct"
BANNER = "⚠ 凍結後改過分析程式：以下所有結果僅屬探索性"
VAD_CSV = DATASETS / "6.3_nrc_vad_emotions.csv"          # git-ignored; may be absent
H2_ITEMS = DATASETS / "6.6_moralchoice_harm.json"


def hashed_files() -> dict[str, Path]:
    """Frozen sha256 key → file: the analysis rules, the condition registry, and the item /
    concept datasets the results depend on (R24)."""
    return {"stats_sha256": Path(stats.__file__),
            "conditions_sha256": Path(conditions.__file__),
            "concepts_sha256": DATASETS / "6.2_concepts.json",
            "vignettes_sha256": DATASETS / "6.6_punishment_vignettes.json",
            "moralchoice_sha256": H2_ITEMS}


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    return o


def _interpretation_md(sentences: list[str], frozen: dict, changed: bool) -> str:
    eq = frozen.get("alpha_eq", {})
    lines = [BANNER, ""] if changed else []
    lines += [
        "# 預先登記分析：判讀結果", "",
        f"- 模型：{frozen.get('model')}",
        f"- 層：{frozen.get('layer')}",
        f"- α_max：{float(frozen['alpha_max']):+.4f}",
        "- α_eq：" + "、".join(f"{e} {v:+.4f}" if v is not None else f"{e} 未能誘發"
                               for e, v in eq.items()),
        f"- 凍結時間：{frozen.get('created_at')}", "",
        "## 判讀（依設計文件 §6.8 判讀規則）", "",
    ]
    lines += [f"- {s}" for s in sentences]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", default=DEFAULT_MODEL)
    ap.add_argument("--allow-changed-analysis", action="store_true")
    ap.add_argument("--scores-dir", type=Path)
    ap.add_argument("--frozen", type=Path)
    ap.add_argument("--out-dir", type=Path)
    args = ap.parse_args(argv)

    frozen_path = args.frozen or RESULTS / "6.5_calibration" / args.model_dir / "frozen_config.json"
    if not frozen_path.exists():
        raise SystemExit(f"找不到凍結檔 {frozen_path}：校準設定還沒凍結，分析拒絕執行。")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    changed = integrity.check_frozen_hashes(frozen, hashed_files(), args.allow_changed_analysis)

    scores_dir = args.scores_dir or RESULTS / "6.7_runs" / args.model_dir / "scores"
    if not scores_dir.is_dir() or not any(scores_dir.glob("*.csv")):
        raise SystemExit(f"找不到分數檔（{scores_dir}）：請先跑完 6.7 的道德題實驗。")
    scores = stats.load_score_table(scores_dir)
    labels = conditions.load_labels()
    vad = pd.read_csv(VAD_CSV) if VAD_CSV.exists() else None
    h2_items = (pd.DataFrame(json.loads(H2_ITEMS.read_text(encoding="utf-8")))[["id", "harm_types"]]
                if H2_ITEMS.exists() else None)

    primary = stats.primary_tests(scores, frozen, labels)
    secondary = stats.secondary_tests(scores, frozen, labels)
    explo = stats.exploratory(scores, frozen, vad, h2_items, labels)
    sentences = stats.interpret(primary, frozen)
    if changed:
        primary["exploratory_changed_rules"] = True
        secondary["exploratory_changed_rules"] = True
        explo = {"exploratory_changed_rules": True, **explo}

    out = args.out_dir or RESULTS / "6.8_analysis" / args.model_dir
    out.mkdir(parents=True, exist_ok=True)
    primary.to_csv(out / "primary.csv", index=False, lineterminator="\n")
    secondary.to_csv(out / "secondary.csv", index=False, lineterminator="\n")
    (out / "exploratory.json").write_text(
        json.dumps(_jsonable(explo), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8", newline="\n")
    (out / "interpretation.md").write_text(_interpretation_md(sentences, frozen, changed),
                                           encoding="utf-8", newline="\n")
    print(f"寫入 {out}：primary.csv、secondary.csv、exploratory.json、interpretation.md", flush=True)


if __name__ == "__main__":
    main()
