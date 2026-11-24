"""6.5 檢查點 2 摘要：讀 frozen_config.json、metrics.csv、texts.jsonl，寫出白話的
checkpoint2_summary.md（繁體中文），交給使用者確認後才開始正式的道德題實驗。

用法：python scripts/6.5_calibration/04_checkpoint2.py --model qwen3-8b [--rate 6.0]
（--rate：每秒前向計算次數，用來估計正式實驗的時間）
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emolab import conditions, freeze, models  # noqa: E402
from emolab.paths import RESULTS  # noqa: E402


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"找不到 {path}：請先跑完 01。")
    with path.open(encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(models.MODELS))
    ap.add_argument("--rate", type=float, default=freeze.DEFAULT_RATE,
                    help="每秒前向計算次數（預設 6.0）")
    ap.add_argument("--results-dir", type=Path, default=RESULTS, help="results 根目錄（測試用）")
    args = ap.parse_args(argv)
    if args.rate <= 0:
        ap.error("--rate must be positive")
    cal_dir = args.results_dir / "6.5_calibration" / models.MODELS[args.model].name
    frozen_path = cal_dir / freeze.FROZEN_NAME
    if not frozen_path.exists():
        raise SystemExit(f"找不到凍結檔 {frozen_path}：請先執行 03_select_and_freeze.py。")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    metrics = freeze.load_metrics(cal_dir / "metrics.csv")
    texts = _read_jsonl(cal_dir / "texts.jsonl")
    n = len(conditions.build_conditions(frozen, conditions.load_labels()))
    md = freeze.render_checkpoint2(frozen, metrics, texts, n, rate=args.rate)
    out = cal_dir / "checkpoint2_summary.md"
    out.write_text(md, encoding="utf-8", newline="\n")
    print(f"已寫入 {out}（{n} 個條件）", flush=True)


if __name__ == "__main__":
    main()
