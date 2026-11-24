"""6.8 圖表：讀 6.8/01 的輸出、分數檔和凍結檔，畫 PNG（dpi 150）到
results/6.8_analysis/<模型名>/figures/ 並印出路徑。

圖：劑量反應（每個任務一張）、情緒地圖（每個任務一張）、憤怒 vs 恐懼（α_eq）、
文字誘發 vs 向量、拒答程度。字型優先用 Microsoft JhengHei，其次 Microsoft YaHei；
都沒有就改用英文標籤。每種核心情緒在所有圖裡用同一種顏色。
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emolab import conditions, results_report, stats  # noqa: E402
from emolab.paths import DATASETS, RESULTS  # noqa: E402

DEFAULT_MODEL = "Qwen_3_8B_instruct"
VAD_CSV = DATASETS / "6.3_nrc_vad_emotions.csv"          # git-ignored; may be absent
FONTS = ("Microsoft JhengHei", "Microsoft YaHei")
DPI = 150
EMO_COLORS = {"anger": "#d62728", "fear": "#7b3fa0", "sadness": "#1f77b4",
              "joy": "#e6a100", "compassion": "#2ca02c", "pain": "#8c564b"}
GREY = "#8a8a8a"
TASK_EN = {"h1_rating": "H1 punishment rating", "h2": "H2 harmful choice"}


def pick_font(available=None) -> str | None:
    """First installed CJK font from FONTS, else None (English labels)."""
    have = set(available) if available is not None else \
        {f.name for f in font_manager.fontManager.ttflist}
    return next((f for f in FONTS if f in have), None)


def make_tr(font: str | None):
    """tr(zh, en): the Chinese text when a CJK font exists, otherwise the English text."""
    return (lambda zh, en: zh) if font else (lambda zh, en: en)


def _style(font):
    rc = {"axes.unicode_minus": False, "axes.spines.top": False, "axes.spines.right": False}
    if font:
        rc["font.family"] = [font, "DejaVu Sans"]
    return plt.rc_context(rc)


def _color(e: str) -> str:
    return EMO_COLORS.get(e, GREY)


def _task_name(task, tr):
    return tr(results_report.TASK_NAMES.get(task, task), TASK_EN.get(task, task))


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def _effects_of(scores, task, conds) -> np.ndarray:
    return np.array([stats._eff(scores, task, c) for c in conds], dtype=float)


def _band(ax, xs, lows, highs, color, alpha, label):
    xs, lows, highs = map(np.asarray, (xs, lows, highs))
    ok = np.isfinite(lows) & np.isfinite(highs)
    if ok.sum() >= 2:
        ax.fill_between(xs[ok], lows[ok], highs[ok], color=color, alpha=alpha, lw=0, label=label)


def _isfinite(x) -> bool:
    """True for a real finite number; None / NaN / non-numbers are missing."""
    try:
        return bool(np.isfinite(float(x)))
    except (TypeError, ValueError):
        return False


def control_ranges(scores, task, labels, a) -> dict:
    """2.5-97.5 percentile range of the control effects, ONLY at the alphas that have data:
    {"concept": {alpha: (lo, hi, n)}, "random": {alpha: (lo, hi, n)}}. No interpolation."""
    def rng(names):
        e = _effects_of(scores, task, names)
        e = e[np.isfinite(e)]
        return (float(np.percentile(e, 2.5)), float(np.percentile(e, 97.5)), int(e.size))             if e.size else None
    out = {"concept": {}, "random": {}}
    for al in (-a, a):
        r = rng([conditions.condition_name(c, al) for c in labels["concepts"]])
        if r:
            out["concept"][al] = r
    for al in (-a, -a / 2, a / 2, a):
        r = rng([conditions.condition_name(f"random{i:02d}", al) for i in range(stats.N_RANDOM)])
        if r:
            out["random"][al] = r
    return out


def _eff_label(task, tr) -> str:
    """Y-axis label with the unit: points on the 1-7 scale, or probability (0-1)."""
    if task == "h1_rating":
        return tr("效果（相對未調整，單位：分，1–7 分量表）", "effect (vs unsteered; points, 1-7 scale)")
    return tr("效果（相對未調整，單位：機率 0–1）", "effect (vs unsteered; probability, 0-1)")


# ---------- 1. dose response ----------

def fig_dose_response(scores, frozen, labels, task, out_dir, font=None) -> Path:
    tr, a = make_tr(font), float(frozen["alpha_max"])
    alphas = [-a, -a / 2, 0.0, a / 2, a]
    base = scores[task]["baseline"] if "baseline" in scores[task].columns else None
    with _style(font):
        fig, ax = plt.subplots(figsize=(10, 6.5))
        ranges = control_ranges(scores, task, labels, a)
        off = a * 0.025
        n_c = max((v[2] for v in ranges["concept"].values()), default=0)
        n_r = max((v[2] for v in ranges["random"].values()), default=0)
        for key, dx, color, zh, en in (
                ("concept", -off, "#9a9a9a",
                 f"概念向量（只在 ±α_max，最多 {n_c} 個）2.5–97.5%",
                 f"concept vectors (only at +-alpha_max, up to {n_c}) 2.5-97.5%"),
                ("random", off, "#c8c8c8",
                 f"隨機方向（只在有資料的 α，最多 {n_r} 個）2.5–97.5%",
                 f"random directions (only where measured, up to {n_r}) 2.5-97.5%")):
            for i, (al, (lo, hi, _)) in enumerate(sorted(ranges[key].items())):
                ax.vlines(al + dx, lo, hi, color=color, lw=12, alpha=0.7, zorder=1,
                          label=tr(zh, en) if i == 0 else None)
        for e in labels["core"]:
            ys, los, his = [], [], []
            for al in alphas:
                if al == 0 or base is None:
                    ys.append(0.0 if base is not None else np.nan)
                    los.append(ys[-1]); his.append(ys[-1])
                    continue
                cond = conditions.condition_name(e, al)
                if cond not in scores[task].columns:
                    ys.append(np.nan); los.append(np.nan); his.append(np.nan)
                    continue
                m, l, h = stats._mean_ci((scores[task][cond] - base).to_numpy(dtype=float))
                ys.append(m); los.append(l); his.append(h)
            ax.plot(alphas, ys, "-o", color=_color(e), label=e, lw=2, ms=5)
            _band(ax, alphas, los, his, _color(e), 0.18, None)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(alphas)
        ax.set_xticklabels([f"{x:+.2f}" for x in alphas])
        ax.set_xlabel(tr("調整強度 α（負值 = 反方向）", "steering strength alpha (negative = reversed)"))
        ax.set_ylabel(_eff_label(task, tr))
        ax.set_title(tr(f"劑量反應：{_task_name(task, tr)}", f"Dose response: {_task_name(task, tr)}"))
        ax.legend(fontsize=9, ncol=2, frameon=False)
        return _save(fig, Path(out_dir) / f"dose_response_{task}.png")


# ---------- 2. emotion map ----------

def fig_emotion_map(explo, vad, task, out_dir, labels, font=None) -> Path | None:
    tr = make_tr(font)
    t = (explo.get("emotion_map") or {}).get(task) or {}
    if vad is None or "effects" not in t:
        return None
    v = vad.drop_duplicates("label").set_index("label")
    top = {d["label"] for d in t.get("top_deviations", [])}
    effects = {e: float(y) for e, y in t["effects"].items() if e in v.index and _isfinite(y)}
    n_total = len(t["effects"])
    if not effects:
        return None
    core = set(labels["core"])
    with _style(font):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), sharey=True)
        for ax, (col, zh, en) in zip(axes, (("valence", "正負面（V）", "valence (V)"),
                                            ("arousal", "激動程度（A）", "arousal (A)"),
                                            ("dominance", "掌控感（D）", "dominance (D)"))):
            for e, y in effects.items():
                x = float(v.loc[e, col])
                hot = e in top
                ax.scatter(x, y, s=90 if hot else 40, color=_color(e) if e in core else GREY,
                           edgecolor="black" if hot else "none", lw=1.6, zorder=3)
                ax.annotate(e, (x, y), fontsize=8, xytext=(3, 3), textcoords="offset points",
                            fontweight="bold" if hot else "normal",
                            color="black" if hot else "#444444")
            ax.axhline(0, color="black", lw=0.8)
            ax.set_xlabel(tr(zh, en))
        axes[0].set_ylabel(_eff_label(task, tr) + tr(" @ +α_max", " @ +alpha_max"))
        fig.suptitle(tr(f"情緒地圖：{_task_name(task, tr)}（粗框 = 偏離三軸預測最多的三種；"
                        f"{len(effects)}/{n_total} 種情緒有資料）",
                        f"Emotion map: {_task_name(task, tr)} (outlined = top-3 deviators; "
                        f"{len(effects)}/{n_total} emotions with data)"))
        return _save(fig, Path(out_dir) / f"emotion_map_{task}.png")


# ---------- 3. anger vs fear at alpha_eq ----------

def fig_anger_vs_fear(primary, out_dir, font=None) -> Path | None:
    tr = make_tr(font)
    rows = primary[primary["hypothesis"] == "H3"]
    if rows.empty:
        return None
    r = rows.iloc[0]
    vals = [r[k] for k in ("x_effect", "y_effect", "mean_d", "x_ci_low", "x_ci_high",
                           "y_ci_low", "y_ci_high", "ci_low", "ci_high")]
    if any(results_report._isnan(x) for x in vals):
        return None
    items = [(tr("憤怒（α_eq）", "anger (alpha_eq)"), r["x_effect"], r["x_ci_low"], r["x_ci_high"],
              _color("anger")),
             (tr("恐懼（α_eq）", "fear (alpha_eq)"), r["y_effect"], r["y_ci_low"], r["y_ci_high"],
              _color("fear")),
             (tr("H3 差異（憤怒 − 恐懼）", "H3 difference (anger - fear)"), r["mean_d"],
              r["ci_low"], r["ci_high"], "black")]
    with _style(font):
        fig, ax = plt.subplots(figsize=(8, 5.5))
        for i, (name, m, lo, hi, c) in enumerate(items):
            ax.errorbar(i, m, yerr=[[m - lo], [hi - m]], fmt="o", color=c, capsize=6, ms=9, lw=2)
        ax.axhline(0, color="black", lw=0.8)
        ax.axvline(1.5, color=GREY, lw=0.8, ls=":")
        ax.set_xticks(range(3))
        ax.set_xticklabels([i[0] for i in items])
        ax.set_xlim(-0.6, 2.6)
        ax.set_ylabel(_eff_label("h1_rating", tr) + tr("，95% CI", ", 95% CI"))
        ax.set_title(tr("憤怒 vs 恐懼（誘發強度相同，H1 懲罰評分）",
                        "Anger vs fear at equal induction (H1 rating)"))
        return _save(fig, Path(out_dir) / "anger_vs_fear_eq.png")


# ---------- 4. text vs vector ----------

def fig_text_vs_vector(secondary, out_dir, labels, font=None) -> Path | None:
    tr = make_tr(font)
    tv = secondary[secondary["hypothesis"] == "text_vs_vector"]
    if tv.empty:
        return None
    emos = labels["core"]
    tasks = [t for t in ("h1_rating", "h2") if (tv["task"] == t).any()]
    with _style(font):
        fig, axes = plt.subplots(1, len(tasks), figsize=(14, 5.5), squeeze=False)
        for ax, task in zip(axes[0], tasks):
            sub = {str(r["contrast"]).split(" ")[0].removeprefix("text_"): r
                   for _, r in tv[tv["task"] == task].iterrows()}
            for i, e in enumerate(emos):
                r = sub.get(e)
                if r is None:
                    continue
                for dx, key, mk, lab in ((-0.12, "text", "s", tr("文字誘發", "text induction")),
                                         (0.12, "vector", "o", tr("向量 +α_max", "vector +alpha_max"))):
                    m, lo, hi = (r[f"{key}_effect"], r[f"{key}_ci_low"], r[f"{key}_ci_high"])
                    if results_report._isnan(m):
                        continue
                    ax.errorbar(i + dx, m, yerr=[[m - lo], [hi - m]], fmt=mk, color=_color(e),
                                mfc=_color(e) if key == "vector" else "white", capsize=4, ms=7,
                                label=lab if i == 0 else None)
            ax.axhline(0, color="black", lw=0.8)
            ax.set_xticks(range(len(emos)))
            ax.set_xticklabels(emos, rotation=30)
            ax.set_title(_task_name(task, tr))
            ax.set_ylabel(_eff_label(task, tr) + tr("，95% CI", ", 95% CI"))
            ax.legend(frameon=False, fontsize=9)
        fig.suptitle(tr("文字誘發 vs 向量調整（實心 = 向量，空心 = 文字）",
                        "Text induction vs vector steering (filled = vector, open = text)"))
        return _save(fig, Path(out_dir) / "text_vs_vector.png")


# ---------- 5. refusal ----------

def fig_refusal(explo, frozen, labels, out_dir, font=None) -> Path | None:
    tr = make_tr(font)
    ref = explo.get("refusal") or {}
    tasks = [t for t in ("h1_rating", "h2") if ref.get(t)]
    if not tasks:
        return None
    a = float(frozen["alpha_max"])
    with _style(font):
        fig, axes = plt.subplots(1, len(tasks), figsize=(13, 5), squeeze=False)
        for ax, task in zip(axes[0], tasks):
            acc = {g: [] for g in results_report.GROUPS}
            for cond, v in ref[task].items():
                g = results_report.condition_group(cond, labels, a)
                if g and not results_report._isnan(v):
                    acc[g].append(float(v))
            gs = [g for g in results_report.GROUPS if acc[g]]
            ax.bar(range(len(gs)), [np.mean(acc[g]) for g in gs], color=GREY)
            ax.set_xticks(range(len(gs)))
            ax.set_xticklabels([tr(results_report.GROUP_NAMES[g], g) for g in gs], rotation=20)
            ax.axhline(0, color="black", lw=0.8)
            ax.set_ylabel(tr("無效回答的平均機率質量（0–1）", "mean invalid probability mass (0-1)"))
            ax.set_title(_task_name(task, tr))
        fig.suptitle(tr("拒答程度（依條件分組）", "Refusal / invalid mass by condition group"))
        return _save(fig, Path(out_dir) / "refusal.png")


# ---------- driver ----------

def _safe(name, fn, *args) -> Path | None:
    """Run one figure; a missing-data skip or an unexpected error is printed, never fatal."""
    try:
        p = fn(*args)
    except Exception as exc:  # noqa: BLE001 - one bad figure must not kill the others
        print(f"略過圖 {name}：{type(exc).__name__}: {exc}", flush=True)
        return None
    if p is None:
        print(f"略過圖 {name}：缺少所需資料", flush=True)
    return p


def make_all(scores, frozen, labels, primary, secondary, explo, vad, out_dir, font=None) -> list[Path]:
    paths = []
    for task in ("h1_rating", "h2"):
        if task in scores:
            paths.append(_safe(f"dose_response_{task}", fig_dose_response, scores, frozen, labels,
                               task, out_dir, font))
    for task in ("h1_rating", "h2"):
        paths.append(_safe(f"emotion_map_{task}", fig_emotion_map, explo, vad, task, out_dir,
                           labels, font))
    paths += [_safe("anger_vs_fear_eq", fig_anger_vs_fear, primary, out_dir, font),
              _safe("text_vs_vector", fig_text_vs_vector, secondary, out_dir, labels, font),
              _safe("refusal", fig_refusal, explo, frozen, labels, out_dir, font)]
    return [p for p in paths if p is not None]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", default=DEFAULT_MODEL)
    ap.add_argument("--scores-dir", type=Path)
    ap.add_argument("--frozen", type=Path)
    ap.add_argument("--out-dir", type=Path)
    args = ap.parse_args(argv)

    out = args.out_dir or RESULTS / "6.8_analysis" / args.model_dir
    frozen_path = args.frozen or RESULTS / "6.5_calibration" / args.model_dir / "frozen_config.json"
    scores_dir = args.scores_dir or RESULTS / "6.7_runs" / args.model_dir / "scores"
    if not frozen_path.exists():
        raise SystemExit(f"找不到凍結檔 {frozen_path}：請先完成校準。")
    if not scores_dir.is_dir() or not any(scores_dir.glob("*.csv")):
        raise SystemExit(f"找不到分數檔（{scores_dir}）：請先跑完 6.7 的道德題實驗。")
    try:
        primary, secondary, explo, _ = results_report.load_analysis(out)
    except FileNotFoundError as exc:
        raise SystemExit(f"缺少 6.8/01 的輸出（{exc}）：請先執行 6.8_analysis/01_stats.py。")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    scores = stats.load_score_table(scores_dir)
    vad = pd.read_csv(VAD_CSV) if VAD_CSV.exists() else None
    font = pick_font()
    if font is None:
        print("找不到中文字型（Microsoft JhengHei / YaHei），圖上改用英文標籤。", flush=True)
    for p in make_all(scores, frozen, conditions.load_labels(), primary, secondary, explo, vad,
                      out / "figures", font):
        print(p, flush=True)


if __name__ == "__main__":
    main()
