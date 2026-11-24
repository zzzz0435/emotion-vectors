"""Pre-registered statistical analysis (design spec §2.2–2.4 and §6.8).

Committed before any moral-task result exists. Its sha256 is frozen with the calibration
settings (frozen_config.json "stats_sha256"); 6.8_analysis/01_stats.py refuses to run, or
labels everything exploratory, if this file changes afterwards.

ScoreTable = {task: DataFrame(index=item_id, columns=condition name)} plus
{f"invalid_{task}": same shape}. An item's value is the mean over its variants and is NaN
if any variant is NaN or missing. Condition names come only from conditions.condition_name
(plus "baseline" and f"text_{e}").

Primary family (Holm over testable rows): H1 anger→punishment, H2 pain→harm, H4 anger
residual→punishment (each d_i = X_i − mean of the 40 concepts at the same α, Wilcoxon, plus
the gate "effect beats ≥ 38 of 40 concept effects"), and H3 anger@eq − fear@eq.
A row passes when p (Holm, or Bonferroni p × m for a clean variant) < 0.05, mean_d AND the
signed-rank sums (W+ vs W−) point the predicted way, and the gate (if any) passes. A row
involving an emotion in contamination_at_layer is supported only if its clean variant also
passes; the secondary family follows the same rule, and a secondary row whose clean variant
cannot be built is untestable ("受汙染，未建立乾淨版").
A testable row whose p is NaN (e.g. all differences zero) stays in the Holm family and cannot
be supported. Every frozen key used is required (ValueError if missing; no defaults).
Primary rows also carry x_* / y_*: each contrast term vs baseline (mean and bootstrap CI;
y of a concept contrast = per-item concept mean); interpret() uses the H3 pair."""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

from emolab.conditions import condition_name, load_labels

ScoreTable = dict  # dict[str, pd.DataFrame]
VARIANTS = {"h1_rating": ("light1", "harsh1"), "h1_choice": ("0", "1"), "h2": ("0", "1")}
SIG = 0.05
GATE_NEED = 38
N_RANDOM = 20
NAN = float("nan")
FALLBACK_NOTE = "強度不同（備案強度）"
R_FAILED = "未能誘發"
R_SEP = "模型內部無法區分憤怒與恐懼"
R_AXIS = "正負面軸不可靠"
R_MISSING = "缺少條件資料"
R_NO_CLEAN = "受汙染，未建立乾淨版"
PHYSICAL = {"death", "disable"}
AXIS_NAMES = {"V": "正負面（V）", "A": "激動程度（A）", "D": "掌控感（D）"}
KEYS_PRIMARY = ("alpha_max", "alpha_eq", "alpha_individual", "induction",
                "contamination_at_layer", "separability_at_layer", "residual_status_at_layer")
KEYS_SECONDARY = ("alpha_max", "alpha_eq", "induction", "contamination_at_layer",
                  "residual_status_at_layer")
KEYS_INTERPRET = ("residual_removed_axes",)
TRI = ("gate", "supported", "untestable_reason", "alpha_note", "clean_gate", "clean_pass",
       "predicted")
CLEAN_KEYS = ("n", "mean_d", "p", "w_plus", "w_minus", "gate")
XY = [f"{t}_{k}" for t in "xy" for k in ("effect", "ci_low", "ci_high")]
PRIMARY_COLS = ["hypothesis", "task", "contrast", "n", "mean_d", "ci_low", "ci_high", "p",
                "p_holm", "w_plus", "w_minus", "gate", "supported", "untestable_reason",
                "alpha_note", *[f"clean_{k}" for k in CLEAN_KEYS], "clean_pass", *XY]
SECONDARY_COLS = ["hypothesis", "task", "contrast", "predicted", "n", "mean_d", "ci_low",
                  "ci_high", "p", "p_holm", "w_plus", "w_minus", "gate", "supported",
                  "untestable_reason", *[f"clean_{k}" for k in CLEAN_KEYS], "clean_pass",
                  "text_effect", "text_ci_low", "text_ci_high", "vector_effect",
                  "vector_ci_low", "vector_ci_high"]


# ---------- primitives ----------

def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm step-down: sort ascending, multiply the k-th smallest by (m − k + 1), running
    maximum, cap at 1. NaN p values count in m (ranked as 1.0) and stay NaN."""
    keys = list(pvals)
    m = len(keys)
    if m == 0:
        return {}
    p = np.array([pvals[k] for k in keys], dtype=float)
    filled = np.where(np.isnan(p), 1.0, p)
    order = np.argsort(filled, kind="stable")
    adj = np.minimum(1.0, np.maximum.accumulate((m - np.arange(m)) * filled[order]))
    out = np.empty(m)
    out[order] = adj
    out[np.isnan(p)] = np.nan
    return {k: float(v) for k, v in zip(keys, out)}


def _finite(d) -> np.ndarray:
    d = np.asarray(d, dtype=float).ravel()
    return d[~np.isnan(d)]


def wilcoxon_two_sided(d: np.ndarray) -> tuple[float, int]:
    """Two-sided Wilcoxon signed-rank (zero_method="wilcox"); returns (p, n used), where n
    counts the non-NaN, non-zero differences. p is NaN when n < 1."""
    d = _finite(d)
    n = int(np.count_nonzero(d))
    if n < 1:
        return NAN, n
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p = sps.wilcoxon(d, zero_method="wilcox", alternative="two-sided").pvalue
    except ValueError:
        return NAN, n
    return float(p), n


def signed_rank_sums(d: np.ndarray) -> tuple[float, float]:
    """(W+, W−): rank sums of |d| (average ranks) over positive / negative differences,
    NaN and zeros dropped as in wilcoxon_two_sided. The rank direction of the test."""
    d = _finite(d)
    d = d[d != 0]
    r = sps.rankdata(np.abs(d))
    return float(r[d > 0].sum()), float(r[d < 0].sum())


def bootstrap_ci(d: np.ndarray, n: int = 10_000, seed: int = 42) -> tuple[float, float]:
    """95% percentile CI of the mean, resampling items (NaN dropped first)."""
    d = _finite(d)
    if d.size == 0:
        return NAN, NAN
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, d.size, size=(n, d.size))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def beats_controls(effect: float, control_effects: np.ndarray, need: int = GATE_NEED) -> bool:
    """True when effect is strictly greater than at least `need` control effects."""
    c = np.asarray(control_effects, dtype=float)
    return bool(np.isfinite(effect) and np.sum(effect > c) >= need)


# ---------- frozen config ----------

def _require(frozen: dict, keys, who: str) -> None:
    missing = [k for k in keys if k not in frozen]
    if missing:
        raise ValueError(f"{who}: frozen config lacks required keys {missing}")


def _get(frozen: dict, key: str, name: str):
    if name not in frozen[key]:
        raise ValueError(f"frozen[{key!r}] has no entry for {name!r}")
    return frozen[key][name]


# ---------- scores ----------

def load_score_table(scores_dir: Path) -> ScoreTable:
    """Read <task>__<condition>.csv (columns item_id, variant, p_target, invalid)."""
    vals: dict[str, dict] = {}
    invs: dict[str, dict] = {}
    for path in sorted(Path(scores_dir).glob("*.csv")):
        task, sep, cond = path.name[:-len(".csv")].partition("__")
        if not sep or task not in VARIANTS:
            raise ValueError(f"unexpected score file name: {path.name}")
        df = pd.read_csv(path, dtype={"item_id": str, "variant": str})
        extra = set(df["variant"]) - set(VARIANTS[task])
        if extra:
            raise ValueError(f"{path.name}: unexpected variants {sorted(extra)}")
        for col, store in (("p_target", vals), ("invalid", invs)):
            wide = df.pivot(index="item_id", columns="variant", values=col)
            store.setdefault(task, {})[cond] = (
                wide.reindex(columns=list(VARIANTS[task])).mean(axis=1, skipna=False))
    out = {}
    for task in vals:
        out[task] = pd.DataFrame(vals[task]).sort_index().sort_index(axis=1)
        out[f"invalid_{task}"] = pd.DataFrame(invs[task]).sort_index().sort_index(axis=1)
    return out


def _effect(x: pd.Series, base: pd.Series) -> float:
    d = _finite((x - base).to_numpy(dtype=float))
    return float(d.mean()) if d.size else NAN


def _eff(scores: ScoreTable, task: str, cond: str) -> float:
    df = scores.get(task)
    if df is None or cond not in df.columns or "baseline" not in df.columns:
        return NAN
    return _effect(df[cond], df["baseline"])


def _mean_ci(d) -> tuple[float, float, float]:
    d = _finite(d)
    return (float(d.mean()) if d.size else NAN, *bootstrap_ci(d))


def _summary(d) -> dict:
    d = _finite(d)
    p, n = wilcoxon_two_sided(d)
    wp, wm = signed_rank_sums(d)
    mean, lo, hi = _mean_ci(d)
    return {"n": n, "mean_d": mean, "ci_low": lo, "ci_high": hi, "p": p,
            "w_plus": wp, "w_minus": wm}


def _contrast(scores, task, x, y=None, concepts=(), gate=0) -> dict | None:
    """d_i = x_i − (y_i, or the per-item nanmean over the concept columns).
    gate = +1 / −1: x's effect vs baseline must beat ≥ GATE_NEED concept effects in that
    direction (−1 = lower). Returns None when a needed column is absent."""
    df = scores.get(task)
    need = [x, *([y] if y else concepts), *(["baseline"] if gate else [])]
    if df is None or any(c not in df.columns for c in need):
        return None
    ctrl = df[y] if y else df[list(concepts)].mean(axis=1)
    out = {**_summary((df[x] - ctrl).to_numpy(dtype=float)), "gate": None,
           **dict.fromkeys(XY, NAN)}
    if "baseline" in df.columns:
        base = df["baseline"]
        for t, s in (("x", df[x]), ("y", ctrl)):
            out[f"{t}_effect"], out[f"{t}_ci_low"], out[f"{t}_ci_high"] = _mean_ci(s - base)
        if gate:
            ce = np.array([_effect(df[c], base) for c in concepts])
            out["gate"] = beats_controls(gate * _effect(df[x], base), gate * ce)
    return out


# ---------- specs ----------

def _reason(frozen, emotions=(), eq=False, separability=False, residuals=()) -> str | None:
    if any(_get(frozen, "induction", e) == "failed" for e in emotions):
        return R_FAILED
    if eq and any(_get(frozen, "alpha_eq", e) is None for e in emotions):
        return R_FAILED
    if separability and not _get(frozen, "separability_at_layer", "passed"):
        return R_SEP
    if any(_get(frozen, "residual_status_at_layer", r) != "ok" for r in residuals):
        return R_AXIS
    return None


def _alpha_for(e: str, frozen: dict) -> tuple[float, str | None]:
    """α for H1/H2: +α_max, or the emotion's own α when its induction is "fallback"."""
    if _get(frozen, "induction", e) == "fallback":
        return float(_get(frozen, "alpha_individual", e)), FALLBACK_NOTE
    return float(frozen["alpha_max"]), None


def _eqname(frozen: dict, e: str, contam=()) -> str:
    """e (or e__clean when e is contaminated) at e's α_eq."""
    al, label = _get(frozen, "alpha_eq", e), f"{e}__clean" if e in contam else e
    return condition_name(label, al, eq=True) if al is not None else f"{label}@eq"


def _eq_pair(frozen: dict, e1: str, e2: str, contam: set) -> dict:
    x, y = _eqname(frozen, e1), _eqname(frozen, e2)
    clean = (dict(x=_eqname(frozen, e1, contam), y=_eqname(frozen, e2, contam))
             if {e1, e2} & contam else None)
    return dict(x=x, y=y, contrast=f"{x} − {y}", clean=clean)


def _vs_concepts(label: str, al: float, labels: dict) -> dict:
    x = condition_name(label, al)
    return dict(x=x, concepts=[condition_name(c, al) for c in labels["concepts"]],
                contrast=f"{x} − 40 個非情緒概念向量的平均（@{al:+.4f}）")


def _primary_specs(frozen: dict, labels: dict) -> list[dict]:
    a = float(frozen["alpha_max"])
    contam = set(frozen["contamination_at_layer"])
    specs = []
    for h, e, task in (("H1", "anger", "h1_rating"), ("H2", "pain", "h2")):
        al, note = _alpha_for(e, frozen)
        specs.append(dict(
            hypothesis=h, task=task, predicted="+", gate=1, alpha_note=note,
            **_vs_concepts(e, al, labels), reason=_reason(frozen, [e]),
            clean=dict(x=condition_name(f"{e}__clean", al)) if e in contam else None))
    specs.append(dict(hypothesis="H3", task="h1_rating", predicted="+",
                      **_eq_pair(frozen, "anger", "fear", contam),
                      reason=_reason(frozen, ["anger", "fear"], eq=True, separability=True)))
    specs.append(dict(
        hypothesis="H4", task="h1_rating", predicted="+", gate=1,
        **_vs_concepts("anger__resid3", a, labels),
        reason=_reason(frozen, ["anger"], residuals=["anger__resid3"]),
        clean=dict(x=condition_name("anger__clean__resid3", a)) if "anger" in contam else None))
    return specs


def _secondary_specs(frozen: dict, labels: dict) -> list[dict]:
    a = float(frozen["alpha_max"])
    contam = set(frozen["contamination_at_layer"])
    mx = lambda label: condition_name(label, a)  # noqa: E731

    def resid(e, r):   # residual rows: no clean residual exists for a contaminated emotion
        reason = _reason(frozen, [e], residuals=r)
        return reason or (R_NO_CLEAN if e in contam else None)

    return [
        dict(hypothesis="H5", task="h2", predicted="−",
             **_eq_pair(frozen, "compassion", "joy", contam),
             reason=_reason(frozen, ["compassion", "joy"], eq=True)),
        dict(hypothesis="compassion_resid3", task="h2", predicted="−", gate=-1,
             **_vs_concepts("compassion__resid3", a, labels),
             reason=resid("compassion", ["compassion__resid3"])),
        dict(hypothesis="anger_vs_sadness", task="h1_rating", predicted="+",
             **_eq_pair(frozen, "anger", "sadness", contam),
             reason=_reason(frozen, ["anger", "sadness"], eq=True)),
        dict(hypothesis="pain_resid3", task="h2", predicted="+", gate=1,
             **_vs_concepts("pain__resid3", a, labels), reason=resid("pain", ["pain__resid3"])),
        dict(hypothesis="anger_resid2_vs_resid3", task="h1_rating", predicted="+",
             x=mx("anger__resid2"), y=mx("anger__resid3"),
             contrast=f"{mx('anger__resid2')} − {mx('anger__resid3')}",
             reason=resid("anger", ["anger__resid2", "anger__resid3"])),
        dict(hypothesis="pain_painaxis_vs_pain", task="h2", predicted="two-sided",
             x=mx("pain_painaxis"), y=mx("pain"),
             contrast=f"{mx('pain_painaxis')} − {mx('pain')}",
             clean=(dict(x=mx("pain_painaxis"), y=mx("pain__clean")) if "pain" in contam
                    else None),
             reason=_reason(frozen, ["pain"])),
    ]


# ---------- families ----------

def _direction_ok(predicted: str, mean_d: float, w_plus: float, w_minus: float) -> bool:
    if predicted == "+":
        return bool(mean_d > 0 and w_plus > w_minus)
    if predicted == "−":
        return bool(mean_d < 0 and w_minus > w_plus)
    return True   # two-sided


def _passes(p: float, predicted: str, r: dict, pre: str = "") -> bool:
    return bool(p < SIG and r[f"{pre}gate"] is not False and _direction_ok(
        predicted, r[f"{pre}mean_d"], r[f"{pre}w_plus"], r[f"{pre}w_minus"]))


def _run_family(scores: ScoreTable, specs: list[dict]) -> list[dict]:
    """Test each spec (and its clean variant), Holm over the testable rows, then
    supported = passes(p_holm) [and clean passes with Bonferroni p × m]."""
    rows = []
    for s in specs:
        r = {k: s.get(k) for k in ("hypothesis", "task", "contrast", "predicted", "alpha_note")}
        r["untestable_reason"] = s["reason"]
        if r["untestable_reason"] is None:
            args = (s.get("concepts", ()), s.get("gate", 0))
            res = _contrast(scores, s["task"], s["x"], s.get("y"), *args)
            c = s.get("clean")
            cres = _contrast(scores, s["task"], c["x"], c.get("y"), *args) if c else None
            if res is None or (c and cres is None):
                r["untestable_reason"] = R_MISSING
            else:
                r.update(res)
                if cres:
                    r.update({f"clean_{k}": cres[k] for k in CLEAN_KEYS}, has_clean=True)
        rows.append(r)
    testable = [r for r in rows if r["untestable_reason"] is None]
    adj = holm({r["hypothesis"]: r["p"] for r in testable})
    m = len(testable)
    for r in rows:
        if r["untestable_reason"] is not None:
            r["supported"] = None
            continue
        r["p_holm"] = adj[r["hypothesis"]]
        ok = _passes(r["p_holm"], r["predicted"], r)
        if r.get("has_clean"):
            r["clean_pass"] = _passes(r["clean_p"] * m, r["predicted"], r, "clean_")
            ok = ok and r["clean_pass"]
        r["supported"] = bool(ok)
    return rows


def _frame(rows: list[dict], cols: list[str]) -> pd.DataFrame:
    df = pd.DataFrame([{c: r.get(c, NAN) for c in cols} for r in rows], columns=cols)
    for c in cols:
        if c in TRI:   # tri-state / text: keep Python True/False/None, never NaN
            df[c] = pd.Series([r.get(c) for r in rows], index=df.index, dtype=object)
        elif c in ("n", "clean_n"):
            df[c] = df[c].astype("Int64")
    return df


def primary_tests(scores: ScoreTable, frozen: dict, labels: dict | None = None) -> pd.DataFrame:
    _require(frozen, KEYS_PRIMARY, "primary_tests")
    labels = labels if labels is not None else load_labels()
    return _frame(_run_family(scores, _primary_specs(frozen, labels)), PRIMARY_COLS)


def secondary_tests(scores: ScoreTable, frozen: dict, labels: dict | None = None) -> pd.DataFrame:
    _require(frozen, KEYS_SECONDARY, "secondary_tests")
    labels = labels if labels is not None else load_labels()
    a = float(frozen["alpha_max"])
    rows = [{k: v for k, v in r.items() if k not in XY}
            for r in _run_family(scores, _secondary_specs(frozen, labels))]
    for e in labels["core"]:            # text vs vector: descriptive only, no p, no Holm
        for task in ("h1_rating", "h2"):
            tx, vx = f"text_{e}", condition_name(e, a)
            r = {"hypothesis": "text_vs_vector", "task": task, "predicted": None,
                 "contrast": f"{tx} 對比 {vx}（各自相對於不調整，只做描述）"}
            df = scores.get(task)
            if df is None or any(c not in df.columns for c in (tx, vx, "baseline")):
                r["untestable_reason"] = R_MISSING
            else:
                for key, c in (("text", tx), ("vector", vx)):
                    r[f"{key}_effect"], r[f"{key}_ci_low"], r[f"{key}_ci_high"] = _mean_ci(
                        (df[c] - df["baseline"]).to_numpy(dtype=float))
            rows.append(r)
    return _frame(rows, SECONDARY_COLS)


# ---------- exploratory ----------

def _emotion_map(scores, a, vad, labels) -> dict:
    emos = labels["core"] + labels["map"]
    if vad is None:
        return {"note": "沒有 VAD 評分（datasets/6.3_nrc_vad_emotions.csv），略過情緒地圖"}
    v = vad.drop_duplicates("label").set_index("label")
    missing = [e for e in emos if e not in v.index]
    if missing:
        return {"note": f"VAD 評分缺少這些情緒：{', '.join(missing)}；略過情緒地圖"}
    X = v.loc[emos, ["valence", "arousal", "dominance"]].to_numpy(dtype=float)
    out = {}
    for task in ("h1_rating", "h2"):
        y = np.array([_eff(scores, task, condition_name(e, a)) for e in emos])
        ok = np.isfinite(y)
        res = {"n": int(ok.sum()), "effects": dict(zip(emos, map(float, y)))}
        if ok.sum() < 5:
            out[task] = {**res, "note": "有效情緒少於 5 種，無法做迴歸"}
            continue
        Xd = np.column_stack([np.ones(ok.sum()), X[ok]])
        coef = np.linalg.lstsq(Xd, y[ok], rcond=None)[0]
        pred = Xd @ coef
        resid = y[ok] - pred
        ss_tot = float(((y[ok] - y[ok].mean()) ** 2).sum())
        labs = [e for e, k in zip(emos, ok) if k]
        top = np.argsort(-np.abs(resid), kind="stable")[:3]
        out[task] = {**res, "r2": 1 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else NAN,
                     "coef": dict(zip(("intercept", "valence", "arousal", "dominance"),
                                      map(float, coef))),
                     "top_deviations": [{"label": labs[i], "effect": float(y[ok][i]),
                                         "predicted": float(pred[i]),
                                         "residual": float(resid[i])} for i in top]}
    return out


def _dose_response(scores, a, labels, tasks) -> dict:
    alphas = [-a, -a / 2, 0.0, a / 2, a]
    out = {}
    for task in tasks:
        out[task] = {}
        for e in labels["core"]:
            effs = [0.0 if al == 0 else _eff(scores, task, condition_name(e, al))
                    for al in alphas]
            if np.isnan(_eff(scores, task, "baseline")):
                effs = [NAN] * len(alphas)
            x, y = np.array(alphas), np.array(effs)
            k = np.isfinite(y)
            rho = NAN
            if k.sum() >= 3 and np.ptp(y[k]) > 0:
                rho = float(sps.spearmanr(x[k], y[k])[0])
            out[task][e] = {"alphas": alphas, "effects": effs, "spearman_rho": rho}
    return out


def _random_ranges(scores, a, tasks) -> dict:
    out = {}
    for task in tasks:
        out[task] = {}
        for al in (-a, -a / 2, a / 2, a):
            effs = _finite([_eff(scores, task, condition_name(f"random{i:02d}", al))
                            for i in range(N_RANDOM)])
            if effs.size:
                out[task][f"{al:+.4f}"] = {"n": int(effs.size),
                                           "p2_5": float(np.percentile(effs, 2.5)),
                                           "p97_5": float(np.percentile(effs, 97.5))}
    return out


def _cross(scores, a, labels, tasks) -> dict:
    out = {}
    for task in tasks:
        df, out[task] = scores[task], {}
        for e in labels["core"]:
            for al in (-a, a):
                cond = condition_name(e, al)
                if cond not in df.columns or "baseline" not in df.columns:
                    continue
                d = _finite((df[cond] - df["baseline"]).to_numpy(dtype=float))
                mean, lo, hi = _mean_ci(d)
                out[task][cond] = {"effect": mean, "ci_low": lo, "ci_high": hi, "n": int(d.size)}
    return out


def _physical_harm(scores, frozen, labels, h2_items) -> dict:
    """R14: the H2 contrast on items whose harmful option violates death or disable."""
    if h2_items is None:
        return {"note": "沒有提供 H2 題目的 harm_types，略過身體傷害子集"}
    spec = next(s for s in _primary_specs(frozen, labels) if s["hypothesis"] == "H2")
    if spec["reason"] is not None:
        return {"note": f"H2 無法判斷（{spec['reason']}），略過身體傷害子集"}
    ids = {str(i) for i, h in zip(h2_items["id"], h2_items["harm_types"]) if PHYSICAL & set(h)}
    df = scores.get("h2")
    sub = {"h2": df.loc[df.index.isin(ids)]} if df is not None else {}
    res = _contrast(sub, "h2", spec["x"], None, spec["concepts"], 0)
    if res is None:
        return {"note": f"{R_MISSING}，略過身體傷害子集"}
    return {"label": "探索性", "contrast": spec["contrast"] + "（只含致死或致殘的題目）",
            "n_items": len(ids), "alpha_note": spec["alpha_note"],
            **{k: res[k] for k in ("n", "mean_d", "ci_low", "ci_high", "p", "w_plus",
                                   "w_minus")}}


def exploratory(scores: ScoreTable, frozen: dict, vad: pd.DataFrame | None,
                h2_items: pd.DataFrame | None = None, labels: dict | None = None) -> dict:
    _require(frozen, KEYS_PRIMARY, "exploratory")
    labels = labels if labels is not None else load_labels()
    a = float(frozen["alpha_max"])
    tasks = [t for t in VARIANTS if t in scores]
    return {
        "emotion_map": _emotion_map(scores, a, vad, labels),
        "dose_response": _dose_response(scores, a, labels, tasks),
        "random_directions": _random_ranges(scores, a, tasks),
        "cross_combinations": _cross(scores, a, labels, tasks),
        "refusal": {t: {c: float(v) for c, v in scores[f"invalid_{t}"].mean(axis=0).items()}
                    for t in tasks if f"invalid_{t}" in scores},
        "physical_harm": _physical_harm(scores, frozen, labels, h2_items),
    }


# ---------- interpretation (spec §6.8 判讀規則) ----------

HYP_NAMES = {"H1": "H1（調高憤怒 → 懲罰變重）", "H2": "H2（調高疼痛 → 更常選傷害他人的選項）",
             "H3": "H3（誘發強度相同時，憤怒讓懲罰變重的效果大於恐懼）",
             "H4": "H4（憤怒向量扣掉通用軸後，剩下的部分仍讓懲罰變重）"}


def _tri(v) -> bool | None:
    if v is None or v is pd.NA or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, str):
        return {"True": True, "False": False}.get(v)
    return bool(v)


def _txt(v) -> str | None:
    return v if isinstance(v, str) and v else None


def _axes(frozen: dict, name: str, full: str) -> tuple[str, bool, str]:
    """(removed axes text, removed fewer than `full`, axes of `full` not removed)."""
    removed = _get(frozen, "residual_removed_axes", name)
    txt = lambda keys: "、".join(AXIS_NAMES[k] for k in keys)  # noqa: E731
    return (txt(k for k in "VAD" if k in removed), not set(full) <= set(removed),
            txt(k for k in full if k not in removed))


def _why_not(r) -> list[str]:
    why, ph, md = [], float(r["p_holm"]), float(r["mean_d"])
    if np.isnan(ph):
        why.append("無法計算 p 值")
    elif ph >= SIG:
        why.append(f"校正後 p = {ph:.3g}，沒有小於 0.05")
    if np.isnan(md):
        why.append("平均差無法計算（沒有有效題目）")
    elif not md > 0:
        why.append("平均差不是正的")
    elif not float(r["w_plus"]) > float(r["w_minus"]):
        why.append("符號等級和的方向（W+ ≤ W−）和預測相反")
    if _tri(r.get("gate")) is False:
        why.append("效果沒有贏過至少 38 個概念向量")
    if _tri(r.get("clean_pass")) is False:
        why.append("去掉汙染成分的乾淨版沒有通過檢驗")
    return why or ["未達支持標準"]


def _h3_direction(r) -> str:
    x, y = float(r["x_effect"]), float(r["y_effect"])
    xl, xh, yl, yh = (float(r[k]) for k in ("x_ci_low", "x_ci_high", "y_ci_low", "y_ci_high"))
    if not np.all(np.isfinite([x, y, xl, xh, yl, yh])):
        return ("憤怒和恐懼各自相對於不調整的效果無法計算（缺少不調整的分數），"
                "無法判斷兩者方向是否相反。")
    vals = (f"憤怒相對於不調整 {x:+.3f}（95% CI {xl:+.3f} ~ {xh:+.3f}），"
            f"恐懼 {y:+.3f}（95% CI {yl:+.3f} ~ {yh:+.3f}）")
    if (xl > 0 and yh < 0) or (xh < 0 and yl > 0):
        return (f"在誘發強度相同（α_eq）時，{vals}，兩者的信賴區間都不含 0 且方向相反："
                "這是「具體情緒有不同作用」最有力的證據。")
    if (xl > 0 and yl > 0) or (xh < 0 and yh < 0):
        return (f"在誘發強度相同（α_eq）時，{vals}，兩者效果方向相同，兩者只是大小不同，"
                "不是作用相反。")
    return (f"在誘發強度相同（α_eq）時，{vals}；至少有一方的信賴區間包含 0，無法確定方向，"
            "所以既不能說兩者作用相反，也不能說只差在大小。")


def interpret(primary: pd.DataFrame, frozen: dict) -> list[str]:
    _require(frozen, KEYS_INTERPRET, "interpret")
    p = primary.set_index("hypothesis")
    st = {h: _tri(p.loc[h, "supported"]) if h in p.index else None for h in HYP_NAMES}
    out = []
    for h, name in HYP_NAMES.items():
        if h not in p.index:
            continue
        r = p.loc[h]
        note = f"（{_txt(r.get('alpha_note'))}）" if _txt(r.get("alpha_note")) else ""
        if st[h] is None:
            reason = _txt(r.get("untestable_reason")) or "未說明原因"
            extra = "；這本身也是一個發現，不代表憤怒和恐懼的作用相同" if reason == R_SEP else ""
            out.append(f"{name}：無法判斷（{reason}{extra}）。")
        elif st[h]:
            out.append(f"{name}：支持{note}。")
        else:
            out.append(f"{name}：不支持（{'；'.join(_why_not(r))}）{note}。")
    s1, s3, s4 = st["H1"], st["H3"], st["H4"]
    h4_reason = _txt(p.loc["H4", "untestable_reason"]) if "H4" in p.index else None
    axes3, partial3, kept3 = (_axes(frozen, "anger__resid3", "VAD") if h4_reason != R_AXIS
                              else ("", False, ""))
    if partial3:   # spec §6.3: always state which axes the anger residual removed
        out.append(f"H4 的憤怒殘差在這一層只扣掉了{axes3}；{kept3}不可靠，沒有扣掉，"
                   "所以和 H4 有關的結論都只能針對扣掉的這幾個軸。")
    caveat3 = f"（H4 實際只扣掉了{axes3}，所以只能說可以用這幾個軸解釋。）" if partial3 else ""
    if s1 is True and s3 is not True and s4 is False:
        out.append("H1 成立，但 H3、H4 不成立 → 憤怒的效果可以用三個通用軸解釋。" + caveat3)
    if s3 is True and s4 is False:
        axes2, partial2, _ = _axes(frozen, "anger__resid2", "VA")
        out.append(
            "H3 成立，H4 不成立 → 憤怒和恐懼的作用不同，但差異可以用三個通用軸（很可能是掌控感）"
            "解釋。" + caveat3 + "請再看次要檢驗中「只扣兩個軸」（anger_resid2_vs_resid3）的"
            "結果，判斷掌控感是否就是關鍵。"
            + (f"（「只扣兩個軸」的版本實際只扣掉了{axes2}。）" if partial2 else ""))
    if s4 is True:
        out.append(f"H4 成立 → 憤怒有超出{axes3}這幾個軸的作用。H4 只扣掉了這幾個可靠的軸，"
                   "結論只能寫到這裡。" if partial3 else
                   "H4 成立 → 憤怒有超出三個通用軸的作用。這是最強的結論。")
    if s3 is True:
        out.append(_h3_direction(p.loc["H3"]))
    out.append("以上結果只描述調整模型內部方向後答案的變化，不代表模型有情緒或主觀感受。")
    return out
