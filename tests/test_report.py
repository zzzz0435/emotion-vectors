import pandas as pd

from emolab.report import render_checkpoint1

CORE = ["anger", "fear", "sadness", "joy", "compassion", "pain"]


def valid_df(low=()):
    rows = []
    for lab in [*CORE, "pain_painaxis", "calm"]:
        for L, a in [(9, 0.7), (18, 0.9)]:
            if lab in low:
                a -= 0.3
            rows.append({"label": lab, "layer": L, "auroc_vs_neutral": a})
    return pd.DataFrame(rows)


def axes(v_ok=True):
    def lay(r):
        return {k: {"alpha": 1.0, "loo_r": r if k == "V" else 0.8,
                    "reliable": (r >= .5) if k == "V" else True} for k in "VAD"}
    r = 0.7 if v_ok else 0.1
    st = "ok" if v_ok else "untestable"
    return {"fits": {"all": {"9": lay(r), "18": lay(r)}},
            "residuals": {"anger__resid3": {"9": st, "18": st}}}


CONTAM = {"18": {"punishment": {"anger": {"cos": .4, "ref_cos": .1, "p95_concepts": .2, "flagged": True},
                                "fear": {"cos": .0, "ref_cos": .1, "p95_concepts": .2, "flagged": False}},
                 "harming_others": {"anger": {"cos": .0, "ref_cos": .1, "p95_concepts": .2, "flagged": False}}}}
SAE = {"anger": [(123, 0.31)], "fear": [(7, 0.2)]}
PAIN = {"their_layer": 34, "our_layer": 34, "layer_match": True, "primary": "s2",
        "extraction_mismatch": "theirs final_token, ours mean", "model_mismatch": "theirs base, ours chat",
        "vectors": {"s2": {"cos_pain": 0.5, "cos_pain_painaxis": 0.4,
                           "cos_baselines": {"sadness": 0.3, "fear": 0.1, "anger": 0.05, "joy": -0.1}}}}
OK_SEP = {"18": {"auroc": 0.8, "separable": True}}


def render(sep, df=None, ax=None, contam=CONTAM):
    return render_checkpoint1(df if df is not None else valid_df(), sep, contam,
                              ax or axes(), SAE, "x.md", pain_compare=PAIN)


def test_render_checkpoint1_contains_key_sections():
    s = render_checkpoint1(valid_df(), {"18": {"auroc": 0.8123, "separable": True}},
                           CONTAM, axes(), SAE, "x.md")
    for h in ["愤怒 vs 恐惧", "污染检查", "三个轴", "特征字典", "范例文字"]:
        assert h in s
    assert "0.81" in s and "可以分开" in s


def test_not_separable_says_h3_untestable():
    s = render({"9": {"auroc": 0.6, "separable": False}, "18": {"auroc": 0.55, "separable": False}})
    assert "没有分开" in s and "H3" in s and "无法检验" in s and "可以分开" not in s


def test_v_axis_unreliable_says_h4_untestable():
    s = render(OK_SEP, ax=axes(v_ok=False))
    assert "H4 无法检验" in s and "不可靠" in s


def test_v_axis_reliable_no_h4_untestable():
    assert "H4 无法检验" not in render(OK_SEP)


def test_low_auroc_core_flagged():
    s = render(OK_SEP, df=valid_df(low=("fear",)))
    assert "警告" in s and "fear" in s.split("警告")[1].split("\n")[0]
    assert "警告" not in render(OK_SEP)


def test_contamination_flags_listed_per_emotion_and_probe():
    s = render(OK_SEP)
    section = s.split("## 三")[1].split("## 四")[0]
    assert "| anger | punishment | 18 |" in section
    assert "| fear |" not in section and "harming_others" not in section
    assert "没有任何情绪被标记" in render(OK_SEP, contam={"18": {"punishment": {"anger": {"flagged": False}}}})


def test_pain_section_and_number_format():
    s = render(OK_SEP)
    assert "疼痛比较突出" in s and "模型不同" in s
    assert "| 我们的 pain | 0.50 |" in s and "| sadness（对照） | 0.30 |" in s


def test_table_order_core_first():
    s = render(OK_SEP)
    assert s.index("| pain_painaxis |") < s.index("| calm |")


def _load_08():
    import importlib.util
    from pathlib import Path
    p = Path(__file__).parents[1] / "scripts" / "6.2_emotion_vectors" / "08_checkpoint_examples.py"
    spec = importlib.util.spec_from_file_location("ckpt08", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_examples_72_passages_baseline_shared():
    m = _load_08()
    gens = {(None if a == 0 else e, pi, a): f"t-{e}-{pi}-{a}"
            for e in CORE for pi in range(3) for a in m.ALPHAS}
    for pi in range(3):
        gens[(None, pi, 0.0)] = f"base-{pi}"
    md = m.render_examples(CORE, gens)
    assert md.count("**α = ") == 72
    assert md.count("base-0") == 6 and "暂定" in md
    assert md.index("## anger") < md.index("## pain")


def test_parse_sae_tops():
    md = "## anger\n\n### 1. 特征 12（余弦 0.310）\n\n- x\n### 2. 特征 5（余弦 0.200）\n## fear\n\n### 1. 特征 7（余弦 -0.100）\n## joy\n### 1. 特征 9（余弦 0.5）\n"
    assert _load_08().parse_sae_tops(md) == {"anger": [(12, 0.31), (5, 0.2)], "fear": [(7, -0.1)]}
