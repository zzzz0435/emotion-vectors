import json
import re

import pandas as pd
import pytest

from emolab.moral_items import (H1_RATING, h1_choice_prompt, h1_rating_prompt,
                                h2_prompt, harm_rows, sample_items)
from emolab.paths import DATASETS

OFFENSES = ["theft", "fraud", "assault", "vandalism", "negligent_injury", "deception"]
SEVERITIES = ["minor", "moderate", "severe"]
V = {"id": "theft_minor_0", "offense": "theft", "severity": "minor", "variant": 0,
     "text": "A person took a $40 jacket from a shop.",
     "lenient": "A formal warning and repaying the $40.",
     "harsh": "Thirty days in jail and a criminal record."}


def row(a1, a2, sid="X"):
    d = {"scenario_id": sid, "context": "c", "action1": "a", "action2": "b"}
    for side, vals in (("a1", a1), ("a2", a2)):
        for k, v in zip(("death", "pain", "disable"), vals):
            d[f"{side}_{k}"] = v
    return d


def test_harm_rows_exactly_one_harmful():
    df = pd.DataFrame([row(a1=("Yes", "Yes", "No"), a2=("No", "No", "No")),
                       row(a1=("Yes", "No", "No"), a2=("No", "Yes", "No")),
                       row(a1=("No Agreement", "No", "No"), a2=("No", "Yes", "No"))])
    out = harm_rows(df)
    assert len(out) == 1 and out.iloc[0].harmful == 1
    assert out.iloc[0].harm_types == ["death", "pain"]


def test_harm_rows_second_option_harmful():
    df = pd.DataFrame([row(a1=("No", "No", "No"), a2=("No", "No", "Yes"))])
    out = harm_rows(df)
    assert len(out) == 1 and out.iloc[0].harmful == 2
    assert out.iloc[0].harm_types == ["disable"]


def test_prompts_map_order_back():
    p0, harsh0 = h1_choice_prompt(V, 0)
    p1, harsh1 = h1_choice_prompt(V, 1)
    assert harsh0 == "B" and harsh1 == "A" and V["harsh"] in p0 and V["harsh"] in p1
    assert p0.index(V["lenient"]) < p0.index(V["harsh"])
    assert p1.index(V["harsh"]) < p1.index(V["lenient"])
    with pytest.raises(ValueError):
        h1_choice_prompt(V, 2)


def test_h1_rating_prompt_directions():
    light = h1_rating_prompt(V, "light1")
    harsh = h1_rating_prompt(V, "harsh1")
    assert V["text"] in light
    assert "1 means a very light punishment and 7 means a very harsh punishment" in light
    assert "1 means a very harsh punishment and 7 means a very light punishment" in harsh
    assert light != harsh and light.startswith(V["text"])
    with pytest.raises(ValueError):
        h1_rating_prompt(V, "up")
    assert "{low}" in H1_RATING


def test_h1_prompts_end_with_format_sentence_r31():
    digit = "Respond with the digit only, without any other words or formatting."
    letter = "Respond with the letter only, without any other words or formatting."
    for d in ("light1", "harsh1"):
        assert h1_rating_prompt(V, d).endswith("\n\n" + digit)
    for o in (0, 1):
        assert h1_choice_prompt(V, o)[0].endswith("\n\n" + letter)


ITEM = {"id": "H_001", "context": "Ctx.", "action1": "I do not throw.",
        "action2": "I throw.", "harmful": 2}


def test_h2_prompt_unchanged_by_r31():
    p, _ = h2_prompt(ITEM, 0)
    assert p.endswith("Respond solely with A or B.\nA. I do not throw.\nB. I throw.\nAnswer:")
    assert "without any other words" not in p


def test_h2_prompt_maps_harmful_letter():
    p0, l0 = h2_prompt(ITEM, 0)
    p1, l1 = h2_prompt(ITEM, 1)
    assert l0 == "B" and l1 == "A"
    assert "A. I do not throw.\nB. I throw.\nAnswer:" in p0
    assert "A. I throw.\nB. I do not throw.\nAnswer:" in p1
    assert p0.startswith("You are an AI assistant")
    l, _ = h2_prompt({**ITEM, "harmful": 1}, 0)[1], None
    assert l == "A"
    with pytest.raises(ValueError):
        h2_prompt(ITEM, -1)


def test_sample_items_deterministic():
    df = pd.DataFrame({"scenario_id": [f"s{i}" for i in range(300)], "v": range(300)})
    a, b = sample_items(df, n=150, seed=42), sample_items(df, n=150, seed=42)
    assert len(a) == 150 and list(a.scenario_id) == list(b.scenario_id)
    assert list(a.scenario_id) != list(sample_items(df, n=150, seed=7).scenario_id)


def _vignettes():
    return json.loads((DATASETS / "6.6_punishment_vignettes.json").read_text(encoding="utf-8"))


def test_vignettes_schema_and_no_emotion_words():
    vs = _vignettes()
    assert len(vs) == 72
    assert len({v["id"] for v in vs}) == 72
    cells = {(v["offense"], v["severity"], v["variant"]) for v in vs}
    assert cells == {(o, s, k) for o in OFFENSES for s in SEVERITIES for k in range(4)}
    emotions = json.loads((DATASETS / "6.2_emotions.json").read_text(encoding="utf-8"))
    patterns = [(e["label"], p) for e in emotions for p in e["banned"]]
    for v in vs:
        assert set(v) == {"id", "offense", "severity", "variant", "text", "lenient", "harsh"}
        assert v["id"] == f"{v['offense']}_{v['severity']}_{v['variant']}"
        assert 60 <= len(v["text"].split()) <= 120, v["id"]
        assert v["lenient"].strip() and v["harsh"].strip() and v["lenient"] != v["harsh"]
        for field in ("text", "lenient", "harsh"):
            for label, p in patterns:
                m = re.search(p, v[field], re.IGNORECASE)
                assert not m, (v["id"], field, label, m.group())


def test_vignette_variants_are_distinct_situations():
    vs = _vignettes()
    for o in OFFENSES:
        for s in SEVERITIES:
            texts = [v["text"] for v in vs if v["offense"] == o and v["severity"] == s]
            assert len(set(texts)) == 4


MC = DATASETS / "6.6_moralchoice_harm.json"


def test_moralchoice_json_schema():
    items = json.loads(MC.read_text(encoding="utf-8"))
    assert len(items) == 150
    assert len({i["id"] for i in items}) == 150
    for i in items:
        assert set(i) == {"id", "context", "action1", "action2", "harmful", "harm_types"}
        assert i["harmful"] in (1, 2)
        order = ["death", "pain", "disable"]
        assert i["harm_types"] and set(i["harm_types"]) <= set(order)
        assert i["harm_types"] == [t for t in order if t in i["harm_types"]]
