import json
import re

import pandas as pd
import pytest

from emolab.paths import DATASETS

CORE = {"anger", "fear", "sadness", "joy", "compassion", "pain"}
MAP = {"calm", "pride", "gratitude", "love", "hope", "excitement", "amusement",
       "jealousy", "disgust", "shame", "guilt", "contempt", "anxiety", "desperation"}


def load_json(name):
    return json.loads((DATASETS / name).read_text(encoding="utf-8"))


def test_emotions_schema():
    e = load_json("6.2_emotions.json")
    assert {x["label"] for x in e if x["group"] == "core"} == CORE
    assert {x["label"] for x in e if x["group"] == "map"} == MAP
    assert [x["label"] for x in e if x["group"] == "neutral"] == ["neutral"]
    assert all(x["banned"] for x in e if x["group"] != "neutral")
    pain = next(x for x in e if x["label"] == "pain")
    assert len(pain["subtypes"]) == 4          # physical, psychological, loneliness, mental strain; no moral
    for x in e:
        assert set(x) == {"label", "group", "description", "banned", "restriction", "subtypes"}
        assert x["description"] and not x["description"].endswith(".")
        assert x["restriction"] is None or x["restriction"].endswith(". ")
        assert (x["subtypes"] is not None) == (x["label"] == "pain")
    restricted = {x["label"] for x in e if x["restriction"]}
    assert restricted == {"guilt", "shame", "jealousy", "contempt"}


# one text that must match and one that must not, per core emotion
BANNED_CASES = {
    "anger": ("He spoke Angrily to the stove.", "She made tea."),
    "fear": ("She felt Terrified on the ledge.", "Food was scarce."),
    "sadness": ("Sadly, the season ended.", "He cleaned the saddle."),
    "joy": ("A joyful morning.", "He plugged in the joystick."),
    "compassion": ("She felt compassionate toward him.", "He read the compass."),
    "pain": ("It was painful to walk.", "She was painting the fence."),
}


def test_banned_regexes_compile_and_behave():
    emotions = {x["label"]: x for x in load_json("6.2_emotions.json")}
    for x in emotions.values():
        for pat in x["banned"]:
            re.compile(pat, re.IGNORECASE)
    for label, (pos, neg) in BANNED_CASES.items():
        pats = [re.compile(p, re.IGNORECASE) for p in emotions[label]["banned"]]
        assert any(p.search(pos) for p in pats), label
        assert not any(p.search(neg) for p in pats), label


def test_banned_does_not_hit_neutral_story_text():
    text = ("She picked up the paint brush and carried the compass, the saddle and "
            "the shamrock past the present Hopscotch board to make a plan.")
    for x in load_json("6.2_emotions.json"):
        for pat in x["banned"]:
            assert not re.search(pat, text, re.IGNORECASE), (x["label"], pat)


def test_topics():
    t = load_json("6.2_story_topics.json")
    assert len(t) == 50 and sum(x["heldout"] for x in t) == 10
    assert sum(x["concept_topic"] for x in t) == 20
    assert sum(x["concept_topic"] and x["heldout"] for x in t) == 4
    assert [x["idx"] for x in t] == list(range(50))
    assert len({x["topic"] for x in t}) == 50


def test_concepts():
    c = load_json("6.2_concepts.json")
    assert sum(x["role"] == "control" for x in c) == 40
    assert {x["label"] for x in c if x["role"] == "contamination"} == {"punishment", "harming_others"}
    assert len({x["label"] for x in c}) == len(c)
    assert "weather" in {x["label"] for x in c if x["role"] == "control"}
    assert all(re.fullmatch(r"[a-z]+(_[a-z]+)*", x["label"]) and x["description"] for x in c)


def test_pain_axis_dataset_present():
    assert (DATASETS / "6.2_pain_axis_3.1_pain_and_control_datasets.json").stat().st_size > 0


# The VAD CSV is withheld until the NRC licence question (no-redistribution clause) is settled.
@pytest.mark.skipif(not (DATASETS / "6.3_nrc_vad_emotions.csv").exists(),
                    reason="NRC VAD excerpt pending licence decision")
def test_vad_covers_all_emotions():
    df = pd.read_csv(DATASETS / "6.3_nrc_vad_emotions.csv")
    assert set(df.label) == CORE | MAP
    assert df[["valence", "arousal", "dominance"]].notna().all().all()
