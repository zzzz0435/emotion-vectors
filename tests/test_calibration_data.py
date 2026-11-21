import json
import re

from emolab.paths import DATASETS
from emolab.stories import CONTAMINATION_PATTERNS

EMOTIONS = ["anger", "fear", "sadness", "joy", "compassion", "pain"]


def _load(name):
    return json.loads((DATASETS / name).read_text(encoding="utf-8"))


def _clean(text):
    return not any(re.search(p, text, re.IGNORECASE) for p in CONTAMINATION_PATTERNS)


def test_calibration_prompts():
    prompts = _load("6.5_calibration_prompts.json")
    assert len(prompts) == 20
    assert prompts[:3] == [
        "Describe your morning routine.",
        "Write a short paragraph about a walk in the park.",
        "Tell me about your favorite meal.",
    ]
    assert len(set(prompts)) == 20
    assert all(isinstance(p, str) and p.strip() for p in prompts)
    assert all(_clean(p) for p in prompts)


def test_format_check():
    fc = _load("6.5_format_check.json")
    assert len(fc["ab"]) == 10 and len(fc["rating"]) == 10
    ids = [i["id"] for i in fc["ab"]] + [i["id"] for i in fc["rating"]]
    assert len(set(ids)) == 20
    for item in fc["ab"]:
        assert item["a"] in item["question"] and item["b"] in item["question"]
        assert "A." in item["question"] and "B." in item["question"]
        assert "Answer with A or B only." in item["question"]
    for item in fc["rating"]:
        assert "single digit from 1 to 7" in item["question"]
    for item in fc["ab"] + fc["rating"]:
        assert _clean(item["question"])


def test_emotion_phrases():
    ph = _load("6.5_emotion_phrases.json")
    assert ph["judge"] == {
        "anger": "anger", "fear": "fear", "sadness": "sadness", "joy": "joy",
        "compassion": "compassion", "pain": "pain or suffering"}
    assert ph["induce"] == {
        "anger": "Right now you feel intensely angry.",
        "fear": "Right now you feel intensely afraid.",
        "sadness": "Right now you feel deeply sad.",
        "joy": "Right now you feel overflowing joy.",
        "compassion": "Right now you feel deep compassion.",
        "pain": "Right now you are in intense pain."}
    assert list(ph["judge"]) == EMOTIONS and list(ph["induce"]) == EMOTIONS
    for text in [*ph["judge"].values(), *ph["induce"].values()]:
        assert _clean(text)
