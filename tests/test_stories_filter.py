import json

from emolab.paths import DATASETS
from emolab.stories import CONTAMINATION_PATTERNS, build_jobs, check_story, select_balanced, targets_from_jobs

PAD = "word " * 100


def test_flags_thinking_leak():
    assert "thinking_leak" in check_story("<think>hmm</think> " + PAD, [], "emotion")


def test_flags_emotion_inflections():
    pats = [r"\bangr\w*", r"\bmad\b"]
    assert "names_emotion" in check_story("He answered Angrily. " + PAD, pats, "emotion")
    assert check_story("She made tea. " + PAD, pats, "emotion") == []


def test_contamination_only_for_non_contamination_kind():
    t = "He was punished. " + PAD
    assert "contamination" in check_story(t, [], "emotion")
    assert check_story(t, [], "contamination") == []
    assert "names_emotion" in check_story("He was mad. " + PAD, [r"\bmad\b"], "contamination")


def test_word_bounds():
    assert "too_short" in check_story("word " * 50, [], "neutral")
    assert "too_long" in check_story("word " * 400, [], "neutral")


def test_flags_non_english():
    assert "non_english" in check_story("She held a 硌 stone. " + PAD, [], "neutral")
    assert "non_english" in check_story("こんにちは " + PAD, [], "neutral")
    assert check_story("A café, naïve. " + PAD, [], "neutral") == []


def test_contamination_patterns_cover_examples():
    assert len(CONTAMINATION_PATTERNS) == 19
    assert "contamination" in check_story("They jailed him. " + PAD, [], "concept")


def row(label, t, s, r, text=PAD, kind="emotion"):
    return dict(label=label, topic_idx=t, subtype_idx=s, replicate=r, text=text, kind=kind)


def test_select_balanced_reports_shortfall():
    rows = [row("anger", 0, 0, 0), row("anger", 0, 0, 1, "He was punished. " + PAD)]
    kept, short = select_balanced(rows, {("anger", 0, 0): 2})
    assert [r["replicate"] for r in kept] == [0]
    assert short == {("anger", 0, 0): 1}


def test_select_balanced_takes_first_n_by_replicate_and_empty_cells():
    rows = [row("anger", 0, 0, r) for r in (2, 0, 1)]
    kept, short = select_balanced(rows, {("anger", 0, 0): 2, ("fear", 1, 0): 2})
    assert sorted(r["replicate"] for r in kept) == [0, 1]
    assert short == {("fear", 1, 0): 2}


def test_select_balanced_uses_banned_and_kind():
    rows = [dict(row("anger", 0, 0, 0, "He was mad. " + PAD))]
    kept, short = select_balanced(rows, {("anger", 0, 0): 1}, banned={"anger": [r"\bmad\b"]})
    assert kept == [] and short == {("anger", 0, 0): 1}


def test_targets_from_jobs_matches_table():
    emotions = json.loads((DATASETS / "6.2_emotions.json").read_text(encoding="utf-8"))
    topics = json.loads((DATASETS / "6.2_story_topics.json").read_text(encoding="utf-8"))
    concepts = json.loads((DATASETS / "6.2_concepts.json").read_text(encoding="utf-8"))
    t = targets_from_jobs(build_jobs(emotions, topics, concepts))
    assert sum(t[k] for k in t if k[0] == "anger") == 100
    assert sum(t[k] for k in t if k[0] == "calm") == 50
    assert sum(t[k] for k in t if k[0] == "pain") == 100
