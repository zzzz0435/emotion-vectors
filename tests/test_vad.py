import pytest

from emolab.vad import extract_vad

LEX = "term\tvalence\tarousal\tdominance\nanger\t-0.666\t0.730\t0.314\njoy\t0.960\t0.648\t0.588\nother\t0\t0\t0\n"


def test_extract_vad_success(tmp_path):
    p = tmp_path / "lex.txt"
    p.write_text(LEX, encoding="utf-8")
    df = extract_vad(p, {"anger": "anger", "joy": "joy"})
    assert list(df.columns) == ["label", "word", "valence", "arousal", "dominance", "lexicon_version"]
    assert df.set_index("label").loc["anger", "valence"] == -0.666
    assert len(df) == 2


def test_extract_vad_missing_word_raises(tmp_path):
    p = tmp_path / "lex.txt"
    p.write_text(LEX, encoding="utf-8")
    with pytest.raises(ValueError, match="compassion"):
        extract_vad(p, {"anger": "anger", "compassion": "compassion"})
