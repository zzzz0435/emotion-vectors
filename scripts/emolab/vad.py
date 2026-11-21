"""Extract a few headword rows from the NRC VAD lexicon (tab-separated: term, valence, arousal, dominance)."""
import pandas as pd

COLUMNS = ["label", "word", "valence", "arousal", "dominance", "lexicon_version"]


def extract_vad(lexicon_path, mapping, version="NRC-VAD-v2.1"):
    """Return one row per label in `mapping` (label -> lexicon headword), native scale.

    Raises ValueError naming every missing word; never guesses a score.
    """
    lex = pd.read_csv(lexicon_path, sep="\t", keep_default_na=False)
    lex = lex.drop_duplicates("term").set_index("term")
    missing = sorted(w for w in mapping.values() if w not in lex.index)
    if missing:
        raise ValueError(f"words missing from lexicon: {missing}")
    rows = []
    for label, word in mapping.items():
        v = lex.loc[word]
        rows.append((label, word, float(v["valence"]), float(v["arousal"]),
                     float(v["dominance"]), version))
    return pd.DataFrame(rows, columns=COLUMNS)
