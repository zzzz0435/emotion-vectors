"""Extract the 20 emotion headwords from the NRC VAD Lexicon v2.1 (native -1..1 scale).

Reads : the lexicon file, by default <cache_dir>/nrc_vad/NRC-VAD-Lexicon-v2.1/NRC-VAD-Lexicon-v2.1.txt
        (download the zip from https://saifmohammad.com/WebPages/nrc-vad.html and unzip it there;
        the lexicon must not be redistributed, so it is never stored in the repo)
Writes: datasets/6.3_nrc_vad_emotions.csv (git-ignored; regenerate with this script)
Needs : pandas
Usage : python scripts/6.3_axes/00_extract_vad.py [--lexicon PATH]
"""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

from emolab.paths import DATASETS, cache_dir
from emolab.vad import extract_vad

# label -> lexicon headword
MAPPING = {l: l for l in (
    "anger", "fear", "sadness", "joy", "compassion", "pain", "calm", "pride", "gratitude",
    "love", "hope", "excitement", "amusement", "jealousy", "disgust", "shame", "guilt",
    "contempt", "anxiety", "desperation")}


def main():
    default = cache_dir() / "nrc_vad" / "NRC-VAD-Lexicon-v2.1" / "NRC-VAD-Lexicon-v2.1.txt"
    ap = argparse.ArgumentParser()
    ap.add_argument("--lexicon", type=Path, default=default)
    args = ap.parse_args()
    try:
        df = extract_vad(args.lexicon, MAPPING)
    except (ValueError, FileNotFoundError) as e:
        sys.exit(f"error: {e}")
    out = DATASETS / "6.3_nrc_vad_emotions.csv"
    df.to_csv(out, index=False, lineterminator="\n")
    print(f"wrote {out} ({len(df)} rows)")


if __name__ == "__main__":
    main()
