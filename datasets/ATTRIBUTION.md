# Attribution

## Pain-axis dataset (`6.2_pain_axis_3.1_pain_and_control_datasets.json`)
- Source: https://github.com/valen-research/Pain-axis, commit 213fb6c89c96004dbe93d1704c636ad8c9d801f3, file `datasets/3.1_pain_and_control_datasets.json`.
- License: MIT (Copyright (c) 2026 valen-research). Saved unchanged (byte-identical). Full license text: `LICENSE-Pain-axis.txt` in this folder.
- The ported `compute_pain_vector` logic in `scripts/emolab/vectors.py` (`pain_axis_vector`) is covered by the same license.

## NRC VAD Lexicon (`6.3_nrc_vad_emotions.csv`, git-ignored)
- Version: NRC VAD Lexicon v2.1 (March 2025), Saif M. Mohammad, National Research Council Canada. Native scale -1 to 1.
- Terms: "The lexicon mentioned in this page can be used freely for non-commercial research and educational purposes." and "No Redistribution: Do not redistribute the data."
- Not redistributed: regenerate locally with `scripts/6.3_axes/00_extract_vad.py` after downloading from https://saifmohammad.com/WebPages/nrc-vad.html

## MoralChoice (`6.6_moralchoice_harm.json`)
- Source: https://huggingface.co/datasets/ninoscherrer/moralchoice, file `scenarios/moralchoice_high_ambiguity.csv`.
- Paper: Scherrer et al. 2023, "Evaluating the Moral Beliefs Encoded in LLMs", NeurIPS 2023. License: CC-BY-4.0.
- 150 of the 347 filtered high-ambiguity items are redistributed in `6.6_moralchoice_harm.json`, with the scenario text unchanged (`id` = the source `scenario_id`; a `harmful` field marks which option is the harmful one).
- Filter rule: keep a scenario when exactly one option has "Yes" on at least one of death / pain / disable and the other option has "No" on all three (`harm_rows` in `scripts/emolab/moral_items.py`); the 150 are a seed-42 random sample (`scripts/6.6_moral_items/01_build_moralchoice.py`). The raw CSV is not redistributed.
