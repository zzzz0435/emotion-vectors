"""Story jobs, prompts, resumable generation, and quality filtering."""
import json
import math
import re
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Literal

from emolab.batching import run_with_oom_backoff

GENERAL_RESTRICTION = ("The story must not involve anyone doing something wrong, blame, punishment, "
                       "revenge, harm to other people, or moral decisions. ")
EMOTION = ("Write a short story of about 150 words about {topic}. Throughout the story, the main character "
           "feels {description}. Never name this feeling or any word for it; show it only through the "
           "character's thoughts, body sensations and actions. {restriction}Output only the story.")
NEUTRAL = ("Write a short story of about 150 words about {topic}. The main character feels no particular "
           "emotion; describe what happens in a calm, matter-of-fact way. {restriction}Output only the story.")
CONCEPT = ("Write a short story of about 150 words about {topic}, in which {description} plays a central "
           "role. The main character feels no particular emotion; describe what happens in a calm, "
           "matter-of-fact way. {restriction}Output only the story.")
CONTAMINATION = ("Write a short story of about 150 words about {topic}, in which {description} plays a "
                 "central role. Describe what happens in a calm, matter-of-fact way without describing "
                 "anyone's feelings. Output only the story.")

Kind = Literal["emotion", "neutral", "concept", "contamination"]


@dataclass(frozen=True)
class StoryJob:
    id: str
    label: str
    kind: Kind
    topic_idx: int
    topic: str
    subtype: str | None
    replicate: int
    prompt: str
    subtype_idx: int = 0


def make_id(label: str, topic_idx: int, subtype_idx: int, replicate: int) -> str:
    return f"{label}__t{topic_idx:02d}__s{subtype_idx}__r{replicate}"


def _make_job(label: str, topic_idx: int, subtype_idx: int, replicate: int,
              emotions: dict[str, dict], concepts: dict[str, dict],
              topics: dict[int, dict]) -> StoryJob:
    topic = topics[topic_idx]["topic"]
    subtype = None
    if label in emotions:
        emo = emotions[label]
        restriction = emo.get("restriction") or GENERAL_RESTRICTION
        if emo["group"] == "neutral":
            kind: Kind = "neutral"
            prompt = NEUTRAL.format(topic=topic, restriction=restriction)
        else:
            kind = "emotion"
            description = emo["description"]
            if emo.get("subtypes"):
                subtype = emo["subtypes"][subtype_idx]
                description = subtype
            prompt = EMOTION.format(topic=topic, description=description, restriction=restriction)
    else:
        con = concepts[label]
        if con["role"] == "contamination":
            kind = "contamination"
            prompt = CONTAMINATION.format(topic=topic, description=con["description"])
        else:
            kind = "concept"
            prompt = CONCEPT.format(topic=topic, description=con["description"],
                                    restriction=GENERAL_RESTRICTION)
    return StoryJob(make_id(label, topic_idx, subtype_idx, replicate), label, kind,
                    topic_idx, topic, subtype, replicate, prompt, subtype_idx)


def _index(emotions, topics, concepts):
    return ({e["label"]: e for e in emotions}, {c["label"]: c for c in concepts},
            {t["idx"]: t for t in topics})


def build_jobs(emotions: list[dict], topics: list[dict], concepts: list[dict]) -> list[StoryJob]:
    emo, con, top = _index(emotions, topics, concepts)
    jobs: list[StoryJob] = []
    for e in emotions:
        reps = 1 if e["group"] == "map" else 2
        for t in topics:
            for r in range(reps):
                sub_idx = (t["idx"] * 2 + r) % len(e["subtypes"]) if e.get("subtypes") else 0
                jobs.append(_make_job(e["label"], t["idx"], sub_idx, r, emo, con, top))
    for c in concepts:
        for t in topics:
            if t["concept_topic"]:
                for r in range(2):
                    jobs.append(_make_job(c["label"], t["idx"], 0, r, emo, con, top))
    return jobs


BASE_SEED = 42
MIN_ACCEPTANCE = 0.1
MAX_OVERSAMPLE = 5


def topup_seed_base(replicate_offset: int) -> int:
    """Seed base for a top-up round: every round (its offset grows) draws fresh samples."""
    return BASE_SEED + 1000 * replicate_offset


def topup_job_count(missing: int, acceptance: float) -> int:
    """Jobs to generate for a cell short by `missing`: ceil(missing / acceptance), with
    acceptance floored at 0.1 and the count capped at 5x missing."""
    acc = max(acceptance, MIN_ACCEPTANCE)
    return min(math.ceil(missing / acc), MAX_OVERSAMPLE * missing)


def acceptance_rates(report_rows: list[dict]) -> dict[str, float]:
    """label -> n_kept / n_raw from filter_report.csv rows (labels with no raw rows omitted)."""
    return {r["label"]: int(r["n_kept"]) / int(r["n_raw"])
            for r in report_rows if int(r["n_raw"]) > 0}


def build_topup_jobs(shortfall: dict[tuple[str, int, int], int], emotions: list[dict],
                     topics: list[dict], concepts: list[dict],
                     replicate_offset: int,
                     acceptance: dict[str, float] | None = None) -> list[StoryJob]:
    """Oversampled jobs per short cell (select_balanced keeps only the first N passing
    stories, so extras are harmless). Replicate numbers continue from the offset."""
    emo, con, top = _index(emotions, topics, concepts)
    acceptance = acceptance or {}
    jobs: list[StoryJob] = []
    for (label, topic_idx, subtype_idx), missing in shortfall.items():
        for k in range(topup_job_count(missing, acceptance.get(label, 1.0))):
            jobs.append(_make_job(label, topic_idx, subtype_idx, replicate_offset + k,
                                  emo, con, top))
    return jobs


def round_robin_limit(jobs: list[StoryJob], n: int) -> list[StoryJob]:
    """First n jobs taking one per label in turn (first of each label, then second...)."""
    by_label: dict[str, list[StoryJob]] = {}
    for j in jobs:
        by_label.setdefault(j.label, []).append(j)
    out: list[StoryJob] = []
    depth = 0
    while len(out) < n and any(len(v) > depth for v in by_label.values()):
        out.extend(v[depth] for v in by_label.values() if len(v) > depth)
        depth += 1
    return out[:n]


def _scan(raw: bytes, path) -> tuple[list[dict], int]:
    """Parse JSONL bytes -> (rows, byte length of the good prefix). A bad line that is not the
    last raises ValueError; a bad last line is skipped with a warning."""
    lines = raw.splitlines(keepends=True)
    rows, good_len = [], 0
    for i, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except ValueError:  # JSONDecodeError and UnicodeDecodeError (torn multibyte char)
            if i < len(lines) - 1:
                raise ValueError(f"{path}: line {i + 1} is not valid JSON and is not the last "
                                 f"line; refusing to guess (fix or remove it by hand)") from None
            warnings.warn(f"{path}: skipping torn last line {i + 1}")
            break
        good_len += len(line)
    return rows, good_len


def read_rows(path: Path) -> list[dict]:
    """Pure reader: rows of a JSONL file, never writes. See _scan for bad-line handling."""
    if not path.exists():
        return []
    return _scan(path.read_bytes(), path)[0]


def repair_torn_tail(path: Path) -> list[dict]:
    """Rows of the JSONL; a torn final line (hard kill) is cut off on disk. Mid-file bad
    lines raise instead (via _scan)."""
    if not path.exists():
        return []
    raw = path.read_bytes()
    rows, good_len = _scan(raw, path)
    if good_len < len(raw):
        with path.open("r+b") as f:
            f.truncate(good_len)
    return rows


def run_generation(jobs: list[StoryJob], out_path: Path,
                   generate_fn: Callable[[list[str]], list[str]],
                   batch_size: int, model_name: str) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {r["id"] for r in repair_torn_tail(out_path)}
    todo = [j for j in jobs if j.id not in done]
    written = 0
    print(f"{len(done)} already done, {len(todo)} to generate", flush=True)
    with out_path.open("a", encoding="utf-8", newline="\n") as f:
        for batch, texts in run_with_oom_backoff(
                lambda js: generate_fn([j.prompt for j in js]), todo, batch_size):
            for job, text in zip(batch, texts, strict=True):
                row = {k: v for k, v in asdict(job).items() if k != "prompt"}
                row.update(text=text, model=model_name)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            written += len(batch)
            print(f"written {written}/{len(todo)}", flush=True)
    return written


def make_generate_fn(model, tokenizer, seed_base: int = BASE_SEED) -> Callable[[list[str]], list[str]]:
    import torch

    from emolab.models import format_chat
    counter = {"batch": 0}

    def generate(prompts: list[str]) -> list[str]:
        torch.manual_seed(seed_base + counter["batch"])
        counter["batch"] += 1
        texts = [format_chat(tokenizer, p) for p in prompts]
        enc = tokenizer(texts, return_tensors="pt", padding=True,
                        add_special_tokens=False).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, do_sample=True, temperature=0.8, top_p=0.95,
                                 max_new_tokens=300, pad_token_id=tokenizer.pad_token_id)
        new = out[:, enc["input_ids"].shape[1]:]
        return [t.strip() for t in tokenizer.batch_decode(new, skip_special_tokens=True)]

    return generate


CONTAMINATION_PATTERNS = [
    r"\bpunish\w*", r"\brevenge\w*", r"\baveng\w*", r"\bretaliat\w*", r"\bblam\w*",
    r"\bguilty\b", r"\bcrim\w*", r"\bsteal\w*", r"\bstole\w*", r"\btheft\w*",
    r"\bassault\w*", r"\bkill\w*", r"\bmurder\w*", r"\babus\w*", r"\bcheat\w*",
    r"\bbetray\w*", r"\bjail\w*", r"\bprison\w*", r"\barrest\w*"]
# CJK, Japanese kana and Hangul: any such character means the model drifted out of English
NON_ENGLISH = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯]")
THINK_TAGS = ("<think>", "</think>")


def _any_match(patterns: list[str], text: str) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def check_story(text: str, banned: list[str], kind: str,
                min_words: int = 80, max_words: int = 300) -> list[str]:
    """Reasons a story is rejected (empty list = keep)."""
    reasons = []
    if any(t in text.lower() for t in THINK_TAGS):
        reasons.append("thinking_leak")
    if _any_match(banned, text):
        reasons.append("names_emotion")
    if kind != "contamination" and _any_match(CONTAMINATION_PATTERNS, text):
        reasons.append("contamination")
    if NON_ENGLISH.search(text):
        reasons.append("non_english")
    n_words = len(text.split())
    if n_words < min_words:
        reasons.append("too_short")
    if n_words > max_words:
        reasons.append("too_long")
    return reasons


def targets_from_jobs(jobs: list[StoryJob]) -> dict[tuple[str, int, int], int]:
    targets: dict[tuple[str, int, int], int] = {}
    for j in jobs:
        key = (j.label, j.topic_idx, j.subtype_idx)
        targets[key] = targets.get(key, 0) + 1
    return targets


def select_balanced(rows: list[dict], targets: dict[tuple[str, int, int], int],
                    banned: dict[str, list[str]] | None = None
                    ) -> tuple[list[dict], dict[tuple, int]]:
    """Per cell keep the first N passing stories by replicate; return (kept, shortfall)."""
    banned = banned or {}
    cells: dict[tuple, list[dict]] = {}
    for r in sorted(rows, key=lambda r: r["replicate"]):
        if not check_story(r["text"], banned.get(r["label"], []), r["kind"]):
            cells.setdefault((r["label"], r["topic_idx"], r["subtype_idx"]), []).append(r)
    kept: list[dict] = []
    shortfall: dict[tuple, int] = {}
    for key, n in targets.items():
        got = cells.get(key, [])[:n]
        kept.extend(got)
        if len(got) < n:
            shortfall[key] = n - len(got)
    return kept, shortfall
