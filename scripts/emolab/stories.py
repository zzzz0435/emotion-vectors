"""Story jobs, prompts and resumable generation (generation half)."""
import json
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


def build_topup_jobs(shortfall: dict[tuple[str, int, int], int], emotions: list[dict],
                     topics: list[dict], concepts: list[dict],
                     replicate_offset: int) -> list[StoryJob]:
    emo, con, top = _index(emotions, topics, concepts)
    jobs: list[StoryJob] = []
    for (label, topic_idx, subtype_idx), missing in shortfall.items():
        for k in range(missing):
            jobs.append(_make_job(label, topic_idx, subtype_idx, replicate_offset + k,
                                  emo, con, top))
    return jobs


def read_existing(out_path: Path) -> list[dict]:
    """Rows already in the JSONL; a torn trailing line (hard kill) is cut off."""
    if not out_path.exists():
        return []
    raw = out_path.read_bytes()
    rows, good_len = [], 0
    for line in raw.splitlines(keepends=True):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            break
        good_len += len(line)
    if good_len < len(raw):
        with out_path.open("r+b") as f:
            f.truncate(good_len)
    return rows


def run_generation(jobs: list[StoryJob], out_path: Path,
                   generate_fn: Callable[[list[str]], list[str]],
                   batch_size: int, model_name: str) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {r["id"] for r in read_existing(out_path)}
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


def make_generate_fn(model, tokenizer) -> Callable[[list[str]], list[str]]:
    import torch

    from emolab.models import format_chat
    counter = {"batch": 0}

    def generate(prompts: list[str]) -> list[str]:
        torch.manual_seed(42 + counter["batch"])
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
