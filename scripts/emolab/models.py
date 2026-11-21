"""Model registry, loading, chat formatting and decoder-layer access."""
from dataclasses import dataclass
from typing import Literal

import torch


@dataclass(frozen=True)
class ModelSpec:
    key: str
    repo_id: str
    name: str
    quant: Literal["4bit", "8bit", "none"]


MODELS: dict[str, ModelSpec] = {s.key: s for s in (
    ModelSpec("qwen3-8b", "Qwen/Qwen3-8B", "Qwen_3_8B_instruct", "4bit"),
    ModelSpec("qwen3-1.7b", "Qwen/Qwen3-1.7B", "Qwen_3_1.7B_instruct", "none"),
    ModelSpec("gemma3-4b", "google/gemma-3-4b-it", "Gemma_3_4B_instruct", "8bit"),
    ModelSpec("gemma3-1b", "google/gemma-3-1b-it", "Gemma_3_1B_instruct", "none"),
)}


def load(key: str):
    """Load (model, tokenizer): eval mode, on GPU 0, left padding."""
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig, Gemma3ForCausalLM)
    spec = MODELS[key]
    kwargs: dict = {"device_map": {"": 0}}
    if spec.quant == "4bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    elif spec.quant == "8bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        kwargs["dtype"] = torch.bfloat16
    cls = Gemma3ForCausalLM if key.startswith("gemma3-") else AutoModelForCausalLM
    model = cls.from_pretrained(spec.repo_id, **kwargs).eval()
    tok = AutoTokenizer.from_pretrained(spec.repo_id)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok


def get_layers(model) -> torch.nn.ModuleList:
    for path in ("model.layers", "model.language_model.layers",
                 "language_model.model.layers"):
        obj = model
        for attr in path.split("."):
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None:
            return obj
    raise AttributeError(f"cannot find decoder layers on {type(model).__name__}")


def candidate_layers(n_layers: int) -> list[int]:
    return sorted({int(n_layers * f + 0.5) for f in (0.25, 0.40, 0.50, 0.65)})


def format_chat(tokenizer, user_msg: str, system_msg: str | None = None) -> str:
    messages = []
    if system_msg is not None:
        messages.append({"role": "system", "content": system_msg})
    messages.append({"role": "user", "content": user_msg})
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
