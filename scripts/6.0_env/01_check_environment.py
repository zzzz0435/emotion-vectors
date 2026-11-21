"""Check GPU / library versions and (optionally) model load, VRAM and speed.

Reads : nothing (models are loaded from the Hugging Face cache)
Writes: results/6.0_env/env_report.json
Needs : torch (CUDA), transformers, bitsandbytes
Usage : python scripts/6.0_env/01_check_environment.py --models qwen3-1.7b qwen3-8b
"""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
from importlib import metadata

import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "6.0_env" / "env_report.json"
MODEL_CHOICES = ["qwen3-1.7b", "qwen3-8b", "gemma3-1b", "gemma3-4b"]


def version_info() -> dict:
    cuda = torch.cuda.is_available()
    props = torch.cuda.get_device_properties(0) if cuda else None
    return {
        "gpu": props.name if props else None,
        "vram_total_mb": round(props.total_memory / 2**20) if props else None,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": metadata.version("transformers"),
        "bitsandbytes": metadata.version("bitsandbytes"),
    }


def check_model(key: str) -> dict:
    """Load model `key`, measure peak VRAM and tokens/s for 64 new tokens.

    Returns {"loaded", "peak_vram_mb", "tokens_per_s", "error"}; Gemma failures
    write error = "需要用户登录 Hugging Face 并同意 Gemma 条款" without raising.
    """
    import gc
    import time

    from emolab.models import format_chat, load

    failed = {"loaded": False, "peak_vram_mb": 0.0, "tokens_per_s": 0.0}
    model = None
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model, tok = load(key)
        inputs = tok(format_chat(tok, "Write one sentence about the sea."),
                     return_tensors="pt", add_special_tokens=False).to("cuda:0")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model.generate(**inputs, max_new_tokens=64, min_new_tokens=64,
                             do_sample=False)
        torch.cuda.synchronize()
        n_new = out.shape[1] - inputs["input_ids"].shape[1]
        return {"loaded": True,
                "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1),
                "tokens_per_s": round(n_new / (time.perf_counter() - t0), 2),
                "error": None}
    except Exception as e:
        if key.startswith("gemma3-"):
            return {**failed, "error": "需要用户登录 Hugging Face 并同意 Gemma 条款"}
        return {**failed, "error": repr(e)}
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=[], choices=MODEL_CHOICES)
    args = ap.parse_args()
    report = version_info()
    report["models"] = {}
    for key in args.models:
        try:
            report["models"][key] = check_model(key)
        except Exception as e:  # one failing model must not stop the others
            report["models"][key] = {"loaded": False, "peak_vram_mb": 0.0,
                                     "tokens_per_s": 0.0, "error": repr(e)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
