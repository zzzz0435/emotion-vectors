"""Project directories and the (out-of-repo) cache location."""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASETS = REPO_ROOT / "datasets"
RESULTS = REPO_ROOT / "results"
DEFAULT_CACHE = Path.home() / ".cache" / "emotional"


def cache_dir() -> Path:
    """Large intermediates live here; must not be inside the repo (OneDrive)."""
    path = Path(os.environ.get("EMOTIONAL_CACHE", DEFAULT_CACHE)).resolve()
    if path == REPO_ROOT or REPO_ROOT in path.parents:
        raise ValueError(f"cache dir must be outside the repo: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def results_dir(section: str, model_name: str | None = None) -> Path:
    path = RESULTS / section
    if model_name:
        path = path / model_name
    path.mkdir(parents=True, exist_ok=True)
    return path
