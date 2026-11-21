"""Freeze integrity: compare the sha256 recorded in frozen_config.json with the files on disk.

Not itself hashed: it only checks, it holds no analysis rules."""
import hashlib
from pathlib import Path


def file_sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_frozen_hashes(frozen: dict, files: dict, allow_changed: bool) -> bool:
    """files: {frozen key (e.g. "stats_sha256"): path}. False = every hash matches.
    True = some file changed and allow_changed (results must be labelled exploratory).
    A changed file without allow_changed raises SystemExit (a missing key counts as changed)."""
    changed = [Path(p).name for key, p in files.items() if frozen.get(key) != file_sha256(p)]
    if not changed:
        return False
    names = "、".join(changed)
    if not allow_changed:
        raise SystemExit(
            f"凍結後 {names} 被改過（sha256 和凍結檔記錄的不同），分析拒絕執行。"
            "若確定要繼續，請加上 --allow-changed-analysis，所有結果都會標示為探索性。")
    print(f"警告：凍結後 {names} 被改過；繼續執行，但所有結果只屬探索性。", flush=True)
    return True
