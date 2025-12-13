from __future__ import annotations

import re
import zipfile
from pathlib import Path
from typing import Iterable, Optional

AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"}
TRANSCRIPT_EXTS = {".txt", ".docx", ".json"}

def secure_extract_zip(zip_path: Path, extract_dir: Path) -> None:
    """Extract zip safely (prevents Zip Slip)."""
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            member_path = Path(member.filename)
            if member_path.is_absolute():
                raise ValueError(f"Zip contains absolute path: {member.filename}")
            resolved = (extract_dir / member_path).resolve()
            if not str(resolved).startswith(str(extract_dir.resolve())):
                raise ValueError(f"Zip traversal attempt: {member.filename}")
            resolved.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member, "r") as src, open(resolved, "wb") as dst:
                dst.write(src.read())

def walk_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file():
            yield p

def normalize_key(name: str) -> str:
    """Normalize a filename stem to match audio <-> transcript."""
    s = name.lower().strip()
    s = re.sub(r"^gt[_\-\s]+", "", s)  # remove gt_ prefix if any
    s = re.sub(r"\s+", " ", s)
    s = s.replace("-", "_")
    s = re.sub(r"_+", "_", s)
    return s

def guess_pair_key(path: Path) -> str:
    return normalize_key(path.stem)

def human_seconds(seconds: Optional[float]) -> str:
    if seconds is None:
        return "-"
    seconds = int(round(seconds))
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"
