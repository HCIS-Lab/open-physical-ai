from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Context:
    name: str
    session_directory: Path
    manifest_path: Path | None = None
