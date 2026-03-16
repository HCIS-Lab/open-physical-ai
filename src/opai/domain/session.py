from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DemoAsset:
    demo_id: str
    source_path: str
    stored_path: str
    original_filename: str


@dataclass(frozen=True)
class MappingAsset:
    source_path: str
    stored_path: str
    original_filename: str


@dataclass(frozen=True)
class SessionManifest:
    session_name: str
    demos: tuple[DemoAsset, ...]
    mapping: MappingAsset | None = None
