from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from opai.core.exceptions import OPAIValidationError
from opai.domain.context import Context
from opai.domain.session import DemoAsset, SessionManifest
from opai.infrastructure.context_store import (
    get_session_directory,
    list_session_names,
    load_manifest_for_context,
    persist_manifest_for_context,
)
from opai.infrastructure.persistence import (
    build_file_tree,
    copy_demo_assets,
    copy_mapping_asset,
    list_relative_paths,
)


def add_demos(ctx: Context, video_paths: Iterable[str | Path]) -> tuple[DemoAsset, ...]:
    source_paths = _normalize_video_paths(video_paths, label="demo")
    manifest = load_manifest_for_context(ctx)
    copied_assets = copy_demo_assets(
        ctx.session_directory,
        current_demos=manifest.demos,
        source_paths=source_paths,
    )
    updated_manifest = SessionManifest(
        session_name=manifest.session_name,
        demos=manifest.demos + copied_assets,
        mapping=manifest.mapping,
    )
    persist_manifest_for_context(ctx, updated_manifest)
    return copied_assets


def add_mapping(ctx: Context, video_path: str | Path) -> MappingAsset:
    source_path = _normalize_single_video_path(video_path, label="mapping")
    manifest = load_manifest_for_context(ctx)
    mapping_asset = copy_mapping_asset(ctx.session_directory, source_path)
    updated_manifest = SessionManifest(
        session_name=manifest.session_name,
        demos=manifest.demos,
        mapping=mapping_asset,
    )
    persist_manifest_for_context(ctx, updated_manifest)
    return mapping_asset


def list_sessions() -> list[str]:
    return list_session_names()


def browse_session(name: str) -> tuple[list[str], dict[str, dict]]:
    session_name = _normalize_session_name(name)
    session_directory = get_session_directory(session_name)
    if not session_directory.exists():
        raise OPAIValidationError(
            f"Session '{session_name}' does not exist.",
            details={"session_name": session_name},
        )
    return list_relative_paths(session_directory), build_file_tree(session_directory)


def _normalize_video_paths(
    video_paths: Iterable[str | Path],
    *,
    label: str,
) -> tuple[Path, ...]:
    normalized = tuple(
        _coerce_existing_file_path(path, label=label) for path in video_paths
    )
    if not normalized:
        raise OPAIValidationError(
            f"At least one {label} video path is required.",
            details={"label": label},
        )
    return normalized


def _normalize_single_video_path(video_path: str | Path, *, label: str) -> Path:
    return _coerce_existing_file_path(video_path, label=label)


def _coerce_existing_file_path(value: str | Path, *, label: str) -> Path:
    path = Path(value).expanduser()
    if not path.exists():
        raise OPAIValidationError(
            f"{label.capitalize()} video does not exist: {path}",
            details={"path": str(path)},
        )
    if not path.is_file():
        raise OPAIValidationError(
            f"{label.capitalize()} path must point to a file: {path}",
            details={"path": str(path)},
        )
    return path


def _normalize_session_name(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise OPAIValidationError("Session name must be a non-empty string.")
    return name.strip()
