from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from opai.core.exceptions import OPAIValidationError
from opai.domain.context import Context
from opai.domain.session import DemoAsset, SessionManifest
from opai.infrastructure.context_store import (
    SESSION_MANIFEST_FILENAME,
    SESSION_ROOT_DIRNAME,
    get_active_context,
    get_session_directory,
    list_session_names,
    load_manifest_for_context,
    persist_manifest_for_context,
    session_root,
)
from opai.infrastructure.persistence import (
    build_file_tree,
    copy_demo_assets,
    copy_mapping_asset,
    list_relative_paths,
    load_session_manifest,
)


@dataclass(frozen=True)
class SessionSummary:
    name: str
    demo_count: int
    has_mapping: bool
    file_count: int
    is_active: bool


@dataclass(frozen=True)
class SessionCatalog:
    root_dirname: str
    root_path: Path
    sessions: tuple[SessionSummary, ...]


@dataclass(frozen=True)
class SessionBrowseView:
    root_dirname: str
    root_path: Path
    session_name: str
    session_path: Path
    demo_count: int
    has_mapping: bool
    file_count: int
    file_paths: tuple[str, ...]
    tree_payload: dict[str, dict]


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
    return [session.name for session in describe_sessions().sessions]


def browse_session(name: str) -> tuple[list[str], dict[str, dict]]:
    view = describe_session(name)
    return list(view.file_paths), view.tree_payload


def describe_sessions() -> SessionCatalog:
    active_context = get_active_context()
    active_name = active_context.name if active_context is not None else None
    summaries = tuple(
        _build_session_summary(session_name, active_name=active_name)
        for session_name in list_session_names()
    )
    return SessionCatalog(
        root_dirname=SESSION_ROOT_DIRNAME,
        root_path=session_root(),
        sessions=summaries,
    )


def describe_session(name: str) -> SessionBrowseView:
    session_name = _normalize_session_name(name)
    session_directory = get_session_directory(session_name)
    if not session_directory.exists():
        raise OPAIValidationError(
            f"Session '{session_name}' does not exist.",
            details={"session_name": session_name},
        )
    manifest = load_session_manifest(
        session_directory / SESSION_MANIFEST_FILENAME,
        session_name=session_name,
    )
    file_paths = tuple(list_relative_paths(session_directory))
    return SessionBrowseView(
        root_dirname=SESSION_ROOT_DIRNAME,
        root_path=session_root(),
        session_name=session_name,
        session_path=session_directory,
        demo_count=len(manifest.demos),
        has_mapping=manifest.mapping is not None,
        file_count=len(file_paths),
        file_paths=file_paths,
        tree_payload=build_file_tree(session_directory),
    )


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


def _build_session_summary(
    session_name: str,
    *,
    active_name: str | None,
) -> SessionSummary:
    session_directory = get_session_directory(session_name)
    manifest = load_session_manifest(
        session_directory / SESSION_MANIFEST_FILENAME,
        session_name=session_name,
    )
    return SessionSummary(
        name=session_name,
        demo_count=len(manifest.demos),
        has_mapping=manifest.mapping is not None,
        file_count=len(list_relative_paths(session_directory)),
        is_active=session_name == active_name,
    )
