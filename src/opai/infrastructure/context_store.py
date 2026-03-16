from __future__ import annotations

from pathlib import Path

from opai.domain.context import Context
from opai.domain.session import SessionManifest
from opai.infrastructure.persistence import (
    load_session_manifest,
    write_session_manifest,
)

_ACTIVE_CONTEXT: Context | None = None
SESSION_ROOT_DIRNAME = ".opai_sessions"
SESSION_MANIFEST_FILENAME = "session.json"


def init_context(name: str) -> Context:
    session_directory = session_root() / name
    manifest_path = session_directory / SESSION_MANIFEST_FILENAME
    _ensure_session_structure(session_directory)
    manifest = load_session_manifest(manifest_path, session_name=name)
    write_session_manifest(manifest_path, manifest)

    global _ACTIVE_CONTEXT
    _ACTIVE_CONTEXT = Context(
        name=manifest.session_name,
        session_directory=session_directory,
        manifest_path=manifest_path,
    )
    return _ACTIVE_CONTEXT


def get_active_context() -> Context | None:
    return _ACTIVE_CONTEXT


def list_session_names() -> list[str]:
    root = session_root()
    if not root.exists():
        return []
    return sorted(
        path.name
        for path in root.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )


def load_manifest_for_context(ctx: Context) -> SessionManifest:
    manifest_path = _require_manifest_path(ctx)
    return load_session_manifest(manifest_path, session_name=ctx.name)


def persist_manifest_for_context(ctx: Context, manifest: SessionManifest) -> Path:
    manifest_path = _require_manifest_path(ctx)
    return write_session_manifest(manifest_path, manifest)


def get_session_directory(name: str) -> Path:
    return session_root() / name


def session_root() -> Path:
    return Path.cwd() / SESSION_ROOT_DIRNAME


def _ensure_session_structure(session_directory: Path) -> None:
    (session_directory / "captures" / "demos").mkdir(parents=True, exist_ok=True)
    (session_directory / "captures" / "mapping").mkdir(parents=True, exist_ok=True)


def _require_manifest_path(ctx: Context) -> Path:
    if ctx.manifest_path is not None:
        return ctx.manifest_path
    return ctx.session_directory / SESSION_MANIFEST_FILENAME
