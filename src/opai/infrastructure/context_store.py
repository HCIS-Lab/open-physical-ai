from __future__ import annotations

import importlib.resources
import shutil
from pathlib import Path

from opai.core.exceptions import OPAIPackageResourceError, OPAIWorkflowError
from opai.domain.context import Context
from opai.domain.session import SessionManifest
from opai.infrastructure.persistence import (
    load_session_manifest,
    write_session_manifest,
)

_ACTIVE_CONTEXT: Context | None = None
SESSION_ROOT_DIRNAME = "sessions"
SESSION_MANIFEST_FILENAME = "session.json"
DEFAULT_SLAM_SETTING_FILENAME = "gopro13_fisheye_ratio_4-3_2-7k.yaml"


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
    slam_setting_full_path = get_slam_config_dir() / DEFAULT_SLAM_SETTING_FILENAME
    shutil.copy(slam_setting_full_path, session_directory / "slam_settings.yaml")
    return _ACTIVE_CONTEXT


def get_slam_config_dir() -> Path:
    config_dir = importlib.resources.files("opai.configs")
    if not isinstance(config_dir, Path):
        raise OPAIPackageResourceError("Failed to load slam configs directory")
    return config_dir / "slam"


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


def get_mapping_video(ctx: Context) -> Path | None:
    manifest = load_manifest_for_context(ctx)
    if manifest.mapping is None:
        return None
    absolute_path = (ctx.session_directory / manifest.mapping.stored_path).resolve()
    if absolute_path.exists() and absolute_path.is_file():
        return absolute_path

    raise OPAIWorkflowError(
        f"Session mapping video is missing from stored session artifacts: {absolute_path}",
        details={
            "session_name": ctx.name,
            "asset_kind": "mapping",
            "stored_path": manifest.mapping.stored_path,
            "resolved_path": str(absolute_path),
        },
    )


def get_demo_videos(ctx: Context) -> list[Path]:
    manifest = load_manifest_for_context(ctx)
    demo_video_paths: list[Path] = []
    for demo in manifest.demos:
        absolute_path = (ctx.session_directory / demo.stored_path).resolve()
        if absolute_path.exists() and absolute_path.is_file():
            demo_video_paths.append(absolute_path)
            continue

        raise OPAIWorkflowError(
            f"Session demo video is missing from stored session artifacts: {absolute_path}",
            details={
                "session_name": ctx.name,
                "asset_kind": "demo",
                "stored_path": demo.stored_path,
                "resolved_path": str(absolute_path),
                "asset_id": demo.demo_id,
            },
        )
    return demo_video_paths


def get_session_directory(name: str) -> Path:
    return session_root() / name


def session_root() -> Path:
    return Path.cwd() / SESSION_ROOT_DIRNAME


def _ensure_session_structure(session_directory: Path) -> None:
    (session_directory / "captures" / "demos").mkdir(parents=True, exist_ok=True)
    (session_directory / "captures" / "mapping").mkdir(parents=True, exist_ok=True)
    (session_directory / "gopro_thumbnails").mkdir(parents=True, exist_ok=True)


def _require_manifest_path(ctx: Context) -> Path:
    if ctx.manifest_path is not None:
        return ctx.manifest_path
    return ctx.session_directory / SESSION_MANIFEST_FILENAME
