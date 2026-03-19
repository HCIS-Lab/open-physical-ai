from __future__ import annotations

import html
import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from opai.core.exceptions import (
    OPAIContextError,
    OPAIDependencyError,
    OPAIValidationError,
)
from opai.domain.calibration import CalibrationResult
from opai.domain.context import Context
from opai.domain.gopro import GPThumbnail
from opai.domain.session import DemoAsset, MappingAsset
from opai.infrastructure.context_store import get_active_context, init_context

_SESSION_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def init(name: str) -> Context:
    normalized_name = _normalize_session_name(name)
    return init_context(normalized_name)


def get_context() -> Context:
    ctx = get_active_context()
    if ctx is None:
        raise OPAIContextError(
            "No active context found. Call opai.init(name) before notebook-facing operations."
        )
    return ctx


def calibrate(
    frames: Sequence[np.ndarray],
    row_count: int,
    col_count: int,
    square_length: float,
    marker_length: float,
    dictionary: str,
) -> CalibrationResult:
    ctx = get_context()
    try:
        from opai.application.calibration import calibrate as calibrate_with_context
    except ModuleNotFoundError as exc:
        raise OPAIDependencyError(
            "Calibration dependencies are unavailable. Install the project's "
            "OpenCV calibration stack before calling opai.calibrate(...)."
        ) from exc

    return calibrate_with_context(
        ctx=ctx,
        frames=frames,
        row_count=row_count,
        col_count=col_count,
        square_length=square_length,
        marker_length=marker_length,
        dictionary=dictionary,
    )


def add_demos(video_paths: Sequence[str | Path]) -> tuple[DemoAsset, ...]:
    ctx = get_context()
    from opai.application.session import add_demos as add_demos_with_context

    return add_demos_with_context(ctx, video_paths)


def add_mapping(video_path: str | Path) -> MappingAsset:
    ctx = get_context()
    from opai.application.session import add_mapping as add_mapping_with_context

    return add_mapping_with_context(ctx, video_path)


def list_sessions() -> list[str]:
    from opai.application.session import list_sessions as list_available_sessions

    return list_available_sessions()


def browse_session(name: str) -> list[str]:
    normalized_name = _normalize_session_name(name)
    try:
        from rich.console import Console
        from rich.tree import Tree
    except ModuleNotFoundError as exc:
        raise OPAIDependencyError(
            "Session browsing requires the 'rich' package. Install project dependencies before calling opai.browse_session(...)."
        ) from exc

    from opai.application.session import browse_session as browse_named_session

    file_paths, tree_payload = browse_named_session(normalized_name)
    tree = Tree(normalized_name)
    _append_tree_nodes(tree, tree_payload)
    Console().print(tree)
    return file_paths


def register_gopro(serial_number: str, download_thumbnails: bool = True) -> None:
    ctx = get_context()
    try:
        from opai.application.gopro import register_gopro as register_gopro_with_context
    except ModuleNotFoundError as exc:
        raise OPAIDependencyError(
            "GoPro registration requires the 'zeroconf' package. Install project dependencies before calling opai.register_gopro(...)."
        ) from exc
    register_gopro_with_context(
        ctx, serial_number, download_thumbnails=download_thumbnails
    )


def main() -> None:
    print(
        "Use opai.init(name), opai.add_demos(...), opai.add_mapping(...), and opai.calibrate(...) from Python."
    )


def _normalize_session_name(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise OPAIValidationError("Session name must be a non-empty string.")
    normalized_name = name.strip()
    if not _SESSION_NAME_PATTERN.fullmatch(normalized_name):
        raise OPAIValidationError(
            "Session name may only contain letters, numbers, '.', '_' and '-'.",
            details={"session_name": normalized_name},
        )
    return normalized_name


def _append_tree_nodes(tree, payload: dict[str, dict]) -> None:
    for name, child in payload.items():
        branch = tree.add(name)
        if isinstance(child, dict):
            _append_tree_nodes(branch, child)
