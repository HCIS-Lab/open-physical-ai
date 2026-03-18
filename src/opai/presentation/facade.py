from __future__ import annotations

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
    Console, Tree = _load_rich_components()
    from opai.application.session import describe_sessions

    catalog = describe_sessions()
    console = Console()
    tree = Tree(
        f"[bold]{catalog.root_dirname}[/] [dim]{catalog.root_path}[/]",
        guide_style="dim",
    )
    session_names = [session.name for session in catalog.sessions]
    if not catalog.sessions:
        tree.add("[yellow]No sessions found[/]")
        console.print(tree)
        return session_names

    for session in catalog.sessions:
        tags: list[str] = []
        if session.is_active:
            tags.append("current")
        tags.append(f"demos={session.demo_count}")
        tags.append(f"mapping={'yes' if session.has_mapping else 'no'}")
        tags.append(f"files={session.file_count}")
        tree.add(f"[bold cyan]{session.name}[/] [dim]({', '.join(tags)})[/]")
    console.print(tree)
    return session_names


def browse_session(name: str) -> list[str]:
    normalized_name = _normalize_session_name(name)
    Console, Tree = _load_rich_components()
    from opai.application.session import describe_session

    view = describe_session(normalized_name)
    console = Console()
    tree = Tree(
        f"[bold]{view.root_dirname}[/] [dim]{view.root_path}[/]",
        guide_style="dim",
    )
    session_branch = tree.add(
        "[bold magenta]"
        f"{view.session_name}"
        "[/] [dim]"
        f"(path={view.session_path.name}, demos={view.demo_count}, "
        f"mapping={'yes' if view.has_mapping else 'no'}, files={view.file_count})"
        "[/]"
    )
    session_branch.add(f"[dim]path:[/] [cyan]{view.session_path}[/]")
    _append_tree_nodes(session_branch, view.tree_payload)
    console.print(tree)
    return list(view.file_paths)


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
        is_directory = isinstance(child, dict) and bool(child)
        label = f"[bold blue]{name}/[/]" if is_directory else f"[green]{name}[/]"
        branch = tree.add(label)
        if is_directory:
            _append_tree_nodes(branch, child)


def _load_rich_components():
    try:
        from rich.console import Console
        from rich.tree import Tree
    except ModuleNotFoundError as exc:
        raise OPAIDependencyError(
            "Session browsing requires the 'rich' package. Install project dependencies before calling opai.list_sessions(...) or opai.browse_session(...)."
        ) from exc
    return Console, Tree
