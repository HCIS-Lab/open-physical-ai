from pathlib import Path
from typing import Optional

from opai.domain.context import Context

_ACTIVE_CONTEXT = None  # type: Optional[Context]


def init_context(name: str) -> Context:
    session_directory = Path.cwd() / ".opai_sessions" / name
    session_directory.mkdir(parents=True, exist_ok=True)

    global _ACTIVE_CONTEXT
    _ACTIVE_CONTEXT = Context(name=name, session_directory=session_directory)
    return _ACTIVE_CONTEXT


def get_active_context() -> Optional[Context]:
    return _ACTIVE_CONTEXT
