from typing import Any, Sequence

from opai.core.exceptions import (
    OPAIContextError,
    OPAIDependencyError,
    OPAIValidationError,
)
from opai.domain.calibration import CalibrationResult
from opai.domain.context import Context
from opai.infrastructure.context_store import get_active_context, init_context


def init(name: str) -> Context:
    if not name.strip():
        raise OPAIValidationError("Session name must be a non-empty string.")
    return init_context(name)


def get_context() -> Context:
    ctx = get_active_context()
    if ctx is None:
        raise OPAIContextError(
            "No active context found. Call opai.init(name) before notebook-facing operations."
        )
    return ctx


def calibrate(
    frames: Sequence[Any],
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


def main() -> None:
    print("Use opai.init(name) and opai.calibrate(...) from Python.")
