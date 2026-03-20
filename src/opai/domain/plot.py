from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

BASE_SUBPLOT_WIDTH = 4.0
BASE_SUBPLOT_HEIGHT = 3.0
MAX_FIGURE_WIDTH = 16.0
MAX_FIGURE_HEIGHT = 12.0
MAX_CANVAS_WIDTH = 1600
MAX_CANVAS_HEIGHT = 1200


@dataclass(frozen=True)
class PlotGrid:
    item_count: int
    nrows: int
    ncols: int


def get_plot_grid(
    item_count: int,
    nrows: int | None = None,
    ncols: int | None = None,
) -> PlotGrid:
    if item_count <= 0:
        raise ValueError("item_count must be positive.")
    if nrows is not None and nrows <= 0:
        raise ValueError("nrows must be positive when provided.")
    if ncols is not None and ncols <= 0:
        raise ValueError("ncols must be positive when provided.")

    if nrows is not None:
        resolved_rows = nrows
        resolved_cols = ncols or int(math.ceil(item_count / nrows))
    elif ncols is not None:
        resolved_cols = ncols
        resolved_rows = int(math.ceil(item_count / ncols))
    else:
        resolved_rows = int(math.sqrt(item_count))
        resolved_cols = int(math.ceil(item_count / resolved_rows))

    return PlotGrid(
        item_count=item_count,
        nrows=resolved_rows,
        ncols=resolved_cols,
    )


def plot_frames(
    frames: Sequence[np.ndarray],
    nrows: int | None = None,
    ncols: int | None = None,
    *,
    frames_are_bgr: bool = True,
) -> None:
    from matplotlib import pyplot

    grid = get_plot_grid(len(frames), nrows=nrows, ncols=ncols)
    fig, axes = pyplot.subplots(
        grid.nrows,
        grid.ncols,
        figsize=_get_figsize(grid),
    )
    flat_axes = np.atleast_1d(axes).reshape(-1)

    for axis, frame in zip(flat_axes, frames):
        image = _prepare_frame(frame, grid)
        if image.ndim != 2 and frames_are_bgr:
            image = image[..., ::-1]
        axis.imshow(image)
        axis.set_axis_off()

    for axis in flat_axes[len(frames) :]:
        axis.set_axis_off()

    fig.tight_layout()
    pyplot.show()
    pyplot.close(fig)


def _get_figsize(grid: PlotGrid) -> tuple[float, float]:
    width = BASE_SUBPLOT_WIDTH * grid.ncols
    height = BASE_SUBPLOT_HEIGHT * grid.nrows
    scale = min(1.0, MAX_FIGURE_WIDTH / width, MAX_FIGURE_HEIGHT / height)
    return width * scale, height * scale


def _prepare_frame(frame: np.ndarray, grid: PlotGrid) -> np.ndarray:
    max_height = max(1, MAX_CANVAS_HEIGHT // grid.nrows)
    max_width = max(1, MAX_CANVAS_WIDTH // grid.ncols)
    height, width = frame.shape[:2]
    stride = max(1, math.ceil(height / max_height), math.ceil(width / max_width))
    return frame[::stride, ::stride]
