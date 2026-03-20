from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

BASE_SUBPLOT_WIDTH = 8.0
BASE_SUBPLOT_HEIGHT = 3.0
MAX_FIGURE_WIDTH = 16.0
MAX_FIGURE_HEIGHT = 12.0
MIN_SINGLE_ROW_FIGURE_HEIGHT = 4.5
MAX_CANVAS_WIDTH = 4072
MAX_CANVAS_HEIGHT = 2304


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
        if frame.ndim != 2 and frames_are_bgr:
            frame = frame[..., ::-1]
        axis.imshow(frame)
        axis.set_axis_off()

    for axis in flat_axes[len(frames) :]:
        axis.set_axis_off()

    fig.tight_layout()
    pyplot.show()
    pyplot.close(fig)


def _get_figsize(
    grid: PlotGrid,
    *,
    imsize: float = 3.0,
    add_vert: float = 0.6,
) -> tuple[float, float]:
    return grid.ncols * imsize, grid.nrows * imsize + add_vert
