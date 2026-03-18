from __future__ import annotations

import math
from dataclasses import dataclass


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
