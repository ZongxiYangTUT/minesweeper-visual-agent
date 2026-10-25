"""Pixel/grid coordinate conversion shared by perception and actuation."""

from __future__ import annotations

from .model import Rect


def cell_bounds(width: int, height: int, rows: int, cols: int,
                row: int, col: int) -> tuple[int, int, int, int]:
    if not (0 <= row < rows and 0 <= col < cols):
        raise IndexError((row, col))
    # Round each edge, avoiding cumulative errors for non-integral cell sizes.
    x0 = round(col * width / cols)
    x1 = round((col + 1) * width / cols)
    y0 = round(row * height / rows)
    y1 = round((row + 1) * height / rows)
    return x0, y0, x1, y1


def cell_center(rect: Rect, rows: int, cols: int, row: int, col: int) -> tuple[int, int]:
    x0, y0, x1, y1 = cell_bounds(rect.width, rect.height, rows, cols, row, col)
    return rect.left + (x0 + x1) // 2, rect.top + (y0 + y1) // 2
