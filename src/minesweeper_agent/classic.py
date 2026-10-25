"""Screen-only recognition of the coloured Windows XP Minesweeper skin.

No executable resources, game memory, or hidden board data are consulted.
The raised bevel distinguishes closed cells; revealed digits have unique colours.
"""
from __future__ import annotations

from collections import Counter

import cv2
import numpy as np

from .geometry import cell_bounds
from .model import Board, COVERED, FLAG, MINE, UNKNOWN, Rect, Settings


def detect_grid(image: np.ndarray, origin: Rect, total_mines: int | None = None) -> Settings:
    """Locate a complete fresh grid from repeating white upper/left bevels."""
    white = (image.min(axis=2) > 235).astype(np.uint8)
    _, _, stats, _ = cv2.connectedComponentsWithStats(white, 8)
    candidates = [(int(x), int(y), int(w), int(h)) for x, y, w, h, area in stats[1:]
                  if 10 <= w <= 64 and 10 <= h <= 64 and abs(w - h) <= 3
                  and area < w * h * .5]
    groups: dict[int, list[tuple[int, int, int, int]]] = {}
    for box in candidates:
        groups.setdefault(box[1], []).append(box)
    proposals = []
    for y, boxes in groups.items():
        xs = sorted(x for x, _, _, _ in boxes)
        if len(xs) < 8:
            continue
        step, count = Counter(b - a for a, b in zip(xs, xs[1:])).most_common(1)[0]
        if not 12 <= step <= 64 or count < 7:
            continue
        # Fractional DPI interpolation shifts the thresholded white edge by one
        # pixel across a large board. Fit the lattice, not exact bounding boxes.
        if any(abs(x - (xs[0] + col * step)) > 1 for col, x in enumerate(xs)):
            continue
        rows = 0
        while True:
            matches = [boxes for row_y, boxes in groups.items() if abs(row_y - (y + rows * step)) <= 1]
            if len(matches) != 1:
                break
            row_xs = sorted(x for x, _, _, _ in matches[0])
            if len(row_xs) != len(xs) or any(abs(x - (xs[0] + col * step)) > 1
                                           for col, x in enumerate(row_xs)):
                break
            rows += 1
        if rows >= 8:
            proposals.append((rows * len(xs), xs[0], y, step, rows, len(xs)))
    if not proposals:
        raise ValueError("Cannot locate a fresh classic grid. Keep the entire game visible and in colour.")
    _, x, y, step, rows, cols = max(proposals)
    if total_mines is None:
        total_mines = {(9, 9): 10, (8, 8): 10, (16, 16): 40, (16, 30): 99}.get((rows, cols))
    if total_mines is None:
        raise ValueError(f"Detected {rows}x{cols} grid; custom grids need an explicit mine count")
    return Settings(rows, cols, Rect(origin.left + x, origin.top + y,
                                    cols * step, rows * step), total_mines)


def classify_classic(tile: np.ndarray) -> int:
    """Reject unfamiliar pixels instead of treating every grey rectangle as blank."""
    h, w = tile.shape[:2]
    if min(h, w) < 12:
        return UNKNOWN
    # Interior excludes bevel/grid lines but includes the whole number glyph.
    inset = max(2, round(min(h, w) * .15))
    inner = tile[inset:-inset, inset:-inset].astype(np.int16)
    b, g, r = (inner[:, :, i] for i in range(3))
    bright = tile.min(axis=2) > 230
    edge = max(2, round(min(h, w) * .16))
    raised = (bright[:edge, :].mean() > .30 and bright[:, :edge].mean() > .30)
    red = (r > 170) & (g < 100) & (b < 100)
    black = (inner.max(axis=2) < 65)
    if raised:
        if red.sum() >= 3 and black.sum() >= 3:
            return FLAG
        # Question marks are deliberately unknown, never considered closed.
        if black.sum() > 2:
            return UNKNOWN
        neutral = (inner.max(axis=2) - inner.min(axis=2) < 15)
        if neutral.mean() > .96 and 130 < float(inner.mean()) < 220:
            return COVERED
        return UNKNOWN
    full = tile.astype(np.int16)
    red_background = (full[:, :, 2] > 170) & (full[:, :, 1] < 100) & (full[:, :, 0] < 100)
    if red_background.mean() > .42:
        return MINE  # clicked mine: red cell background
    colour_masks = {
        1: (b > 170) & (g < 110) & (r < 110),
        2: (g > 65) & (g < 180) & (b < 65) & (r < 65),
        3: red,
        4: (b > 65) & (b < 170) & (g < 65) & (r < 65),
        5: (r > 65) & (r < 170) & (b < 65) & (g < 65),
        6: (b > 65) & (g > 65) & (b < 175) & (g < 175) & (r < 65),
    }
    hits = {n: int(mask.sum()) for n, mask in colour_masks.items() if mask.sum() >= 3}
    if len(hits) == 1:
        return next(iter(hits))
    if hits:
        return UNKNOWN
    if black.sum() >= 3:
        # A mine is broad in both dimensions and has a white glint; 7 does not.
        if (inner.min(axis=2) > 235).any():
            return MINE
        # Seven's ink lives in a narrow glyph; don't accept a broad black blob.
        if black.mean() < .38:
            return 7
        return MINE
    neutral = (inner.max(axis=2) - inner.min(axis=2) < 15)
    if neutral.mean() < .96:
        return UNKNOWN
    gray_ink = (inner.mean(axis=2) > 75) & (inner.mean(axis=2) < 155)
    if gray_ink.sum() >= 5 and (inner.mean(axis=2) > 165).mean() > .4:
        return 8
    if np.percentile(inner, 95) - np.percentile(inner, 5) < 28 and 150 < inner.mean() < 220:
        return 0
    return UNKNOWN


class ClassicRecognizer:
    def read(self, image: np.ndarray, rows: int, cols: int) -> tuple[Board, dict]:
        cells = []
        for row in range(rows):
            values = []
            for col in range(cols):
                x0, y0, x1, y1 = cell_bounds(image.shape[1], image.shape[0], rows, cols, row, col)
                values.append(classify_classic(image[y0:y1, x0:x1]))
            cells.append(tuple(values))
        return Board(tuple(cells)), {}
