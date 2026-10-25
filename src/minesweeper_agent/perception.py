"""Template-based recognition from the user's own classic Minesweeper skin."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from .geometry import cell_bounds
from .model import Board, COVERED, FLAG, MINE, UNKNOWN

LABELS = {"covered": COVERED, "flag": FLAG, "mine": MINE,
          **{str(n): n for n in range(9)}}
TEMPLATE_SIZE = (32, 32)


def extract_cell(image: np.ndarray, rows: int, cols: int, row: int, col: int) -> np.ndarray:
    x0, y0, x1, y1 = cell_bounds(image.shape[1], image.shape[0], rows, cols, row, col)
    tile = image[y0:y1, x0:x1]
    if tile.size == 0:
        raise ValueError("Grid contains an empty cell; check calibration")
    return cv2.resize(tile, TEMPLATE_SIZE, interpolation=cv2.INTER_AREA)


def save_template(image: np.ndarray, rows: int, cols: int,
                  row: int, col: int, label: str, folder: Path) -> Path:
    if label not in LABELS:
        raise ValueError(f"Unknown label {label!r}")
    destination = folder / label
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / f"{uuid4().hex}.png"
    if not cv2.imwrite(str(target), extract_cell(image, rows, cols, row, col)):
        raise OSError(f"Unable to save template {target}")
    return target


class TemplateRecognizer:
    def __init__(self, folder: Path, threshold: float = 0.045,
                 ambiguity_margin: float = 0.008):
        self.threshold = threshold
        self.ambiguity_margin = ambiguity_margin
        self.templates: dict[str, list[np.ndarray]] = {}
        for label in LABELS:
            samples = []
            for path in sorted((folder / label).glob("*.png")):
                img = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if img is not None:
                    samples.append(cv2.resize(img, TEMPLATE_SIZE,
                                              interpolation=cv2.INTER_AREA))
            if samples:
                self.templates[label] = samples

    def classify(self, tile: np.ndarray) -> tuple[int, float]:
        if not self.templates:
            return UNKNOWN, float("inf")
        resized = cv2.resize(tile, TEMPLATE_SIZE, interpolation=cv2.INTER_AREA)
        arr = resized.astype(np.float32)
        distances = sorted((min(float(np.mean(np.abs(arr - sample.astype(np.float32))) / 255)
                                for sample in examples), label)
                           for label, examples in self.templates.items())
        best_score, label = distances[0]
        if best_score > self.threshold:
            return UNKNOWN, best_score
        if len(distances) >= 2 and distances[1][0] - best_score < self.ambiguity_margin:
            return UNKNOWN, best_score
        return LABELS[label], best_score

    def read(self, image: np.ndarray, rows: int, cols: int) -> tuple[Board, dict[tuple[int, int], float]]:
        cells = []
        scores = {}
        for r in range(rows):
            values = []
            for c in range(cols):
                value, score = self.classify(extract_cell(image, rows, cols, r, c))
                values.append(value)
                scores[(r, c)] = score
            cells.append(tuple(values))
        return Board(tuple(cells)), scores
