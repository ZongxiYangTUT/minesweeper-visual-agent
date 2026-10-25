"""Shared domain types; no platform dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

COVERED = -1
FLAG = -2
UNKNOWN = -3
MINE = -4

Cell = tuple[int, int]


@dataclass(frozen=True)
class Rect:
    left: int
    top: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Board rectangle needs positive width and height")


@dataclass(frozen=True)
class Settings:
    rows: int
    cols: int
    rect: Rect
    total_mines: int | None = None

    def __post_init__(self) -> None:
        if self.rows <= 0 or self.cols <= 0:
            raise ValueError("Board dimensions need to be positive")
        if self.rect.width < self.cols * 8 or self.rect.height < self.rows * 8:
            raise ValueError("Selected rectangle is too small for this board")
        if self.total_mines is not None and not (0 <= self.total_mines < self.rows * self.cols):
            raise ValueError("Invalid total mine count")

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "rows": self.rows, "cols": self.cols,
            "rect": vars(self.rect), "total_mines": self.total_mines,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    @staticmethod
    def load(path: Path) -> "Settings":
        data = json.loads(path.read_text(encoding="utf-8"))
        return Settings(rows=int(data["rows"]), cols=int(data["cols"]),
                        rect=Rect(**data["rect"]), total_mines=data.get("total_mines"))


@dataclass(frozen=True)
class Board:
    cells: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if not self.cells or not self.cells[0]:
            raise ValueError("Board cannot be empty")
        if any(len(row) != len(self.cells[0]) for row in self.cells):
            raise ValueError("Every row must have the same length")
        if any(value not in (COVERED, FLAG, UNKNOWN, MINE, *range(9))
               for row in self.cells for value in row):
            raise ValueError("Unexpected tile value")

    @property
    def rows(self) -> int:
        return len(self.cells)

    @property
    def cols(self) -> int:
        return len(self.cells[0])

    def neighbors(self, row: int, col: int) -> list[Cell]:
        return [(r, c) for r in range(max(0, row - 1), min(self.rows, row + 2))
                for c in range(max(0, col - 1), min(self.cols, col + 2))
                if (r, c) != (row, col)]

    def locations(self, value: int) -> set[Cell]:
        return {(r, c) for r in range(self.rows) for c in range(self.cols)
                if self.cells[r][c] == value}

    def render(self) -> str:
        symbols = {COVERED: "#", FLAG: "F", UNKNOWN: "?", MINE: "X", 0: "."}
        output = ["    " + " ".join(str(c % 10) for c in range(self.cols))]
        for r, row in enumerate(self.cells):
            output.append(f"{r:>2}  " + " ".join(symbols.get(v, str(v)) for v in row))
        return "\n".join(output)


@dataclass(frozen=True)
class Move:
    kind: str  # "reveal" or "flag"
    row: int
    col: int
    reason: str
