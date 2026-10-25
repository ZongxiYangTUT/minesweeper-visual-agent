"""One mouse action per call; orchestration and authorization belong to the caller."""

from __future__ import annotations

from .geometry import cell_center
from .model import Move, Settings


def execute(move: Move, settings: Settings) -> tuple[int, int]:
    """Perform one click in an approved single-step or explicitly started auto run."""
    import pyautogui

    x, y = cell_center(settings.rect, settings.rows, settings.cols,
                       move.row, move.col)
    width, height = pyautogui.size()
    if not (0 <= x < width and 0 <= y < height):
        raise ValueError("Click target must be on the Windows primary monitor")
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.1
    if move.kind not in ("reveal", "flag"):
        raise ValueError(f"Unsupported move kind: {move.kind}")
    pyautogui.click(x=x, y=y, button="left" if move.kind == "reveal" else "right")
    return x, y
