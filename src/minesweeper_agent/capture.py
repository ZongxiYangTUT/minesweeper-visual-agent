"""Windows desktop capture; never reads the game's process memory."""

from __future__ import annotations

import ctypes
import sys

import numpy as np

from .model import Rect


def configure_dpi_awareness() -> None:
    """Make screenshot and mouse coordinates agree on scaled Windows desktops."""
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except (AttributeError, OSError):
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except (AttributeError, OSError):
            pass


def screenshot(rect: Rect | None = None) -> tuple[np.ndarray, Rect]:
    """Return a BGR image and its desktop-coordinate rectangle."""
    import cv2
    from mss import mss

    with mss() as sct:
        if rect is None:
            bounds = sct.monitors[0]  # virtual desktop, including secondary monitors
        else:
            bounds = {"left": rect.left, "top": rect.top,
                      "width": rect.width, "height": rect.height}
        shot = sct.grab(bounds)
        image = cv2.cvtColor(np.asarray(shot), cv2.COLOR_BGRA2BGR)
        return image, Rect(int(bounds["left"]), int(bounds["top"]),
                           int(bounds["width"]), int(bounds["height"]))
