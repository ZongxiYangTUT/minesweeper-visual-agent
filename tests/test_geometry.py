import pytest

from minesweeper_agent.geometry import cell_bounds, cell_center
from minesweeper_agent.model import Rect, Settings


def test_centers_with_non_divisible_tile_size():
    rect = Rect(100, 50, 31, 21)
    assert cell_bounds(31, 21, 2, 3, 0, 0) == (0, 0, 10, 10)
    assert cell_bounds(31, 21, 2, 3, 1, 2) == (21, 10, 31, 21)
    assert cell_center(rect, 2, 3, 1, 2) == (126, 65)


def test_settings_reject_invalid_mine_counts():
    with pytest.raises(ValueError):
        Settings(2, 2, Rect(0, 0, 50, 50), total_mines=4)
