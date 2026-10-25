from pathlib import Path

import cv2
import numpy as np
import pytest

from minesweeper_agent.classic import ClassicRecognizer, classify_classic, detect_grid
from minesweeper_agent.model import Board, COVERED as C, FLAG, MINE, UNKNOWN, Rect

FIXTURES = Path(__file__).parent / "fixtures"


def read_fixture(name):
    image = cv2.imread(str(FIXTURES / name))
    assert image is not None
    return image


def fixture_settings():
    initial = read_fixture("classic-125-fresh.png")
    return detect_grid(initial, Rect(0, 0, initial.shape[1], initial.shape[0]))


def test_actual_xp_grid_at_125_percent_dpi():
    s = fixture_settings()
    assert (s.rows, s.cols, s.total_mines) == (9, 9, 10)
    assert s.rect == Rect(15, 70, 180, 180)
    image = read_fixture("classic-125-fresh.png")
    r = s.rect
    board, _ = ClassicRecognizer().read(image[r.top:r.top+r.height, r.left:r.left+r.width], 9, 9)
    assert len(board.locations(C)) == 81


def test_actual_clues_including_dense_red_three_are_not_mines():
    expected = Board((
        (0,0,1,C,1,0,1,C,C), (0,0,1,1,1,0,1,2,C), (1,1,0,0,0,1,1,3,C),
        (C,1,0,0,0,1,C,C,C), (C,2,0,0,0,1,1,2,1), (C,2,1,0,0,0,0,0,0),
        (C,C,1,0,0,0,0,0,0), (C,C,2,1,1,1,1,1,0), (C,C,C,C,C,C,C,1,0),
    ))
    r = fixture_settings().rect
    image = read_fixture("classic-125-clues.png")
    board, _ = ClassicRecognizer().read(image[r.top:r.top+r.height, r.left:r.left+r.width], 9, 9)
    assert board == expected


def test_actual_winning_board_has_71_open_tiles_and_10_flags():
    image = read_fixture("classic-125-won.png")
    r = fixture_settings().rect
    board, _ = ClassicRecognizer().read(image[r.top:r.top+r.height, r.left:r.left+r.width], 9, 9)
    assert len(board.locations(FLAG)) == 10
    assert sum(0 <= value <= 8 for row in board.cells for value in row) == 71
    assert not board.locations(UNKNOWN) | board.locations(MINE) | board.locations(C)


def test_actual_losing_board_keeps_red_digits_distinct_from_exploded_mine():
    image = read_fixture("classic-125-lost.png")
    r = fixture_settings().rect
    board, _ = ClassicRecognizer().read(image[r.top:r.top+r.height, r.left:r.left+r.width], 9, 9)
    assert board.locations(MINE) == {(7, 7)}
    assert len(board.locations(FLAG)) == 9
    assert board.cells[6][7] == 3
    assert board.cells[8][7] == 3
    assert not board.locations(UNKNOWN)


@pytest.mark.parametrize("colour", [(128,128,128), (255,128,0), (255,255,255), (0,0,0)])
def test_foreign_flat_regions_are_not_blank_cells(colour):
    tile = np.full((20,20,3), colour, dtype=np.uint8)
    assert classify_classic(tile) != 0


def test_exploded_mine_and_white_glint_mine():
    tile = np.full((20,20,3), (0,0,255), dtype=np.uint8)
    tile[5:15,5:15] = 0
    assert classify_classic(tile) == MINE
    tile[:] = 192
    tile[5:15,5:15] = 0
    tile[6:8,6:8] = 255
    assert classify_classic(tile) == MINE


def test_missing_grid_is_rejected():
    with pytest.raises(ValueError, match="Cannot locate"):
        detect_grid(np.full((200,200,3), 192, np.uint8), Rect(0,0,200,200))


@pytest.mark.parametrize("difficulty,rows,cols,mines", [
    ("intermediate",16,16,40), ("expert",16,30,99),
])
def test_actual_larger_boards_detect_every_row_and_recognize_the_win(difficulty,rows,cols,mines):
    fresh = read_fixture(f"classic-125-{difficulty}-fresh.png")
    settings = detect_grid(fresh, Rect(0,0,fresh.shape[1],fresh.shape[0]))
    assert (settings.rows,settings.cols,settings.total_mines) == (rows,cols,mines)
    r = settings.rect
    recognizer = ClassicRecognizer()
    board, _ = recognizer.read(fresh[r.top:r.top+r.height,r.left:r.left+r.width],rows,cols)
    assert len(board.locations(C)) == rows*cols
    won = read_fixture(f"classic-125-{difficulty}-won.png")
    board, _ = recognizer.read(won[r.top:r.top+r.height,r.left:r.left+r.width],rows,cols)
    assert len(board.locations(FLAG)) == mines
    assert sum(0<=value<=8 for row in board.cells for value in row) == rows*cols-mines
    from minesweeper_agent.solver import analyze
    assert not analyze(board,mines).errors
