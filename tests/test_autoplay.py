from argparse import Namespace
import json

import pytest

from minesweeper_agent.autoplay import choose_move, play, validate_transition
from minesweeper_agent.model import Board, COVERED as C, FLAG, MINE, UNKNOWN, Move, Rect, Settings
from test_classic import read_fixture


def test_no_guess_without_explicit_option():
    board = Board(((1,1),(C,C)))
    settings = Settings(2,2,Rect(0,0,32,32),1)
    assert choose_move(board, settings, False) == (None, None, "stalled")
    move, probability, status = choose_move(board, settings, True)
    assert (status, probability, move.kind) == ("guess", .5, "reveal")


@pytest.mark.parametrize("value,status", [(UNKNOWN,"unknown"),(MINE,"lost")])
def test_terminal_pixels_never_produce_input(value, status):
    assert choose_move(Board(((value,C),)), Settings(1,2,Rect(0,0,32,16),1), True)[0::2] == (None,status)


def test_unchanged_target_stops_and_old_clues_must_not_change():
    before = Board(((1,C),))
    move = Move("reveal",0,1,"test")
    with pytest.raises(ValueError, match="did not reveal"):
        validate_transition(before,before,move)
    with pytest.raises(ValueError, match="changed unexpectedly"):
        validate_transition(before,Board(((0,0),)),move)


class ReplayGame:
    """The real screenshot after opening is replayed; no desktop input occurs."""
    def __init__(self, change_before_click=False, unchanged_after_click=False):
        self.fresh = read_fixture("classic-125-fresh.png")
        self.clues = read_fixture("classic-125-clues.png")
        self.calls = 0
        self.clicks = []
        self.change_before_click = change_before_click
        self.unchanged_after_click = unchanged_after_click

    def open(self): pass
    def new_game(self, difficulty): pass

    def capture(self):
        self.calls += 1
        changed = self.clicks and not self.unchanged_after_click
        if self.change_before_click and self.calls >= 3:
            changed = True
        image = self.clues if changed else self.fresh
        return image.copy(), Rect(0,0,image.shape[1],image.shape[0])

    def click(self, move, settings):
        self.clicks.append(move)


def args(tmp_path):
    return Namespace(data_dir=tmp_path,games=1,difficulty="beginner",mines=None,
                     max_steps=1,allow_guess=False,stop_on_win=True,delay=.1)


def test_loop_captures_verifies_and_saves_replay(tmp_path, monkeypatch):
    monkeypatch.setattr("minesweeper_agent.autoplay.time.sleep", lambda _:None)
    game = ReplayGame()
    result = play(args(tmp_path), game)
    assert result["status"] == "step_limit"
    assert len(game.clicks) == 1
    assert result["games"][0]["clicks"] == 1
    assert result["games"][0]["difficulty"] == "beginner"
    assert result["games"][0]["guesses"] == 0
    assert (result["games"][0]["rows"],result["games"][0]["cols"],result["games"][0]["mines"]) == (9,9,10)
    from pathlib import Path
    folder = Path(result["log"])
    assert (folder/"report.html").is_file()
    events = [json.loads(line) for line in (folder/"events.jsonl").read_text().splitlines()]
    assert [e["event"] for e in events] == ["start","observe","click","observe"]
    assert events[1]["solve_ms"] >= 0


def test_changed_frame_prevents_planned_click(tmp_path, monkeypatch):
    monkeypatch.setattr("minesweeper_agent.autoplay.time.sleep", lambda _:None)
    game = ReplayGame(change_before_click=True)
    result = play(args(tmp_path),game)
    assert result["status"] == "error"
    assert result["games"][0]["clicks"] == 0
    assert result["games"][0]["status"] == "error"
    assert not game.clicks


def test_no_effect_click_stops_without_repeating(tmp_path, monkeypatch):
    monkeypatch.setattr("minesweeper_agent.autoplay.time.sleep", lambda _:None)
    game = ReplayGame(unchanged_after_click=True)
    assert play(args(tmp_path),game)["status"] == "error"
    assert len(game.clicks) == 1


def test_equal_risk_guesses_prefer_a_chance_of_opening_a_blank_region():
    # Bottom corners and interior share the same global probability. Corners
    # have fewer unknown neighbours, so they win the heuristic tie-break.
    board = Board(((1,C,C,C,C),(C,C,C,C,C),(C,C,C,C,C),(C,C,C,C,C),(C,C,C,C,C)))
    move,p,status = choose_move(board,Settings(5,5,Rect(0,0,100,100),3),True)
    assert status == "guess" and p < 1/3
    assert (move.row,move.col) in {(0,4),(4,0),(4,4)}
