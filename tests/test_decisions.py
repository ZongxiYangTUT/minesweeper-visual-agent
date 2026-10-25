"""Independent game-tree oracle and regression checks for the live policy."""

from functools import lru_cache
from itertools import combinations
from types import SimpleNamespace

import pytest

from minesweeper_agent.autoplay import validate_transition
from minesweeper_agent.controller import execute
from minesweeper_agent.endgame import exact_endgame
from minesweeper_agent.evaluation import Simulation, summarize
from minesweeper_agent.lookahead import progress_choice
from minesweeper_agent.model import Board, COVERED as C, FLAG, Move, Rect, Settings
from minesweeper_agent.policy import _safe_action, choose_move
from minesweeper_agent.solver import Analysis, analyze


def visible_layouts(board, total_mines):
    flags = board.locations(FLAG)
    return tuple(
        frozenset(selected) | flags
        for selected in combinations(sorted(board.locations(C)), total_mines - len(flags))
        if all(
            board.cells[r][c] == sum(n in set(selected) | flags for n in board.neighbors(r, c))
            for r in range(board.rows)
            for c in range(board.cols)
            if board.cells[r][c] >= 0
        )
    )


def oracle_actions(board, total_mines):
    """Exhaust all reveal strategies using actual simulator transitions.

    Unlike the production bit-mask search this groups whole visible Boards,
    has no pruning/known-layout shortcut, and never calls the CSP solver.
    Returns (winning layouts, total inputs on winning branches) per action.
    """
    layouts = visible_layouts(board, total_mines)

    @lru_cache(None)
    def actions(current, ids):
        scores = {}
        for cell in sorted(current.locations(C)):
            groups = {}
            wins = cost = 0
            for layout_id in ids:
                sim = Simulation(current.rows, current.cols, layouts[layout_id])
                sim._cells = [list(row) for row in current.cells]
                sim.apply(Move("reveal", *cell, "oracle"))
                if sim.status == "won":
                    wins += 1
                    cost += 1
                elif sim.status != "lost":
                    groups.setdefault(sim.observe(), []).append(layout_id)
            for observed, group in groups.items():
                child = max(actions(observed, tuple(group)).values(), key=lambda s: (s[0], -s[1]))
                wins += child[0]
                cost += child[1] + child[0]
            scores[cell] = (wins, cost)
        return scores

    return layouts, actions(board, tuple(range(len(layouts))))


def assert_matches_oracle(board, mines):
    analysis = analyze(board, mines)
    if analysis.safe:
        return False
    result = exact_endgame(board, mines, analysis, max_work=5_000_000)
    assert result is not None
    layouts, scores = oracle_actions(board, mines)
    wins, cost = max(scores.values(), key=lambda s: (s[0], -s[1]))
    assert scores[result.cell] == (wins, cost)
    assert result.layouts == len(layouts)
    assert result.win_probability == pytest.approx(wins / len(layouts))
    assert result.winning_reveals == pytest.approx(cost / wins)
    assert result.mine_probability == pytest.approx(analysis.probabilities[result.cell])
    return True


def test_endgame_agrees_with_independent_exhaustive_oracle():
    positions = [(r, c) for r in range(2) for c in range(3)]
    checked = set()
    for mines in combinations(positions, 2):
        for opened in set(positions) - set(mines):
            sim = Simulation(2, 3, mines)
            sim.apply(Move("reveal", *opened, "initial observation"))
            board = sim.observe()
            if sim.status == "playing" and board not in checked:
                if assert_matches_oracle(board, 2):
                    checked.add(board)
    assert len(checked) == 6


def test_endgame_completion_probability_can_outweigh_immediate_safety():
    board = Board(((C, C, 1, C), (C, C, C, 1), (C, C, C, C)))
    assert_matches_oracle(board, 3)
    analysis = analyze(board, 3)
    result = exact_endgame(board, 3, analysis)
    assert result.mine_probability > min(analysis.probabilities.values())
    assert result.win_probability == pytest.approx(19 / 28)


def test_endgame_honors_existing_flags_and_discards_incomplete_work():
    board = Board(((FLAG, 2, C), (C, C, C)))
    assert_matches_oracle(board, 2)
    analysis = analyze(board, 2)
    assert exact_endgame(board, 2, analysis, max_work=0) is None
    assert exact_endgame(board, 2, analysis, max_layouts=1) is None
    assert exact_endgame(board, 2, Analysis()) is None


def test_flags_are_only_spent_to_prepare_a_profitable_chord():
    board = Board(
        (
            (FLAG, FLAG, FLAG, FLAG, FLAG),
            (FLAG, C, C, C, FLAG),
            (FLAG, C, 1, C, FLAG),
            (FLAG, C, C, C, FLAG),
            (FLAG, FLAG, FLAG, FLAG, FLAG),
        )
    )
    # Supply independently established facts to isolate physical-input planning.
    mines = {(1, 1)}
    safe = board.locations(C) - mines
    analysis = Analysis(
        safe=safe,
        mines=mines,
        probabilities={cell: float(cell in mines) for cell in board.locations(C)},
    )
    move = _safe_action(board, analysis)
    assert (move.kind, move.row, move.col) == ("flag", 1, 1)
    raw = [list(row) for row in board.cells]
    raw[1][1] = FLAG
    flagged = Board(tuple(map(tuple, raw)))
    move = _safe_action(flagged, analysis)
    assert (move.kind, move.row, move.col) == ("chord", 2, 2)


def test_known_mines_do_not_cause_cosmetic_flag_inputs():
    board = Board(((1, C, C, C, C, C),))
    settings = Settings(1, 6, Rect(0, 0, 96, 16), 3)
    assert analyze(board, 3).mines == {(0, 1)}
    assert choose_move(board, settings, False) == (None, None, "stalled")
    move, _, status = choose_move(board, settings, True)
    assert status == "guess" and move.kind == "reveal" and move.col != 1


def test_safe_reveals_prefer_a_possible_zero_flood():
    board = Board(((1, C, C, C),))
    move, risk, status = choose_move(board, Settings(1, 4, Rect(0, 0, 64, 16), 1), False)
    assert (move.row, move.col, risk, status) == (0, 3, 0, "proven")


def test_progress_uses_exact_outcome_weights_without_mutating_observations():
    board = Board(((1, C, C, C), (C, C, C, C), (C, C, C, C)))
    analysis = analyze(board, 3)
    rank = lambda cell: (analysis.probabilities[cell], cell)
    choice = progress_choice(board, 3, analysis, rank)
    assert choice is not None
    layouts = visible_layouts(board, 3)
    progress = zero = survivors = 0
    for layout in layouts:
        if choice.cell in layout:
            continue
        survivors += 1
        clue = sum(n in layout for n in board.neighbors(*choice.cell))
        raw = [list(row) for row in board.cells]
        raw[choice.cell[0]][choice.cell[1]] = clue
        outcome = Board(tuple(map(tuple, raw)))
        compatible = visible_layouts(outcome, 3)
        progress += any(
            all(cell not in candidate for candidate in compatible) for cell in outcome.locations(C)
        )
        zero += clue == 0
    assert choice.progress_probability == pytest.approx(progress / len(layouts))
    assert choice.score == pytest.approx((survivors + 0.08 * progress + 0.02 * zero) / len(layouts))
    assert board.cells == ((1, C, C, C), (C, C, C, C), (C, C, C, C))


def test_incomplete_fallback_prevents_lookahead_switching_candidates(monkeypatch):
    board = Board(((1, C, C, C), (C, C, C, C), (C, C, C, C)))
    analysis = analyze(board, 3)
    rank = lambda cell: (analysis.probabilities[cell], cell)
    fallback = min(board.locations(C), key=rank)

    def incomplete_fallback(hypothetical, *args, **kwargs):
        if hypothetical.cells[fallback[0]][fallback[1]] >= 0:
            return Analysis(notes=["search cap"])
        return analyze(hypothetical, *args, **kwargs)

    monkeypatch.setattr("minesweeper_agent.lookahead.analyze", incomplete_fallback)
    assert progress_choice(board, 3, analysis, rank) is None


def test_simulator_zero_flood_and_seed_pairing():
    game = Simulation(1, 4, {(0, 3)})
    game.apply(Move("reveal", 0, 0, "zero"))
    assert game.status == "won" and game.observe().cells == ((0, 0, 1, FLAG),)
    for difficulty in ("beginner", "intermediate", "expert"):
        first = Simulation.seeded(difficulty, 31337)
        second = Simulation.seeded(difficulty, 31337)
        assert first._mines == second._mines and (0, 0) not in first._mines


def test_wrong_flag_chord_really_loses_in_simulator():
    game = Simulation(2, 2, {(0, 0)})
    game.apply(Move("reveal", 1, 1, "clue"))
    game.apply(Move("flag", 0, 1, "deliberately wrong"))
    game.apply(Move("chord", 1, 1, "must explode"))
    assert game.status == "lost"


def test_chord_transition_requires_all_targets_and_matching_flags():
    before = Board(((FLAG, C), (C, 1)))
    action = Move("chord", 1, 1, "test")
    validate_transition(before, Board(((FLAG, 1), (1, 1))), action)
    with pytest.raises(ValueError, match="every intended neighbour"):
        validate_transition(before, Board(((FLAG, C), (1, 1))), action)
    with pytest.raises(ValueError, match="required visible flags"):
        validate_transition(Board(((C, C), (C, 1))), Board(((1, 1), (1, 1))), action)


def test_controller_sends_one_middle_click_for_chord(monkeypatch):
    calls = []
    mouse = SimpleNamespace(size=lambda: (1920, 1080), click=lambda **kwargs: calls.append(kwargs))
    monkeypatch.setitem(__import__("sys").modules, "pyautogui", mouse)
    execute(Move("chord", 1, 1, "test"), Settings(2, 2, Rect(0, 0, 32, 32), 1))
    assert calls == [{"x": 24, "y": 24, "button": "middle"}]


def test_comparison_counts_all_outcomes_and_pairs_only_shared_wins():
    records = []
    for policy, outcomes in (
        ("legacy", ["won", "won", "lost"]),
        ("optimized", ["won", "stalled", "won"]),
    ):
        for seed, status in enumerate(outcomes):
            records.append(
                dict(
                    difficulty="beginner",
                    policy=policy,
                    seed=seed,
                    status=status,
                    clicks=20 if policy == "legacy" else 10,
                    decision_seconds=0,
                )
            )
    summary = summarize(records)["beginner"]
    assert summary["optimized"]["win_rate"] == 2 / 3
    assert summary["paired"]["shared_wins"] == 1
    assert summary["paired"]["new_only_wins"] == summary["paired"]["legacy_only_wins"] == 1
    assert summary["paired"]["legacy_shared_clicks"] == 20
    assert summary["paired"]["optimized_shared_clicks"] == 10
