from minesweeper_agent.model import Board, COVERED, FLAG, UNKNOWN
from minesweeper_agent.solver import analyze
from itertools import combinations
from random import Random
from collections import Counter
from itertools import product
from minesweeper_agent.solver import _enumerate_component


def test_zero_and_one_propagate_to_forced_moves():
    board = Board(((1, 1, 0), (COVERED, COVERED, COVERED)))
    result = analyze(board)
    assert not result.errors
    assert (1, 0) in result.mines
    assert {(1, 1), (1, 2)} <= result.safe
    assert result.next_move().kind == "reveal"


def test_two_valid_layouts_give_half_probability():
    board = Board(((1, 1), (COVERED, COVERED)))
    result = analyze(board)
    assert result.probabilities[(1, 0)] == 0.5
    assert result.probabilities[(1, 1)] == 0.5
    assert result.next_move() is None


def test_total_mines_makes_unconstrained_cells_safe():
    board = Board(((1, COVERED, COVERED, COVERED),))
    result = analyze(board, total_mines=1)
    assert result.mines == {(0, 1)}
    assert {(0, 2), (0, 3)} <= result.safe


def test_impossible_clue_blocks_every_move():
    result = analyze(Board(((2, COVERED),)))
    assert result.errors
    assert result.next_move() is None


def test_unknown_vision_blocks_every_move():
    result = analyze(Board(((0, UNKNOWN),)), total_mines=0)
    assert result.errors
    assert result.next_move() is None


def test_flags_are_assumed_mines():
    result = analyze(Board(((1, FLAG, COVERED),)), total_mines=1)
    # The rightmost cell is not adjacent to the clue; with total=1 it is safe.
    assert (0, 2) in result.safe if not result.errors else False


def test_full_unconstrained_board_with_known_mine_count():
    board = Board(((COVERED, COVERED),))
    result = analyze(board, total_mines=1)
    assert all(p == 0.5 for p in result.probabilities.values())
    assert result.next_move() is None


def test_global_probabilities_agree_with_brute_force_random_small_boards():
    rng = Random(20261025)
    positions = [(r, c) for r in range(3) for c in range(3)]
    for _ in range(75):
        true_mines = set(rng.sample(positions, 2))
        flags = {cell for cell in true_mines if rng.random() < 0.35}
        revealed = {cell for cell in positions if cell not in true_mines
                    and rng.random() < 0.45}
        raw = [[COVERED] * 3 for _ in range(3)]
        for r, c in flags:
            raw[r][c] = FLAG
        base = Board(tuple(tuple(row) for row in raw))
        for r, c in revealed:
            raw[r][c] = sum(neighbor in true_mines for neighbor in base.neighbors(r, c))
        board = Board(tuple(tuple(row) for row in raw))
        covered = sorted(board.locations(COVERED))
        total_remaining = len(true_mines) - len(flags)
        valid = []
        for selected in combinations(covered, total_remaining):
            selected_set = set(selected) | flags
            if all(board.cells[r][c] == sum(n in selected_set for n in board.neighbors(r, c))
                   for r, c in revealed):
                valid.append(selected_set)
        assert valid
        result = analyze(board, total_mines=len(true_mines))
        assert not result.errors
        assert result.layout_count == len(valid)
        for cell in covered:
            expected = sum(cell in layout for layout in valid) / len(valid)
            assert abs(result.probabilities[cell] - expected) < 1e-12
            assert (cell in result.safe) == (expected == 0)
            assert (cell in result.mines) == (expected == 1)


def test_subset_rule_proves_moves_even_when_enumeration_is_capped():
    board = Board(((1, 1, COVERED), (COVERED, COVERED, COVERED)))
    result = analyze(board, max_component_cells=1)
    assert not result.errors
    assert result.safe == {(0, 2), (1, 2)}
    assert result.notes


def test_48_cell_frontier_has_exact_half_probabilities():
    board = Board((tuple(COVERED if col % 2 == 0 else 1 for col in range(95)),))
    capped = analyze(board, total_mines=24, max_component_cells=22)
    assert capped.notes and not capped.probabilities
    result = analyze(board, total_mines=24, max_component_cells=96)
    assert not result.errors and not result.notes
    assert len(result.probabilities) == 48
    assert all(p == .5 for p in result.probabilities.values())
    assert result.next_move() is None


def test_exhausted_search_discards_incomplete_probabilities():
    board = Board(((1, 1), (COVERED, COVERED)))
    result = analyze(board, total_mines=1, max_search_nodes=0)
    assert result.notes and "budget" in result.notes[0]
    assert not result.probabilities
    assert result.layout_count is None
    assert result.next_move() is None


def test_budget_after_one_valid_layout_does_not_turn_it_into_proof():
    board = Board(((COVERED, 1, COVERED, 1, COVERED),))
    complete = analyze(board)
    assert set(complete.probabilities.values()) == {.5}
    partial = analyze(board, max_search_nodes=4)
    assert partial.notes and not partial.errors
    assert not partial.safe and not partial.mines and not partial.probabilities
    assert partial.next_move() is None


def test_grouped_enumeration_counts_each_labelled_layout():
    cells = [(0, col) for col in range(7)]
    constraints = [(frozenset(cells[:4]), 2), (frozenset(cells[2:]), 3)]
    component = _enumerate_component(cells, constraints)
    counts = Counter()
    mine_counts = {cell: Counter() for cell in cells}
    for values in product((0, 1), repeat=len(cells)):
        layout = dict(zip(cells, values))
        if all(sum(layout[c] for c in group) == target for group, target in constraints):
            mines = sum(values)
            counts[mines] += 1
            for cell in cells:
                if layout[cell]:
                    mine_counts[cell][mines] += 1
    assert component.counts == counts
    assert component.mine_counts == mine_counts


def test_global_count_conflict_blocks_even_capped_frontiers():
    # First clue forces one mine; total=0 must block before any early cap return.
    result = analyze(Board(((1, COVERED),)), total_mines=0, max_component_cells=0)
    assert result.errors and result.next_move() is None
