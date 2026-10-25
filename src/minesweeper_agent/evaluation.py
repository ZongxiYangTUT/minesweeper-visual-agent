"""Seeded, screen-independent policy evaluation; never used to control live games.

Policies receive only Board/Settings. Hidden layouts stay inside the simulator.
All policies see the same uniformly sampled first-click-safe layout for a seed.
This models standard Minesweeper, not the XP executable's random-number generator.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from math import sqrt, inf
from pathlib import Path
from random import Random
from statistics import mean, median
import time

from .model import Board, COVERED, FLAG, MINE, Move, Rect, Settings

DIFFICULTIES = {"beginner": (9, 9, 10), "intermediate": (16, 16, 40), "expert": (16, 30, 99)}


class Simulation:
    def __init__(self, rows, cols, mines):
        self.rows, self.cols = rows, cols
        self._mines = frozenset(mines)
        self._cells = [[COVERED] * cols for _ in range(rows)]
        self._neighbors = {
            (r, c): [
                (nr, nc)
                for nr in range(max(0, r - 1), min(rows, r + 2))
                for nc in range(max(0, c - 1), min(cols, c + 2))
                if (nr, nc) != (r, c)
            ]
            for r in range(rows)
            for c in range(cols)
        }
        self._clues = {
            cell: sum(n in self._mines for n in neighbors)
            for cell, neighbors in self._neighbors.items()
        }
        self.status = "playing"

    @classmethod
    def seeded(cls, difficulty, seed):
        rows, cols, count = DIFFICULTIES[difficulty]
        cells = [(r, c) for r in range(rows) for c in range(cols) if (r, c) != (0, 0)]
        return cls(rows, cols, Random(seed).sample(cells, count))

    def observe(self):
        return Board(tuple(map(tuple, self._cells)))

    def apply(self, move):
        if self.status != "playing":
            raise ValueError("Cannot act after the game ended")
        cell = (move.row, move.col)
        value = self._cells[move.row][move.col]
        if move.kind == "flag":
            if value != COVERED:
                raise ValueError("Flag must target a covered cell")
            self._cells[move.row][move.col] = FLAG
            return
        if move.kind == "reveal":
            if value != COVERED:
                raise ValueError("Reveal must target a covered cell")
            targets = [cell]
        elif move.kind == "chord":
            if not 0 <= value <= 8:
                raise ValueError("Chord must target a revealed clue")
            neighbors = self._neighbors[cell]
            if sum(self._cells[r][c] == FLAG for r, c in neighbors) != value:
                raise ValueError("Chord requires exactly the observed number of flags")
            targets = [(r, c) for r, c in neighbors if self._cells[r][c] == COVERED]
            if not targets:
                raise ValueError("Chord has no covered neighbours")
        else:
            raise ValueError(move.kind)
        pending = list(targets)
        while pending:
            r, c = pending.pop()
            if self._cells[r][c] != COVERED:
                continue
            if (r, c) in self._mines:
                self._cells[r][c] = MINE
                self.status = "lost"
                return
            self._cells[r][c] = self._clues[r, c]
            if self._clues[r, c] == 0:
                pending.extend(self._neighbors[r, c])
        revealed = sum(v >= 0 for row in self._cells for v in row)
        if revealed == self.rows * self.cols - len(self._mines):
            self.status = "won"
            for r, c in self._mines:
                self._cells[r][c] = FLAG


def evaluate_game(job):
    difficulty, seed, policy = job
    if policy == "legacy":
        from .legacy_policy import choose_move
    else:
        from .policy import choose_move
    game = Simulation.seeded(difficulty, seed)
    rows, cols, mines = DIFFICULTIES[difficulty]
    settings = Settings(rows, cols, Rect(0, 0, cols * 16, rows * 16), mines)
    count = Counter()
    decision_seconds = 0.0
    status = "step_limit"
    for step in range(1000):
        board = game.observe()
        before = time.perf_counter()
        move, p, kind = choose_move(board, settings, True)
        decision_seconds += time.perf_counter() - before
        if move is None:
            status = kind
            break
        if move.kind == "flag" and (move.row, move.col) not in game._mines:
            raise AssertionError("Policy falsely proved a mine")
        game.apply(move)
        count[move.kind] += 1
        count["guesses"] += int(kind == "guess")
        if game.status == "lost" and kind != "guess":
            raise AssertionError("A proven move hit a mine")
        if game.status != "playing":
            status = game.status
            break
    return {
        "difficulty": difficulty,
        "seed": seed,
        "policy": policy,
        "status": status,
        "clicks": count["reveal"] + count["flag"] + count["chord"],
        "reveals": count["reveal"],
        "flags": count["flag"],
        "chords": count["chord"],
        "guesses": count["guesses"],
        "decision_seconds": round(decision_seconds, 6),
    }


def _deterministic_solver_budget():
    # Machine load must not change whether a seeded comparison completes its
    # search. Retain the exact same node budgets; disable only wall-clock limits
    # inside offline workers. Live play keeps its time guard.
    from . import solver

    original = solver.analyze

    def by_nodes(*args, **kwargs):
        kwargs["max_search_seconds"] = inf
        return original(*args, **kwargs)

    solver.analyze = by_nodes


def summarize(records):
    output = {}
    for difficulty in sorted({r["difficulty"] for r in records}):
        by_policy = {}
        for policy in sorted({r["policy"] for r in records}):
            runs = [r for r in records if r["difficulty"] == difficulty and r["policy"] == policy]
            won = [r for r in runs if r["status"] == "won"]
            by_policy[policy] = {
                "games": len(runs),
                "outcomes": dict(Counter(r["status"] for r in runs)),
                "win_rate": len(won) / len(runs),
                "winning_clicks_mean": mean(r["clicks"] for r in won) if won else None,
                "winning_clicks_median": median(r["clicks"] for r in won) if won else None,
                "mean_decision_seconds": mean(r["decision_seconds"] for r in runs),
            }
        old = {
            r["seed"]: r
            for r in records
            if r["difficulty"] == difficulty and r["policy"] == "legacy"
        }
        new = {
            r["seed"]: r
            for r in records
            if r["difficulty"] == difficulty and r["policy"] == "optimized"
        }
        seeds = old.keys() & new.keys()
        if seeds:
            improved = sum(new[s]["status"] == "won" and old[s]["status"] != "won" for s in seeds)
            regressed = sum(old[s]["status"] == "won" and new[s]["status"] != "won" for s in seeds)
            delta = (improved - regressed) / len(seeds)
            se = sqrt(max(0, (improved + regressed) / len(seeds) - delta * delta) / len(seeds))
            shared = [s for s in seeds if old[s]["status"] == new[s]["status"] == "won"]
            by_policy["paired"] = {
                "new_only_wins": improved,
                "legacy_only_wins": regressed,
                "win_rate_delta": delta,
                "approx_95pct_ci": [delta - 1.96 * se, delta + 1.96 * se],
                "shared_wins": len(shared),
                "legacy_shared_clicks": mean(old[s]["clicks"] for s in shared) if shared else None,
                "optimized_shared_clicks": (
                    mean(new[s]["clicks"] for s in shared) if shared else None
                ),
            }
        output[difficulty] = by_policy
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=10000)
    parser.add_argument("--difficulty", nargs="+", choices=DIFFICULTIES, default=list(DIFFICULTIES))
    parser.add_argument(
        "--policies", nargs="+", choices=("legacy", "optimized"), default=["legacy", "optimized"]
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("data/benchmarks/comparison.json"))
    args = parser.parse_args()
    if args.games < 1 or not 1 <= args.workers <= 16:
        parser.error("games must be positive; workers must be 1..16")
    jobs = [
        (d, s, p)
        for s in range(args.seed_start, args.seed_start + args.games)
        for d in args.difficulty
        for p in args.policies
    ]
    records = []
    started = time.monotonic()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(
        max_workers=args.workers, initializer=_deterministic_solver_budget
    ) as pool:
        for index, record in enumerate(pool.map(evaluate_game, jobs, chunksize=1), 1):
            records.append(record)
            if index % 50 == 0 or index == len(jobs):
                print(f"{index}/{len(jobs)} games, {time.monotonic()-started:.1f}s", flush=True)
                result = {
                    "configuration": {
                        **vars(args),
                        "output": str(args.output),
                        "solver_budget": "deterministic node limits; offline wall-clock limit disabled",
                    },
                    "summary": summarize(records),
                    "games": records,
                }
                args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(summarize(records), indent=2))


if __name__ == "__main__":
    main()
