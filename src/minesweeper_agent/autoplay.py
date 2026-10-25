"""Bounded observe/solve/act/verify loop with replayable, game-only evidence."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import html
import json
from pathlib import Path
import time

import cv2

from .classic import ClassicRecognizer, detect_grid
from .model import Board, COVERED, FLAG, MINE, UNKNOWN, Move, Settings
from .policy import choose_move


def validate_transition(before: Board, after: Board, move: Move):
    if after.locations(UNKNOWN) or after.locations(MINE):
        return  # next policy evaluation reports the specific terminal state
    for r in range(before.rows):
        for c in range(before.cols):
            old, new = before.cells[r][c], after.cells[r][c]
            if old != COVERED and old != new:
                raise ValueError(f"Previously observed cell ({r},{c}) changed unexpectedly")
    target = after.cells[move.row][move.col]
    if move.kind == "reveal" and not 0 <= target <= 8:
        raise ValueError("Click did not reveal the intended tile; stopped")
    if move.kind == "flag" and target != FLAG:
        raise ValueError("Click did not flag the intended tile; stopped")
    if move.kind == "chord":
        neighbors = before.neighbors(move.row, move.col)
        targets = [(r, c) for r, c in neighbors if before.cells[r][c] == COVERED]
        flags = sum(before.cells[r][c] == FLAG for r, c in neighbors)
        if flags != before.cells[move.row][move.col] or not targets:
            raise ValueError("Chord did not have the required visible flags/targets")
        if not all(0 <= after.cells[r][c] <= 8 for r, c in targets):
            raise ValueError("Chord did not open every intended neighbour; stopped")


class RunLog:
    def __init__(self, data_dir: Path):
        self.folder = data_dir / "runs" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        self.folder.mkdir(parents=True)
        self.events = []

    def add(self, event: dict, frame=None):
        event = {"time": datetime.now().astimezone().isoformat(), **event}
        if frame is not None:
            filename = f"frame-{len(self.events):04d}.png"
            if not cv2.imwrite(str(self.folder / filename), frame):
                raise OSError("Unable to save game screenshot")
            event["frame"] = filename
        self.events.append(event)
        with (self.folder / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")

    def finish(self, outcomes: list[dict], status: str):
        summary = {"status": status, "games": outcomes,
                   "wins": sum(g["status"] == "won" for g in outcomes),
                   "losses": sum(g["status"] == "lost" for g in outcomes),
                   "log": str(self.folder.resolve())}
        (self.folder / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        cards = []
        results = ''.join('<tr>' + ''.join(f'<td>{html.escape(str(g.get(key,"")))}</td>'
                                          for key in ("game","difficulty","status","clicks","flags","chords","guesses","seconds")) + '</tr>'
                          for g in outcomes)
        for e in self.events:
            if "frame" in e:
                label = f"Game {e.get('game', '?')} / step {e.get('step', '?')} / {e.get('status', e['event'])}"
                detail = e.get("move", {}).get("reason", e.get("message", ""))
                cards.append(f'<article><h3>{html.escape(label)}</h3><img src="{e["frame"]}">'
                             f'<p>{html.escape(detail)}</p></article>')
        report = ('<!doctype html><meta charset="utf-8"><title>Minesweeper run</title>'
                  '<style>body{font:16px system-ui;background:#14202b;color:#eee;margin:32px}'
                  'main{display:flex;flex-wrap:wrap;gap:20px}article{background:#233545;padding:18px;'
                  'border-radius:12px;max-width:520px}img{image-rendering:pixelated;max-width:100%}'
                  'p{max-width:440px}td,th{padding:8px;text-align:left}</style><h1>Minesweeper visual agent</h1>'
                  f'<p>Status: {html.escape(status)} · Wins: {summary["wins"]} · Losses: {summary["losses"]}</p>'
                  '<table><tr><th>Game</th><th>Difficulty</th><th>Result</th><th>Clicks</th>'
                  '<th>Flags</th><th>Chords</th><th>Guesses</th><th>Seconds</th></tr>'
                  + results + '</table><p>Real screen captures. Each action is followed by another observation.</p><main>'
                  + ''.join(cards) + '</main>')
        (self.folder / "report.html").write_text(report, encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return summary


def play(args, game=None) -> dict:
    if game is None:
        from .windows_game import WindowsGame
        game = WindowsGame(args.game)
    recognizer = ClassicRecognizer()
    log = RunLog(args.data_dir)
    outcomes = []
    final_status = "finished"
    phase = "launch"
    active = None
    frame = None
    game_started = 0.0
    try:
        game.open()
        for number in range(1, args.games + 1):
            active = {"game": number, "difficulty": args.difficulty, "clicks": 0,
                      "guesses": 0, "flags": 0, "chords": 0}
            game_started = time.monotonic()
            frame = None
            phase = "new_game"
            game.new_game(args.difficulty)
            phase = "detect_grid"
            frame, client = game.capture()
            settings = detect_grid(frame, client, args.mines)
            active.update(rows=settings.rows, cols=settings.cols, mines=settings.total_mines)
            expected_size = {"beginner": (9, 9), "intermediate": (16, 16), "expert": (16, 30)}
            if args.difficulty in expected_size and (settings.rows, settings.cols) != expected_size[args.difficulty]:
                raise ValueError("Difficulty shortcut did not select the expected grid")
            settings.save(args.data_dir / "classic-board.json")
            log.add({"event": "start", "game": number, "settings": asdict(settings), "client": asdict(client),
                     "allow_guess": args.allow_guess}, frame)
            print(f"Game {number}: {settings.rows}x{settings.cols}, {settings.total_mines} mines, ROI={settings.rect}", flush=True)
            previous = None
            pending = None
            status = "step_limit"
            for step in range(args.max_steps + 1):
                phase = "observe"
                frame, client = game.capture()
                x, y = settings.rect.left - client.left, settings.rect.top - client.top
                roi = frame[y:y + settings.rect.height, x:x + settings.rect.width]
                board, _ = recognizer.read(roi, settings.rows, settings.cols)
                if pending:
                    validate_transition(previous, board, pending)
                solve_started = time.monotonic()
                move, probability, status = choose_move(board, settings, args.allow_guess)
                solve_ms = round((time.monotonic() - solve_started) * 1000, 3)
                if step == args.max_steps and move:
                    move, status = None, "step_limit"
                event = {"event": "observe", "game": number, "step": step, "status": status,
                         "board": board.cells, "mine_probability": probability, "solve_ms": solve_ms}
                if move:
                    event["move"] = asdict(move)
                log.add(event, frame)
                print(f"  {step:03d} {status}" + (f" {move.kind} ({move.row},{move.col}) p={probability}" if move else ""), flush=True)
                if not move:
                    break
                # Re-read immediately before the click, then the backend checks focus/position.
                time.sleep(.06)
                check_frame, check_client = game.capture()
                if check_client != client:
                    raise ValueError("Window geometry changed before input")
                check, _ = recognizer.read(check_frame[y:y + settings.rect.height, x:x + settings.rect.width],
                                           settings.rows, settings.cols)
                if check != board:
                    raise ValueError("Board changed while planning; stopped before input")
                phase = "click"
                game.click(move, settings)
                active["clicks"] += 1
                active["guesses"] += int(status == "guess")
                active["flags"] += int(move.kind == "flag")
                active["chords"] += int(move.kind == "chord")
                log.add({"event": "click", "game": number, "step": step, "move": asdict(move)})
                previous, pending = board, move
                time.sleep(args.delay)
            outcomes.append({**active, "status": status, "seconds": round(time.monotonic()-game_started, 2)})
            active = None
            if status == "won" and args.stop_on_win:
                break
            if status not in ("won", "lost"):
                final_status = status
                break
    except KeyboardInterrupt:
        final_status = "cancelled"
        log.add({"event": "stop", "message": "Stopped by Escape / Ctrl+C"})
    except Exception as exc:
        final_status = "error"
        log.add({"event": "error", "phase": phase, "game": active["game"] if active else None,
                 "message": str(exc)}, frame)
        print(f"STOP [{phase}]: {exc}", flush=True)
    if active is not None:
        outcomes.append({**active, "status": final_status, "seconds": round(time.monotonic()-game_started, 2)})
    return log.finish(outcomes, final_status)
