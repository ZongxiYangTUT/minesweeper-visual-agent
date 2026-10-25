"""Interactive calibration, annotation, read-only inspection, and one-step play."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import cv2

from .capture import configure_dpi_awareness, screenshot
from .controller import execute
from .model import FLAG, COVERED, UNKNOWN, MINE, Rect, Settings
from .perception import TemplateRecognizer, save_template
from .solver import analyze


def _data_paths(data_dir: Path) -> tuple[Path, Path]:
    return data_dir / "board.json", data_dir / "templates"


def _load(data_dir: Path) -> tuple[Settings, Path]:
    config, templates = _data_paths(data_dir)
    if not config.exists():
        raise FileNotFoundError("No board configuration. Run 'calibrate' first.")
    return Settings.load(config), templates


def _read(data_dir: Path, threshold: float):
    settings, templates = _load(data_dir)
    image, _ = screenshot(settings.rect)
    recognizer = TemplateRecognizer(templates, threshold=threshold)
    board, scores = recognizer.read(image, settings.rows, settings.cols)
    return settings, recognizer, board, scores


def calibrate(args: argparse.Namespace) -> None:
    image, desktop = screenshot()
    name = "Select tile GRID only (no border/counter) | ENTER accept | ESC cancel"
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, min(image.shape[1], 1400), min(image.shape[0], 900))
    try:
        x, y, width, height = cv2.selectROI(name, image, showCrosshair=True,
                                            fromCenter=False)
    finally:
        cv2.destroyWindow(name)
    if width <= 0 or height <= 0:
        raise ValueError("Calibration cancelled or empty rectangle")
    settings = Settings(rows=args.rows, cols=args.cols,
                        rect=Rect(desktop.left + x, desktop.top + y, width, height),
                        total_mines=args.mines)
    config, _ = _data_paths(args.data_dir)
    settings.save(config)
    print(f"Saved {settings.rows}x{settings.cols}, screen ROI={settings.rect}")
    print("Keep the window fixed and unchanged in size. Next run: annotate")


def annotate(args: argparse.Namespace) -> None:
    settings, folder = _load(args.data_dir)
    image, _ = screenshot(settings.rect)
    chosen: list[tuple[int, int] | None] = [None]
    scale = max(2, min(6, 720 // max(image.shape[0], image.shape[1])))
    name = "Annotate: click tile, then c / 0..8 / f / m | r refresh | q quit"

    def choose(event, x, y, _flags, _userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            col = min(settings.cols - 1, int(x * settings.cols / (image.shape[1] * scale)))
            row = min(settings.rows - 1, int(y * settings.rows / (image.shape[0] * scale)))
            chosen[0] = (row, col)
            print(f"Selected tile (row={row}, col={col})")

    cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(name, choose)
    key_to_label = {ord("c"): "covered", ord("f"): "flag", ord("m"): "mine",
                    **{ord(str(n)): str(n) for n in range(9)}}
    print("Select an actual tile, then press its label. Press 'r' after game changes.")
    print("c=covered, f=flag, m=visible mine, 0-8=revealed clue. q=exit.")
    try:
        while True:
            canvas = cv2.resize(image, (image.shape[1] * scale, image.shape[0] * scale),
                                interpolation=cv2.INTER_NEAREST)
            for r in range(1, settings.rows):
                y = round(r * canvas.shape[0] / settings.rows)
                cv2.line(canvas, (0, y), (canvas.shape[1], y), (80, 200, 230), 1)
            for c in range(1, settings.cols):
                x = round(c * canvas.shape[1] / settings.cols)
                cv2.line(canvas, (x, 0), (x, canvas.shape[0]), (80, 200, 230), 1)
            if chosen[0] is not None:
                r, c = chosen[0]
                x0 = round(c * canvas.shape[1] / settings.cols)
                y0 = round(r * canvas.shape[0] / settings.rows)
                x1 = round((c + 1) * canvas.shape[1] / settings.cols)
                y1 = round((r + 1) * canvas.shape[0] / settings.rows)
                cv2.rectangle(canvas, (x0, y0), (x1, y1), (0, 0, 255), 2)
            cv2.imshow(name, canvas)
            key = cv2.waitKey(50) & 0xFF
            if key in (ord("q"), 27):
                return
            if key == ord("r"):
                image, _ = screenshot(settings.rect)
                chosen[0] = None
                print("Captured a fresh screenshot")
            elif key in key_to_label:
                if chosen[0] is None:
                    print("Click a tile first")
                    continue
                row, col = chosen[0]
                label = key_to_label[key]
                target = save_template(image, settings.rows, settings.cols,
                                       row, col, label, folder)
                print(f"Added template '{label}' at ({row},{col}) -> {target.name}")
    finally:
        cv2.destroyWindow(name)


def inspect(args: argparse.Namespace) -> None:
    settings, recognizer, board, scores = _read(args.data_dir, args.threshold)
    print(board.render())
    uncertain = [(r, c, scores[r, c]) for r, c in sorted(board.locations(UNKNOWN))]
    if uncertain:
        print(f"\nUnrecognized ({len(uncertain)}): " +
              ", ".join(f"({r},{c}; score={score:.3f})" for r, c, score in uncertain[:20]))
        print("Use 'annotate' to add missing digit or tile templates.")
    else:
        print("\nAll cells recognized.")
    print("Templates available: " + (", ".join(sorted(recognizer.templates)) or "none"))
    if not uncertain and not board.locations(MINE):
        result = analyze(board, settings.total_mines)
        print(f"Proven safe: {len(result.safe)} | proven mines: {len(result.mines)}")
        if result.errors:
            print("Errors: " + "; ".join(result.errors))


def step(args: argparse.Namespace) -> None:
    settings, recognizer, board, _ = _read(args.data_dir, args.threshold)
    print(board.render())
    analysis = analyze(board, settings.total_mines)
    for note in analysis.notes:
        print("Note:", note)
    if analysis.errors:
        print("STOP:", "; ".join(analysis.errors))
        return
    print(f"Proven safe: {len(analysis.safe)} | proven mines: {len(analysis.mines)}")
    move = analysis.next_move()
    if move is None:
        ranked = sorted(((p, cell) for cell, p in analysis.probabilities.items()
                         if board.cells[cell[0]][cell[1]] == COVERED))
        print("No mathematically forced move. No click will be performed.")
        if ranked:
            p, cell = ranked[0]
            print(f"Lowest estimated mine probability: {cell}: {p:.1%} (NOT auto-clicked)")
        if not any(0 <= v <= 8 for line in board.cells for v in line):
            print("No clue is visible yet. Open the first tile manually, then retry.")
        return

    print(f"Suggestion: {move.kind.upper()} row={move.row}, col={move.col}")
    print("Reason:", move.reason)
    if args.dry_run:
        print("Dry run: no click performed")
        return
    answer = input("Execute EXACTLY ONE mouse click? Type 'yes' to confirm: ").strip().lower()
    if answer != "yes":
        print("Cancelled; no click performed")
        return
    # Re-observe the board before actuation: never click if it changed while
    # the user was considering the suggestion.
    new_image, _ = screenshot(settings.rect)
    latest_board, _ = recognizer.read(new_image, settings.rows, settings.cols)
    if latest_board != board:
        print("STOP: The observed board changed after planning. Run step again.")
        return
    x, y = execute(move, settings)
    print(f"Clicked {move.kind} at desktop ({x},{y}). Checking result...")
    time.sleep(0.4)
    new_image, _ = screenshot(settings.rect)
    after, _ = recognizer.read(new_image, settings.rows, settings.cols)
    print(after.render())
    tile = after.cells[move.row][move.col]
    if tile == UNKNOWN:
        print("STOP: Result is not recognized. Annotate it before continuing.")
    elif move.kind == "reveal" and tile == COVERED:
        print("WARNING: The target is still closed. Check focus, ROI and DPI scaling.")
    elif move.kind == "flag" and tile != FLAG:
        print("WARNING: The expected flag was not recognized.")
    else:
        print("One-step action completed. Rerun 'step' for the next suggestion.")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Screen-only classic Minesweeper agent")
    parser.add_argument("--data-dir", type=Path, default=Path("data"),
                        help="Configuration/template folder (default: ./data)")
    parser.add_argument("--threshold", type=float, default=0.045,
                        help="Max template distance before classification becomes unknown")
    sub = parser.add_subparsers(dest="command", required=True)
    calibration = sub.add_parser("calibrate", help="Select exact tile-grid rectangle")
    calibration.add_argument("--rows", required=True, type=int)
    calibration.add_argument("--cols", required=True, type=int)
    calibration.add_argument("--mines", type=int, default=None,
                             help="Total mine count: e.g. beginner=10")
    sub.add_parser("annotate", help="Teach tile templates by clicking and labeling")
    sub.add_parser("inspect", help="Read-only board inspection")
    steps = sub.add_parser("step", help="Propose and optionally perform exactly one proven move")
    steps.add_argument("--dry-run", action="store_true")
    auto = sub.add_parser("play", help="Launch classic XP Minesweeper and play with screen-only feedback")
    auto.add_argument("--game", type=Path,
                      default=Path(__file__).resolve().parents[2] / "game" / "扫雷_5.1.2600.0.exe")
    auto.add_argument("--difficulty", choices=("beginner", "intermediate", "expert", "current"), default="beginner")
    auto.add_argument("--mines", type=int, default=None, help="Mine count for custom/current grids")
    auto.add_argument("--games", type=int, default=1, help="Maximum games (1-100)")
    auto.add_argument("--max-steps", type=int, default=600)
    auto.add_argument("--delay", type=float, default=.25, help="Seconds to wait after each click")
    auto.add_argument("--allow-guess", action="store_true",
                      help="Allow risk-aware guesses with bounded lookahead; games may be lost")
    auto.add_argument("--stop-on-win", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    if not 0 < args.threshold <= 1:
        parser.error("--threshold must be in (0, 1]")
    if sys.platform != "win32":
        parser.error("Live screen capture/control requires Windows; tests are cross-platform")
    configure_dpi_awareness()
    try:
        if args.command == "play":
            if not 1 <= args.games <= 100 or not 1 <= args.max_steps <= 10000 or not .1 <= args.delay <= 10:
                parser.error("play requires games=1..100, max-steps=1..10000, delay=0.1..10")
            from .autoplay import play
            summary = play(args)
            if summary["status"] == "cancelled":
                return 130
            return 0 if summary["status"] == "finished" else 1
        {"calibrate": calibrate, "annotate": annotate,
         "inspect": inspect, "step": step}[args.command](args)
    except (ValueError, FileNotFoundError, OSError, cv2.error) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nCancelled")
        return 130
    return 0
