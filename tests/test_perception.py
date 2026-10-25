import cv2
import numpy as np

from minesweeper_agent.model import COVERED, UNKNOWN
from minesweeper_agent.perception import TemplateRecognizer, save_template


def test_labeled_templates_recognize_distinct_tiles(tmp_path):
    # Two synthetic tile appearances; no game process or screen is required.
    image = np.empty((32, 64, 3), dtype=np.uint8)
    image[:, :32] = (125, 125, 125)
    image[:, 32:] = (0, 0, 255)
    save_template(image, 1, 2, 0, 0, "covered", tmp_path)
    save_template(image, 1, 2, 0, 1, "1", tmp_path)
    recognizer = TemplateRecognizer(tmp_path)
    board, scores = recognizer.read(image, 1, 2)
    assert board.cells == ((COVERED, 1),)
    assert all(score == 0 for score in scores.values())


def test_missing_templates_result_in_unknown(tmp_path):
    recognizer = TemplateRecognizer(tmp_path)
    value, _ = recognizer.classify(np.zeros((32, 32, 3), dtype=np.uint8))
    assert value == UNKNOWN
