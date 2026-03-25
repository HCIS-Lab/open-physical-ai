from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

import opai
from opai.application import calibration as calibration_module
from opai.core.exceptions import OPAIContextError, OPAIValidationError
from opai.domain.calibration import CharucoBoardConfig, validate_charuco_board_config
from opai.infrastructure import context_store
from opai.infrastructure import persistence as persistence_module


@pytest.fixture(autouse=True)
def reset_active_context(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(context_store, "_ACTIVE_CONTEXT", None)


def test_validate_charuco_board_config_rejects_invalid_marker_length() -> None:
    config = CharucoBoardConfig(
        dictionary="DICT_5X5_100",
        squares_x=11,
        squares_y=8,
        square_length=0.02,
        marker_length=0.02,
        image_width_px=2000,
        image_height_px=1400,
        margin_size_px=20,
    )

    with pytest.raises(OPAIValidationError, match="smaller than square_length"):
        validate_charuco_board_config(config)


def test_generate_charuco_board_requires_context(monkeypatch) -> None:
    monkeypatch.setattr(context_store, "_ACTIVE_CONTEXT", None)

    with pytest.raises(OPAIContextError, match="Call opai.init"):
        opai.generate_charuco_board()


def test_generate_charuco_board_writes_image_and_config(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    opai.init("session-001")
    fake_cv2 = _build_fake_cv2()
    monkeypatch.setattr(calibration_module, "cv2", fake_cv2)
    monkeypatch.setattr(persistence_module, "cv2", fake_cv2)

    artifacts = opai.generate_charuco_board()

    image_path = tmp_path / "sessions" / "session-001" / "charuco_board.png"
    config_path = tmp_path / "sessions" / "session-001" / "charuco_config.json"

    assert artifacts.image_path == image_path
    assert artifacts.config_path == config_path
    assert image_path.read_bytes() == b"fake-png"

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    assert payload == {
        "dictionary": "DICT_5X5_100",
        "squares_x": 11,
        "squares_y": 8,
        "square_length": 0.03,
        "marker_length": 0.022,
        "image_width_px": 2000,
        "image_height_px": 1400,
        "margin_size_px": 20,
        "board_image_path": "charuco_board.png",
    }


def test_generate_charuco_board_rejects_unknown_dictionary(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    opai.init("session-001")
    fake_cv2 = _build_fake_cv2()
    monkeypatch.setattr(calibration_module, "cv2", fake_cv2)
    monkeypatch.setattr(persistence_module, "cv2", fake_cv2)

    with pytest.raises(OPAIValidationError, match="Unsupported ArUco dictionary"):
        opai.generate_charuco_board(dictionary="DICT_DOES_NOT_EXIST")


def _build_fake_cv2() -> SimpleNamespace:
    class FakeBoard:
        def __init__(
            self,
            size: tuple[int, int],
            square_length: float,
            marker_length: float,
            dictionary: object,
        ) -> None:
            self.size = size
            self.square_length = square_length
            self.marker_length = marker_length
            self.dictionary = dictionary

        def generateImage(
            self,
            image_size: tuple[int, int],
            *,
            marginSize: int,
        ) -> np.ndarray:
            width, height = image_size
            return np.full((height, width), fill_value=marginSize, dtype=np.uint8)

    def fake_imwrite(path: str, _image: np.ndarray) -> bool:
        from pathlib import Path

        Path(path).write_bytes(b"fake-png")
        return True

    return SimpleNamespace(
        aruco=SimpleNamespace(
            DICT_5X5_100=100,
            getPredefinedDictionary=lambda dictionary_id: {"id": dictionary_id},
            CharucoBoard=FakeBoard,
        ),
        imwrite=fake_imwrite,
    )
