import json
from types import SimpleNamespace

import numpy as np
import pytest

import opai
from opai.application import calibration as calibration_module


def test_calibrate_requires_context() -> None:
    with pytest.raises(RuntimeError, match="Call opai.init"):
        opai.calibrate([], 3, 3, 1.0, 0.5, "DICT_4X4_50")


def test_init_creates_context_directory(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    ctx = opai.init("session-001")

    assert ctx.name == "session-001"
    assert ctx.session_directory.exists()


def test_calibrate_writes_artifact(tmp_path, monkeypatch) -> None:
    fake_cv2 = _build_fake_cv2()
    monkeypatch.setattr(calibration_module, "cv2", fake_cv2)
    monkeypatch.chdir(tmp_path)
    opai.init("session-001")

    frame = np.zeros((10, 12, 3), dtype=np.uint8)
    result = opai.calibrate(
        [frame],
        3,
        3,
        1.0,
        0.5,
        "DICT_4X4_50",
    )

    output_path = tmp_path / ".opai_sessions" / "session-001" / "calibration.json"
    assert output_path.exists()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["intrinsic_type"] == "FISHEYE"
    assert payload["image_height"] == 10
    assert payload["image_width"] == 12
    assert result.intrinsic_type == "FISHEYE"


def _build_fake_cv2() -> SimpleNamespace:
    board = SimpleNamespace(
        getChessboardCorners=lambda: np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
    )

    aruco = SimpleNamespace(
        DICT_4X4_50=1,
        getPredefinedDictionary=lambda _: "dictionary",
        CharucoBoard=lambda *args, **kwargs: board,
        detectMarkers=lambda *args, **kwargs: (
            [np.zeros((4, 1, 2), dtype=np.float32)],
            np.array([[0]], dtype=np.int32),
            None,
        ),
        interpolateCornersCharuco=lambda **kwargs: (
            4,
            np.array(
                [
                    [[1.0, 1.0]],
                    [[2.0, 1.0]],
                    [[1.0, 2.0]],
                    [[2.0, 2.0]],
                ],
                dtype=np.float32,
            ),
            np.array([[0], [1], [2], [3]], dtype=np.int32),
        ),
        calibrateCameraCharuco=lambda **kwargs: (
            0.1,
            np.array([[10.0, 0.0, 5.0], [0.0, 20.0, 6.0], [0.0, 0.0, 1.0]]),
            np.array([0.1, 0.2, 0.3, 0.4]),
            [np.zeros((3, 1), dtype=np.float32)],
            [np.zeros((3, 1), dtype=np.float32)],
        ),
    )

    return SimpleNamespace(
        aruco=aruco,
        COLOR_BGR2GRAY=1,
        cvtColor=lambda frame, _: frame[:, :, 0],
        projectPoints=lambda **kwargs: (kwargs["objectPoints"][:, :, :2], None),
    )
