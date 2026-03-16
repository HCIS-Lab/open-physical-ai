from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from opai.application import calibration as calibration_module
from opai.application.calibration import (
    _build_intrinsics,
    _compute_mse_reprojection_error,
    _resolve_dictionary,
    calibrate,
)
from opai.core.exceptions import OPAIValidationError, OPAIWorkflowError
from opai.domain.context import Context


class DummyBoard:
    def __init__(self, corners: np.ndarray) -> None:
        self._corners = corners

    def getChessboardCorners(self) -> np.ndarray:
        return self._corners


@pytest.fixture
def fake_cv2(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    fake = SimpleNamespace(
        aruco=SimpleNamespace(
            DICT_4X4_50=1,
            getPredefinedDictionary=lambda dictionary_id: {"id": dictionary_id},
        ),
        projectPoints=lambda **kwargs: (kwargs["objectPoints"][:, :, :2], None),
        COLOR_BGR2GRAY=1,
        cvtColor=lambda frame, _: frame[:, :, 0],
    )
    monkeypatch.setattr(calibration_module, "cv2", fake)
    return fake


def test_resolve_dictionary_rejects_unknown_name(fake_cv2: SimpleNamespace) -> None:
    with pytest.raises(OPAIValidationError, match="Unsupported ArUco dictionary"):
        _resolve_dictionary("DICT_DOES_NOT_EXIST")


def test_calibrate_rejects_invalid_frames(tmp_path, fake_cv2: SimpleNamespace) -> None:
    ctx = Context(name="session", session_directory=tmp_path)
    frame_a = np.zeros((10, 10, 3), dtype=np.uint8)
    frame_b = np.zeros((11, 10, 3), dtype=np.uint8)

    with pytest.raises(OPAIValidationError, match="identical image dimensions"):
        calibrate(ctx, [frame_a, frame_b], 3, 3, 1.0, 0.5, "DICT_4X4_50")


def test_build_intrinsics_zero_fills_missing_distortion() -> None:
    camera_matrix = np.array([[10.0, 1.5, 5.0], [0.0, 20.0, 6.0], [0.0, 0.0, 1.0]])
    dist_coeffs = np.array([0.1, 0.2])

    intrinsics = _build_intrinsics(camera_matrix, dist_coeffs)

    assert intrinsics.aspect_ratio == 0.5
    assert intrinsics.radial_distortion_3 == 0.0
    assert intrinsics.radial_distortion_4 == 0.0
    assert intrinsics.skew == 1.5


def test_compute_mse_reprojection_error_averages_squared_pixel_error(
    monkeypatch,
) -> None:
    board = DummyBoard(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
    )
    observed_corners = [np.array([[[1.0, 1.0]], [[4.0, 4.0]]], dtype=np.float32)]
    observed_ids = [np.array([[0], [1]], dtype=np.int32)]

    def fake_project_points(**_: np.ndarray) -> tuple[np.ndarray, None]:
        return np.array([[[2.0, 3.0]], [[5.0, 6.0]]], dtype=np.float32), None

    monkeypatch.setattr(
        calibration_module,
        "cv2",
        SimpleNamespace(projectPoints=fake_project_points),
    )

    mse = _compute_mse_reprojection_error(
        board=board,
        charuco_corners=observed_corners,
        charuco_ids=observed_ids,
        rvecs=[np.zeros((3, 1))],
        tvecs=[np.zeros((3, 1))],
        camera_matrix=np.eye(3),
        dist_coeffs=np.zeros(4),
    )

    assert mse == pytest.approx(5.0)


def test_compute_mse_reprojection_error_rejects_inconsistent_lengths() -> None:
    board = DummyBoard(np.array([[0.0, 0.0, 0.0]], dtype=np.float32))

    with pytest.raises(
        OPAIWorkflowError, match="inconsistent reprojection input lengths"
    ):
        _compute_mse_reprojection_error(
            board=board,
            charuco_corners=[np.array([[[1.0, 1.0]]], dtype=np.float32)],
            charuco_ids=[np.array([[0]], dtype=np.int32)],
            rvecs=[],
            tvecs=[],
            camera_matrix=np.eye(3),
            dist_coeffs=np.zeros(4),
        )


def test_repo_exceptions_expose_error_codes_and_payload() -> None:
    error = OPAIValidationError(
        "Invalid board parameters.",
        details={"field": "row_count"},
    )

    assert error.error_code == "validation_error"
    assert error.to_dict() == {
        "error_code": "validation_error",
        "message": "Invalid board parameters.",
        "details": {"field": "row_count"},
    }

    overridden = OPAIWorkflowError(
        "Calibration failed.",
        error_code="charuco_calibration_failed",
    )

    assert overridden.error_code == "charuco_calibration_failed"
