from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from opai.application import calibration as calibration_module
from opai.application.calibration import (
    _build_fisheye_calibration_points,
    _build_intrinsics,
    _compute_mse_reprojection_error,
    calibrate,
    sample_video_frames,
)
from opai.core.exceptions import OPAIValidationError, OPAIWorkflowError
from opai.domain.context import Context
from opai.domain.plot import get_plot_grid
from opai.infrastructure import video as video_module


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
        fisheye=SimpleNamespace(
            CALIB_RECOMPUTE_EXTRINSIC=1,
            CALIB_CHECK_COND=2,
            CALIB_FIX_SKEW=4,
            calibrate=lambda *args, **kwargs: (
                0.1,
                np.eye(3),
                np.zeros((4, 1)),
                (),
                (),
            ),
            projectPoints=lambda object_points, *_args: (
                object_points[:, :, :2],
                None,
            ),
        ),
        TERM_CRITERIA_EPS=1,
        TERM_CRITERIA_MAX_ITER=2,
        COLOR_BGR2GRAY=1,
        cvtColor=lambda frame, _: frame[:, :, 0],
    )
    monkeypatch.setattr(calibration_module, "cv2", fake)
    return fake


def test_calibrate_rejects_unknown_dictionary(
    tmp_path,
    fake_cv2: SimpleNamespace,
) -> None:
    ctx = Context(name="session", session_directory=tmp_path)
    frame = np.zeros((10, 10, 3), dtype=np.uint8)

    with pytest.raises(OPAIValidationError, match="Unsupported ArUco dictionary"):
        calibrate(ctx, [frame], 3, 3, 1.0, 0.5, "DICT_DOES_NOT_EXIST")


def test_calibrate_rejects_invalid_frames(tmp_path, fake_cv2: SimpleNamespace) -> None:
    ctx = Context(name="session", session_directory=tmp_path)
    frame_a = np.zeros((10, 10, 3), dtype=np.uint8)
    frame_b = np.zeros((11, 10, 3), dtype=np.uint8)

    with pytest.raises(OPAIValidationError, match="identical image dimensions"):
        calibrate(ctx, [frame_a, frame_b], 3, 3, 1.0, 0.5, "DICT_4X4_50")


def test_build_fisheye_calibration_points_maps_ids_to_board_coordinates() -> None:
    board = DummyBoard(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
    )
    charuco_corners = [
        np.array(
            [
                [[10.0, 20.0]],
                [[30.0, 40.0]],
            ],
            dtype=np.float32,
        )
    ]
    charuco_ids = [np.array([[2], [0]], dtype=np.int32)]

    object_points, image_points = _build_fisheye_calibration_points(
        board=board,
        charuco_corners=charuco_corners,
        charuco_ids=charuco_ids,
    )

    assert np.array_equal(
        object_points[0],
        np.array(
            [
                [[0.0, 1.0, 0.0]],
                [[0.0, 0.0, 0.0]],
            ],
            dtype=np.float64,
        ),
    )
    assert np.array_equal(
        image_points[0],
        np.array(
            [
                [[10.0, 20.0]],
                [[30.0, 40.0]],
            ],
            dtype=np.float64,
        ),
    )


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

    def fake_project_points(
        object_points: np.ndarray,
        _rvec: np.ndarray,
        _tvec: np.ndarray,
        _camera_matrix: np.ndarray,
        _dist_coeffs: np.ndarray,
    ) -> tuple[np.ndarray, None]:
        return np.array([[[2.0, 3.0]], [[5.0, 6.0]]], dtype=np.float32), None

    monkeypatch.setattr(
        calibration_module,
        "cv2",
        SimpleNamespace(fisheye=SimpleNamespace(projectPoints=fake_project_points)),
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


def test_get_plot_grid_auto_computes_near_square_layout() -> None:
    grid = get_plot_grid(5)

    assert grid.item_count == 5
    assert grid.nrows == 2
    assert grid.ncols == 3


def test_sample_video_frames_rejects_non_positive_step(tmp_path) -> None:
    video_path = tmp_path / "demo.mp4"
    video_path.write_bytes(b"demo")

    with pytest.raises(OPAIValidationError, match="frame_sample_step"):
        sample_video_frames(video_path, 0)


def test_sample_video_frames_rejects_missing_path(tmp_path) -> None:
    with pytest.raises(OPAIValidationError, match="does not exist"):
        sample_video_frames(tmp_path / "missing.mp4", 2)


def test_sample_video_frames_rejects_empty_sampling(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video_path = tmp_path / "demo.mp4"
    video_path.write_bytes(b"demo")
    monkeypatch.setattr(
        calibration_module, "sample_video_frames_from_path", lambda *_: ()
    )

    with pytest.raises(OPAIWorkflowError, match="produced no frames"):
        sample_video_frames(video_path, 2)


def test_infrastructure_video_sampling_starts_at_zero_and_respects_step(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    frames = tuple(
        np.full((2, 2, 3), fill_value=index, dtype=np.uint8) for index in range(5)
    )

    class FakeCapture:
        def __init__(self, _path: str) -> None:
            self._frames = list(frames)
            self._released = False

        def isOpened(self) -> bool:
            return True

        def read(self) -> tuple[bool, np.ndarray | None]:
            if not self._frames:
                return False, None
            return True, self._frames.pop(0)

        def release(self) -> None:
            self._released = True

    monkeypatch.setattr(
        video_module,
        "cv2",
        SimpleNamespace(VideoCapture=FakeCapture),
    )

    sampled = video_module.sample_video_frames(tmp_path / "demo.mp4", 2)

    assert [int(frame[0, 0, 0]) for frame in sampled] == [0, 2, 4]


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
