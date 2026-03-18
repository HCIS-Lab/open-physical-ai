from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from opai.core.exceptions import OPAIValidationError, OPAIWorkflowError
from opai.domain.calibration import CalibrationIntrinsics, CalibrationResult
from opai.domain.context import Context
from opai.infrastructure.persistence import write_calibration_result
from opai.infrastructure.video import (
    sample_video_frames as sample_video_frames_from_path,
)


def calibrate(
    ctx: Context,
    frames: Sequence[np.ndarray],
    row_count: int,
    col_count: int,
    square_length: float,
    marker_length: float,
    dictionary: str,
) -> CalibrationResult:
    _validate_inputs(
        frames=frames,
        row_count=row_count,
        col_count=col_count,
        square_length=square_length,
        marker_length=marker_length,
    )

    aruco_dictionary = _resolve_dictionary(dictionary)
    board = cv2.aruco.CharucoBoard(
        (col_count, row_count),
        square_length,
        marker_length,
        aruco_dictionary,
    )

    image_height, image_width = _get_frame_size(frames)
    image_size = (image_width, image_height)

    all_charuco_corners: list[np.ndarray] = []
    all_charuco_ids: list[np.ndarray] = []

    for frame in frames:
        grayscale = _to_grayscale(frame)
        corners, ids, _ = cv2.aruco.detectMarkers(grayscale, aruco_dictionary)
        if ids is None or len(ids) == 0:
            continue

        _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            markerCorners=corners,
            markerIds=ids,
            image=grayscale,
            board=board,
        )
        if charuco_ids is None or charuco_corners is None or len(charuco_ids) < 4:
            continue

        all_charuco_corners.append(charuco_corners)
        all_charuco_ids.append(charuco_ids)

    if not all_charuco_corners:
        raise OPAIWorkflowError(
            "Calibration failed: no frames produced enough ChArUco corners. "
            "Verify the board parameters and frame content."
        )

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        charucoCorners=all_charuco_corners,
        charucoIds=all_charuco_ids,
        board=board,
        imageSize=image_size,
        cameraMatrix=None,
        distCoeffs=None,
    )

    mse_reproj_error = _compute_mse_reprojection_error(
        board=board,
        charuco_corners=all_charuco_corners,
        charuco_ids=all_charuco_ids,
        rvecs=rvecs,
        tvecs=tvecs,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
    )

    result = CalibrationResult(
        mse_reproj_error=float(
            mse_reproj_error if np.isfinite(mse_reproj_error) else ret
        ),
        image_height=image_height,
        image_width=image_width,
        intrinsic_type="FISHEYE",
        intrinsics=_build_intrinsics(camera_matrix, dist_coeffs),
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
    )
    write_calibration_result(ctx.session_directory, result)
    return result


def sample_video_frames(
    video_path: str | Path,
    frame_sample_step: int,
) -> tuple[np.ndarray, ...]:
    path = Path(video_path).expanduser()
    if not path.exists():
        raise OPAIValidationError(
            f"Calibration video does not exist: {path}",
            details={"path": str(path)},
        )
    if not path.is_file():
        raise OPAIValidationError(
            f"Calibration video path must point to a file: {path}",
            details={"path": str(path)},
        )
    if frame_sample_step <= 0:
        raise OPAIValidationError(
            "frame_sample_step must be greater than 0.",
            details={"frame_sample_step": frame_sample_step},
        )

    frames = sample_video_frames_from_path(path, frame_sample_step)
    if not frames:
        raise OPAIWorkflowError(
            "Calibration failed: video sampling produced no frames.",
            details={
                "path": str(path),
                "frame_sample_step": frame_sample_step,
            },
        )
    return frames


def _validate_inputs(
    frames: Sequence[np.ndarray],
    row_count: int,
    col_count: int,
    square_length: float,
    marker_length: float,
) -> None:
    if not frames:
        raise OPAIValidationError("Calibration requires a non-empty list of frames.")
    if row_count <= 1 or col_count <= 1:
        raise OPAIValidationError(
            "row_count and col_count must both be greater than 1."
        )
    if square_length <= 0 or marker_length <= 0:
        raise OPAIValidationError("square_length and marker_length must be positive.")
    if marker_length >= square_length:
        raise OPAIValidationError("marker_length must be smaller than square_length.")

    first_shape = frames[0].shape[:2]
    for frame in frames:
        if frame.shape[:2] != first_shape:
            raise OPAIValidationError(
                "All frames must have identical image dimensions."
            )


def _resolve_dictionary(name: str) -> cv2.aruco.Dictionary:
    dictionary_id = getattr(cv2.aruco, name, None)
    if dictionary_id is None:
        raise OPAIValidationError(f"Unsupported ArUco dictionary: {name}")
    return cv2.aruco.getPredefinedDictionary(dictionary_id)


def _get_frame_size(frames: Sequence[np.ndarray]) -> tuple[int, int]:
    height, width = frames[0].shape[:2]
    return int(height), int(width)


def _to_grayscale(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def _compute_mse_reprojection_error(
    board: cv2.aruco.CharucoBoard,
    charuco_corners: Sequence[np.ndarray],
    charuco_ids: Sequence[np.ndarray],
    rvecs: Sequence[np.ndarray],
    tvecs: Sequence[np.ndarray],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> float:
    chessboard_corners = board.getChessboardCorners()
    squared_error_sum = 0.0
    point_count = 0

    if not (len(charuco_corners) == len(charuco_ids) == len(rvecs) == len(tvecs)):
        raise OPAIWorkflowError(
            "Calibration failed: inconsistent reprojection input lengths."
        )

    for observed_corners, observed_ids, rvec, tvec in zip(
        charuco_corners,
        charuco_ids,
        rvecs,
        tvecs,
    ):
        object_points = chessboard_corners[observed_ids.flatten()].reshape(-1, 1, 3)
        projected_corners, _ = cv2.projectPoints(
            objectPoints=object_points,
            rvec=rvec,
            tvec=tvec,
            cameraMatrix=camera_matrix,
            distCoeffs=dist_coeffs,
        )
        deltas = projected_corners.reshape(-1, 2) - observed_corners.reshape(-1, 2)
        squared_error_sum += float(np.sum(deltas * deltas))
        point_count += int(deltas.shape[0])

    if point_count == 0:
        raise OPAIWorkflowError(
            "Calibration failed: unable to compute reprojection error."
        )
    return squared_error_sum / point_count


def _build_intrinsics(
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> CalibrationIntrinsics:
    distortion = np.ravel(dist_coeffs).astype(float)
    radial = np.pad(distortion, (0, max(0, 4 - distortion.size)), constant_values=0.0)
    fx = float(camera_matrix[0, 0])
    fy = float(camera_matrix[1, 1])

    return CalibrationIntrinsics(
        aspect_ratio=fx / fy if fy else 0.0,
        focal_length=fx,
        principal_pt_x=float(camera_matrix[0, 2]),
        principal_pt_y=float(camera_matrix[1, 2]),
        radial_distortion_1=float(radial[0]),
        radial_distortion_2=float(radial[1]),
        radial_distortion_3=float(radial[2]),
        radial_distortion_4=float(radial[3]),
        skew=float(camera_matrix[0, 1]),
    )
