from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from opai.core.exceptions import (
    OPAIDependencyError,
    OPAIValidationError,
    OPAIWorkflowError,
)
from opai.domain.calibration import (
    CalibrationIntrinsics,
    CalibrationResult,
    CharucoBoardArtifacts,
    CharucoBoardConfig,
    validate_charuco_board_config,
)
from opai.domain.context import Context
from opai.domain.plot import plot_frames
from opai.infrastructure.persistence import (
    write_calibration_result,
    write_charuco_board_config,
    write_charuco_board_image,
)
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
    plot_result: bool = False,
    plot_nrows: int | None = None,
    plot_ncols: int | None = None,
) -> CalibrationResult:
    _validate_inputs(
        frames=frames,
        row_count=row_count,
        col_count=col_count,
        square_length=square_length,
        marker_length=marker_length,
    )

    dictionary_id = getattr(cv2.aruco, dictionary, None)
    if dictionary_id is None:
        raise OPAIValidationError(f"Unsupported ArUco dictionary: {dictionary}")
    aruco_dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    board = cv2.aruco.CharucoBoard(
        (col_count, row_count), square_length, marker_length, aruco_dictionary
    )
    charuco_detector = cv2.aruco.CharucoDetector(board)

    image_height, image_width = (int(value) for value in frames[0].shape[:2])
    image_size = (image_width, image_height)

    all_charuco_corners: list[np.ndarray] = []
    all_charuco_ids: list[np.ndarray] = []
    detected_corner_frames: list[np.ndarray] = []

    for frame in frames:
        grayscale = (
            frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        )
        charuco_corners, charuco_ids, _, _ = charuco_detector.detectBoard(grayscale)

        if charuco_ids is None or charuco_corners is None:
            continue

        if int(charuco_ids.size) < 4:
            continue

        all_charuco_corners.append(charuco_corners)
        all_charuco_ids.append(charuco_ids)
        plotted_frame = frame.copy()
        if plotted_frame.ndim == 2:
            plotted_frame = cv2.cvtColor(plotted_frame, cv2.COLOR_GRAY2BGR)
        cv2.aruco.drawDetectedCornersCharuco(
            image=plotted_frame,
            charucoCorners=charuco_corners,
            charucoIds=charuco_ids,
        )
        detected_corner_frames.append(plotted_frame)

    if not all_charuco_corners:
        raise OPAIWorkflowError(
            "Calibration failed: no frames produced enough ChArUco corners. "
            "Verify the board parameters and frame content."
        )

    if plot_result:
        try:
            plot_frames(
                detected_corner_frames,
                nrows=plot_nrows,
                ncols=plot_ncols,
                frames_are_bgr=True,
            )
        except ModuleNotFoundError as exc:
            raise OPAIDependencyError(
                "Calibration plotting requires the 'matplotlib' package. Install "
                "project dependencies before calling opai.calibrate(...) or "
                "opai.calibrate_with_video(...)."
            ) from exc
        except ValueError as exc:
            raise OPAIValidationError(str(exc)) from exc

    object_points, image_points = _build_fisheye_calibration_points(
        board=board,
        charuco_corners=all_charuco_corners,
        charuco_ids=all_charuco_ids,
    )
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = _calibrate_fisheye(
        object_points=object_points,
        image_points=image_points,
        image_size=image_size,
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


def generate_charuco_board(
    ctx: Context,
    config: CharucoBoardConfig,
) -> CharucoBoardArtifacts:
    validate_charuco_board_config(config)
    normalized_config = CharucoBoardConfig(
        dictionary=config.dictionary.strip(),
        squares_x=config.squares_x,
        squares_y=config.squares_y,
        square_length=config.square_length,
        marker_length=config.marker_length,
        image_width_px=config.image_width_px,
        image_height_px=config.image_height_px,
        margin_size_px=config.margin_size_px,
    )

    dictionary_id = getattr(cv2.aruco, normalized_config.dictionary, None)
    if dictionary_id is None:
        raise OPAIValidationError(
            f"Unsupported ArUco dictionary: {normalized_config.dictionary}"
        )
    aruco_dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    board = cv2.aruco.CharucoBoard(
        (normalized_config.squares_x, normalized_config.squares_y),
        normalized_config.square_length,
        normalized_config.marker_length,
        aruco_dictionary,
    )
    board_image = board.generateImage(
        (normalized_config.image_width_px, normalized_config.image_height_px),
        marginSize=normalized_config.margin_size_px,
    )

    image_path = write_charuco_board_image(ctx.session_directory, board_image)
    config_path = write_charuco_board_config(
        ctx.session_directory,
        normalized_config,
        board_image_path=image_path.name,
    )
    return CharucoBoardArtifacts(
        image_path=image_path,
        config_path=config_path,
        config=normalized_config,
    )


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


def _build_fisheye_calibration_points(
    board: cv2.aruco.CharucoBoard,
    charuco_corners: Sequence[np.ndarray],
    charuco_ids: Sequence[np.ndarray],
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    if len(charuco_corners) != len(charuco_ids):
        raise OPAIWorkflowError(
            "Calibration failed: inconsistent ChArUco observation lengths."
        )

    chessboard_corners = np.asarray(board.getChessboardCorners(), dtype=np.float64)
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []

    for observed_corners, observed_ids in zip(charuco_corners, charuco_ids):
        object_points.append(
            chessboard_corners[observed_ids.flatten()].reshape(-1, 1, 3)
        )
        image_points.append(
            np.asarray(observed_corners, dtype=np.float64).reshape(-1, 1, 2)
        )

    if not object_points:
        raise OPAIWorkflowError(
            "Calibration failed: no ChArUco observations available for fisheye calibration."
        )

    return tuple(object_points), tuple(image_points)


def _calibrate_fisheye(
    object_points: Sequence[np.ndarray],
    image_points: Sequence[np.ndarray],
    image_size: tuple[int, int],
) -> tuple[
    float, np.ndarray, np.ndarray, tuple[np.ndarray, ...], tuple[np.ndarray, ...]
]:
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.fisheye.calibrate(
        object_points,
        image_points,
        image_size,
        np.eye(3, dtype=np.float64),
        np.zeros((4, 1), dtype=np.float64),
        flags=(
            cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
            | cv2.fisheye.CALIB_CHECK_COND
            | cv2.fisheye.CALIB_FIX_SKEW
        ),
        criteria=(
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            100,
            1e-6,
        ),
    )
    return (
        float(ret),
        np.asarray(camera_matrix, dtype=np.float64),
        np.asarray(dist_coeffs, dtype=np.float64),
        tuple(rvecs),
        tuple(tvecs),
    )


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
        projected_corners, _ = cv2.fisheye.projectPoints(
            object_points,
            rvec,
            tvec,
            camera_matrix,
            dist_coeffs,
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
