from __future__ import annotations

import json
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
    CalibrationVerificationFrame,
    CalibrationVerificationResult,
    CharucoBoardArtifacts,
    CharucoBoardConfig,
    validate_charuco_board_config,
)
from opai.domain.context import Context
from opai.domain.plot import plot_frames
from opai.infrastructure.persistence import (
    write_calibration_result,
    write_calibration_verification_result,
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


def verify_calibrated_parameters(
    ctx: Context,
    video_path: str | Path,
    n_check_imgs: int,
    charuco_config_json: str | Path | dict[str, object],
    intrinsics_json: str | Path | dict[str, object],
    plot_result: bool = False,
    plot_nrows: int | None = None,
    plot_ncols: int | None = None,
) -> CalibrationVerificationResult:
    if not isinstance(n_check_imgs, int) or isinstance(n_check_imgs, bool):
        raise OPAIValidationError(
            "n_check_imgs must be an integer greater than 0.",
            details={"n_check_imgs": str(n_check_imgs)},
        )
    if n_check_imgs <= 0:
        raise OPAIValidationError(
            "n_check_imgs must be greater than 0.",
            details={"n_check_imgs": n_check_imgs},
        )

    all_frames = sample_video_frames(video_path=video_path, frame_sample_step=10)
    check_image_count = min(n_check_imgs, len(all_frames))
    sampled_frame_indices = tuple(
        int(index)
        for index in np.linspace(
            0,
            len(all_frames) - 1,
            num=check_image_count,
            dtype=int,
        )
    )
    sampled_frames = tuple(all_frames[index] for index in sampled_frame_indices)

    charuco_payload = _load_json_payload(
        ctx=ctx,
        payload_or_path=charuco_config_json,
        payload_name="charuco_config_json",
    )
    intrinsics_payload = _load_json_payload(
        ctx=ctx,
        payload_or_path=intrinsics_json,
        payload_name="intrinsics_json",
    )

    charuco_config = _build_charuco_board_config_from_payload(charuco_payload)
    validate_charuco_board_config(charuco_config)
    camera_matrix, dist_coeffs, intrinsics_image_size = (
        _build_fisheye_parameters_from_payload(intrinsics_payload)
    )

    image_height, image_width = (int(value) for value in sampled_frames[0].shape[:2])
    for frame in sampled_frames[1:]:
        if frame.shape[:2] != (image_height, image_width):
            raise OPAIWorkflowError(
                "Intrinsics verification failed: sampled frames have inconsistent image dimensions."
            )

    if (
        intrinsics_image_size is not None
        and (
            image_width,
            image_height,
        )
        != intrinsics_image_size
    ):
        raise OPAIValidationError(
            "Calibration intrinsics image size does not match the verification video frames.",
            details={
                "intrinsics_image_width": intrinsics_image_size[0],
                "intrinsics_image_height": intrinsics_image_size[1],
                "video_image_width": image_width,
                "video_image_height": image_height,
            },
        )

    dictionary_id = getattr(cv2.aruco, charuco_config.dictionary, None)
    if dictionary_id is None:
        raise OPAIValidationError(
            f"Unsupported ArUco dictionary: {charuco_config.dictionary}"
        )

    aruco_dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    board = cv2.aruco.CharucoBoard(
        (charuco_config.squares_x, charuco_config.squares_y),
        charuco_config.square_length,
        charuco_config.marker_length,
        aruco_dictionary,
    )
    charuco_detector = cv2.aruco.CharucoDetector(board)

    board_points = np.asarray(board.getChessboardCorners(), dtype=np.float64)
    frame_results: list[CalibrationVerificationFrame] = []
    verification_frames: list[np.ndarray] = []
    total_squared_error_sum = 0.0
    total_corner_count = 0

    for sampled_frame_index, frame in zip(sampled_frame_indices, sampled_frames):
        grayscale = (
            frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        )
        charuco_corners, charuco_ids, _, _ = charuco_detector.detectBoard(grayscale)
        if charuco_ids is None or charuco_corners is None:
            continue

        observed_ids = np.asarray(charuco_ids, dtype=np.int32).reshape(-1)
        if observed_ids.size < 4:
            continue

        if np.any(observed_ids < 0) or np.any(observed_ids >= board_points.shape[0]):
            raise OPAIWorkflowError(
                "Intrinsics verification failed: detected ChArUco ids are outside the configured board range."
            )

        object_points = board_points[observed_ids].reshape(-1, 1, 3)
        image_points = np.asarray(charuco_corners, dtype=np.float64).reshape(-1, 1, 2)

        pose_found, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            camera_matrix,
            dist_coeffs,
        )
        if not pose_found:
            continue

        reprojected_points, _ = cv2.fisheye.projectPoints(
            object_points,
            rvec,
            tvec,
            camera_matrix,
            dist_coeffs,
        )
        detected_points = image_points.reshape(-1, 2)
        projected_points = reprojected_points.reshape(-1, 2)
        deltas = projected_points - detected_points
        squared_error_sum = float(np.sum(deltas * deltas))
        corner_count = int(deltas.shape[0])
        total_squared_error_sum += squared_error_sum
        total_corner_count += corner_count
        verification_frames.append(
            _draw_calibration_verification_overlay(
                frame=frame,
                detected_points=detected_points,
                reprojected_points=projected_points,
            )
        )
        frame_results.append(
            CalibrationVerificationFrame(
                sampled_frame_index=int(sampled_frame_index),
                detected_corner_count=corner_count,
                mse_reproj_error=squared_error_sum / corner_count,
            )
        )

    if not frame_results or total_corner_count == 0:
        raise OPAIWorkflowError(
            "Intrinsics verification failed: no sampled frames produced a valid ChArUco pose."
        )

    if plot_result:
        try:
            plot_frames(
                verification_frames,
                nrows=plot_nrows,
                ncols=plot_ncols,
                frames_are_bgr=True,
            )
        except ValueError as exc:
            raise OPAIValidationError(str(exc)) from exc

    result = CalibrationVerificationResult(
        requested_check_image_count=int(n_check_imgs),
        sampled_image_count=int(check_image_count),
        verified_image_count=len(frame_results),
        skipped_image_count=int(check_image_count - len(frame_results)),
        total_detected_corner_count=total_corner_count,
        mse_reproj_error=total_squared_error_sum / total_corner_count,
        frame_results=tuple(frame_results),
    )
    write_calibration_verification_result(ctx.session_directory, result)
    return result


def _draw_calibration_verification_overlay(
    frame: np.ndarray,
    detected_points: np.ndarray,
    reprojected_points: np.ndarray,
) -> np.ndarray:
    overlay = np.ascontiguousarray(frame.copy())
    if overlay.ndim == 2:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)

    for detected_point, reprojected_point in zip(detected_points, reprojected_points):
        detected_pixel = tuple(int(round(value)) for value in detected_point)
        reprojected_pixel = tuple(int(round(value)) for value in reprojected_point)
        cv2.arrowedLine(
            overlay,
            detected_pixel,
            reprojected_pixel,
            (0, 255, 255),
            1,
            tipLength=0.2,
        )
        cv2.circle(
            overlay,
            detected_pixel,
            5,
            (0, 255, 0),
            2,
        )
        cv2.circle(
            overlay,
            reprojected_pixel,
            3,
            (0, 0, 255),
            -1,
        )
    return overlay


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


def _load_json_payload(
    ctx: Context,
    payload_or_path: str | Path | dict[str, object],
    payload_name: str,
) -> dict[str, object]:
    if isinstance(payload_or_path, dict):
        return payload_or_path

    if not isinstance(payload_or_path, (str, Path)):
        raise OPAIValidationError(
            f"{payload_name} must be a dict payload or a JSON file path.",
            details={"payload_name": payload_name},
        )

    raw_path = Path(payload_or_path).expanduser()
    candidate_paths = (
        (raw_path,)
        if raw_path.is_absolute()
        else (ctx.session_directory / raw_path, raw_path)
    )

    resolved_path: Path | None = None
    for candidate_path in candidate_paths:
        if candidate_path.exists():
            resolved_path = candidate_path
            break

    if resolved_path is None:
        raise OPAIValidationError(
            f"{payload_name} JSON file was not found.",
            details={
                "payload_name": payload_name,
                "path": str(raw_path),
            },
        )

    try:
        payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise OPAIValidationError(
            f"{payload_name} must be valid JSON.",
            details={
                "payload_name": payload_name,
                "path": str(resolved_path),
            },
        ) from exc
    if not isinstance(payload, dict):
        raise OPAIValidationError(
            f"{payload_name} must decode to a JSON object.",
            details={
                "payload_name": payload_name,
                "path": str(resolved_path),
            },
        )
    return payload


def _build_charuco_board_config_from_payload(
    payload: dict[str, object],
) -> CharucoBoardConfig:
    dictionary = payload.get("dictionary")
    if not isinstance(dictionary, str):
        raise OPAIValidationError(
            "charuco_config_json is missing a valid dictionary field.",
            details={"field": "dictionary"},
        )

    try:
        return CharucoBoardConfig(
            dictionary=dictionary.strip(),
            squares_x=int(payload["squares_x"]),
            squares_y=int(payload["squares_y"]),
            square_length=float(payload["square_length"]),
            marker_length=float(payload["marker_length"]),
            image_width_px=int(payload["image_width_px"]),
            image_height_px=int(payload["image_height_px"]),
            margin_size_px=int(payload["margin_size_px"]),
        )
    except KeyError as exc:
        raise OPAIValidationError(
            "charuco_config_json is missing required ChArUco fields.",
            details={"field": str(exc)},
        ) from exc
    except (TypeError, ValueError) as exc:
        raise OPAIValidationError(
            "charuco_config_json contains invalid ChArUco field values.",
            details={"error": str(exc)},
        ) from exc


def _build_fisheye_parameters_from_payload(
    payload: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, tuple[int, int] | None]:
    intrinsics_payload = payload.get("intrinsics")
    if not isinstance(intrinsics_payload, dict):
        raise OPAIValidationError(
            "intrinsics_json must include an intrinsics object.",
            details={"field": "intrinsics"},
        )

    try:
        focal_length = float(intrinsics_payload["focal_length"])
        aspect_ratio = float(intrinsics_payload["aspect_ratio"])
        principal_pt_x = float(intrinsics_payload["principal_pt_x"])
        principal_pt_y = float(intrinsics_payload["principal_pt_y"])
        skew = float(intrinsics_payload["skew"])
        radial_distortion_1 = float(intrinsics_payload["radial_distortion_1"])
        radial_distortion_2 = float(intrinsics_payload["radial_distortion_2"])
        radial_distortion_3 = float(intrinsics_payload["radial_distortion_3"])
        radial_distortion_4 = float(intrinsics_payload["radial_distortion_4"])
    except KeyError as exc:
        raise OPAIValidationError(
            "intrinsics_json is missing required intrinsic fields.",
            details={"field": str(exc)},
        ) from exc
    except (TypeError, ValueError) as exc:
        raise OPAIValidationError(
            "intrinsics_json contains invalid intrinsic field values.",
            details={"error": str(exc)},
        ) from exc

    values_to_check = [
        focal_length,
        aspect_ratio,
        principal_pt_x,
        principal_pt_y,
        skew,
        radial_distortion_1,
        radial_distortion_2,
        radial_distortion_3,
        radial_distortion_4,
    ]
    if not all(np.isfinite(value) for value in values_to_check):
        raise OPAIValidationError("intrinsics_json contains non-finite numeric values.")
    if focal_length <= 0.0:
        raise OPAIValidationError("intrinsics focal_length must be positive.")
    if aspect_ratio <= 0.0:
        raise OPAIValidationError("intrinsics aspect_ratio must be positive.")

    focal_length_y = focal_length / aspect_ratio
    camera_matrix = np.array(
        [
            [focal_length, skew, principal_pt_x],
            [0.0, focal_length_y, principal_pt_y],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    dist_coeffs = np.array(
        [
            [radial_distortion_1],
            [radial_distortion_2],
            [radial_distortion_3],
            [radial_distortion_4],
        ],
        dtype=np.float64,
    )

    if "image_width" not in payload or "image_height" not in payload:
        return camera_matrix, dist_coeffs, None

    try:
        image_width = int(payload["image_width"])
        image_height = int(payload["image_height"])
    except (TypeError, ValueError) as exc:
        raise OPAIValidationError(
            "intrinsics_json image_width and image_height must be integers.",
            details={"error": str(exc)},
        ) from exc
    if image_width <= 0 or image_height <= 0:
        raise OPAIValidationError(
            "intrinsics_json image_width and image_height must be positive."
        )
    return camera_matrix, dist_coeffs, (image_width, image_height)


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
