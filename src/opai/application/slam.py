from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np

from opai.application.imu import extract_imu_from_video
from opai.core.exceptions import OPAIWorkflowError
from opai.domain.context import Context
from opai.domain.slam import MappingRunResult
from opai.infrastructure.context_store import get_demo_videos, get_mapping_video
from opai.infrastructure.docker import (
    ALLOWED_RETURN_CODES,
    pull_docker_image,
    run_mapping_container,
    run_trajectory_extraction_container,
)
from opai.infrastructure.logger import get_logger
from opai.infrastructure.video import (
    convert_video_to_fps,
    get_video_duration_seconds,
    get_video_fps,
    get_video_resolution,
)

DEFAULT_SLAM_SETTING_FILENAME = "gopro13_fisheye_ratio_4-3_2-7k.yaml"
DEFAULT_RESOLUTION = (2704, 2028)  # (width, height)
DEFAULT_DOCKER_IMAGE = "authoree13/orb-slam3:latest"
TARGET_SLAM_FPS = 60
HIGH_SPEED_SOURCE_FPS = 120.0
FPS_TOLERANCE = 1.0
PREPARED_MAPPING_VIDEO_FILENAME = "mapping_input_60fps.mp4"
PREPARED_TRAJECTORY_VIDEO_FILENAME = "trajectory_input_60fps.mp4"
DEFAULT_TRAJECTORY_TIMEOUT_MULTIPLE = 10.0

logger = get_logger(__name__)


def run_mapping(ctx: Context, slam_setting_path: str) -> MappingRunResult:
    logger.info("Starting SLAM mapping for session %s", ctx.name)
    original_video_path = get_mapping_video(ctx)
    if original_video_path is None:
        raise OPAIWorkflowError(
            "No mapping video found. Call opai.add_mapping_video(...) or opai.add_mapping(...) before opai.run_mapping(...)."
        )

    work_directory = ctx.session_directory / "slam" / "mapping"
    work_directory.mkdir(parents=True, exist_ok=True)
    _clear_mapping_outputs(work_directory)

    prepared_video_path = original_video_path
    source_fps = get_video_fps(original_video_path)
    if abs(source_fps - HIGH_SPEED_SOURCE_FPS) < FPS_TOLERANCE:
        logger.info(
            "Converting mapping video from %.1f fps to %s fps",
            source_fps,
            TARGET_SLAM_FPS,
        )
        prepared_video_path = convert_video_to_fps(
            original_video_path,
            work_directory / PREPARED_MAPPING_VIDEO_FILENAME,
            TARGET_SLAM_FPS,
        )
    else:
        logger.info(
            "Using mapping video without FPS conversion: %s", prepared_video_path
        )

    video_resolution = get_video_resolution(prepared_video_path)
    logger.info(
        "Prepared mapping video resolution is %sx%s",
        video_resolution[0],
        video_resolution[1],
    )
    if video_resolution != DEFAULT_RESOLUTION:
        raise OPAIWorkflowError(
            "Mapping requires a "
            f"{DEFAULT_RESOLUTION[0]}x{DEFAULT_RESOLUTION[1]} resolution video. "
            f"Video resolution is {video_resolution}."
        )

    imu_json_path = extract_imu_from_video(
        original_video_path, work_directory / "imu_data.json"
    )
    logger.info("Using IMU payload at %s", imu_json_path)
    slam_mask = np.zeros((DEFAULT_RESOLUTION[1], DEFAULT_RESOLUTION[0]), dtype=np.uint8)
    slam_mask = draw_predefined_mask(
        slam_mask, color=255, mirror=True, gripper=False, finger=True
    )
    mask_path = work_directory / "slam_mask.png"
    cv2.imwrite(str(mask_path), slam_mask)
    logger.info("Wrote SLAM mask to %s", mask_path)

    map_path = work_directory / "map_atlas.osa"
    trajectory_csv_path = work_directory / "mapping_camera_trajectory.csv"
    stdout_log_path = work_directory / "slam_stdout.txt"
    stderr_log_path = work_directory / "slam_stderr.txt"

    logger.info("Pulling SLAM Docker image %s", DEFAULT_DOCKER_IMAGE)
    pull_docker_image(DEFAULT_DOCKER_IMAGE)
    logger.info("Running SLAM container with settings file %s", slam_setting_path)
    run_mapping_container(
        docker_image=DEFAULT_DOCKER_IMAGE,
        prepared_video_path=prepared_video_path,
        work_directory=work_directory,
        slam_setting_path=slam_setting_path,
        stdout_path=stdout_log_path,
        stderr_path=stderr_log_path,
        enable_gui=True,
    )

    logger.info("Completed SLAM mapping for session %s", ctx.name)
    return MappingRunResult(
        input_video_path=original_video_path,
        imu_json_path=imu_json_path,
        mask_path=mask_path,
        map_path=map_path,
        trajectory_csv_path=trajectory_csv_path,
        stdout_log_path=stdout_log_path,
        stderr_log_path=stderr_log_path,
    )


def run_extract_trajectories_batch(ctx: Context) -> dict[str, object]:
    atlas_path = ctx.session_directory / "slam" / "mapping" / "map_atlas.osa"
    if not atlas_path.is_file():
        raise OPAIWorkflowError(
            "Mapping trajectory extraction requires a map atlas file. Call opai.run_mapping(...) before calling opai.run_extract_trajectories_batch(...)."
        )
    slam_setting_path = ctx.session_directory / "slam_settings.yaml"
    if not slam_setting_path.is_file():
        raise OPAIWorkflowError(
            "Trajectory extraction requires a session slam_settings.yaml file. Reinitialize the session with opai.init(...) before calling opai.run_extract_trajectories_batch(...)."
        )

    demo_video_paths = get_demo_videos(ctx)
    if not demo_video_paths:
        logger.info(
            "No demo videos registered for session %s. Nothing to extract.",
            ctx.name,
        )
        return {
            "processed_videos": [],
            "total_processed": 0,
        }

    logger.info(
        "Starting trajectory extraction batch for session %s with %s demo videos",
        ctx.name,
        len(demo_video_paths),
    )
    pull_docker_image(DEFAULT_DOCKER_IMAGE)
    processed_videos: list[dict] = []

    for demo_video_path in demo_video_paths:
        demo_directory = demo_video_path.parent
        trajectory_csv_path = demo_directory / "camera_trajectory.csv"
        stdout_log_path = demo_directory / "slam_stdout.txt"
        stderr_log_path = demo_directory / "slam_stderr.txt"
        imu_json_path = demo_directory / "imu_data.json"
        mask_path = demo_directory / "slam_mask.png"
        prepared_video_path = demo_video_path

        if trajectory_csv_path.is_file():
            logger.warning(
                "camera_trajectory.csv already exists, skipping %s",
                demo_directory.name,
            )
            continue

        logger.info(
            "Preparing trajectory extraction inputs for %s", demo_directory.name
        )

        try:
            source_fps = get_video_fps(demo_video_path)
            if abs(source_fps - HIGH_SPEED_SOURCE_FPS) < FPS_TOLERANCE:
                logger.info(
                    "Converting demo video %s from %.1f fps to %s fps",
                    demo_video_path,
                    source_fps,
                    TARGET_SLAM_FPS,
                )
                prepared_video_path = convert_video_to_fps(
                    demo_video_path,
                    demo_directory / PREPARED_TRAJECTORY_VIDEO_FILENAME,
                    TARGET_SLAM_FPS,
                )
            else:
                logger.info(
                    "Using demo video without FPS conversion: %s",
                    prepared_video_path,
                )

            video_resolution = get_video_resolution(prepared_video_path)
            if video_resolution != DEFAULT_RESOLUTION:
                raise OPAIWorkflowError(
                    "Trajectory extraction requires a "
                    f"{DEFAULT_RESOLUTION[0]}x{DEFAULT_RESOLUTION[1]} resolution video. "
                    f"Video resolution is {video_resolution}.",
                    details={
                        "video_path": str(prepared_video_path),
                        "demo_directory": str(demo_directory),
                    },
                )

            extract_imu_from_video(demo_video_path, imu_json_path)
            slam_mask = np.zeros(
                (DEFAULT_RESOLUTION[1], DEFAULT_RESOLUTION[0]),
                dtype=np.uint8,
            )
            slam_mask = draw_predefined_mask(
                slam_mask, color=255, mirror=True, gripper=False, finger=True
            )
            wrote_mask = cv2.imwrite(str(mask_path), slam_mask)
            if not wrote_mask:
                raise OPAIWorkflowError(
                    "Failed to write the SLAM mask image.",
                    details={"path": str(mask_path)},
                )

            duration_seconds = get_video_duration_seconds(prepared_video_path)
            timeout_seconds = None
            if duration_seconds > 0:
                timeout_seconds = duration_seconds * DEFAULT_TRAJECTORY_TIMEOUT_MULTIPLE

            result = run_trajectory_extraction_container(
                docker_image=DEFAULT_DOCKER_IMAGE,
                prepared_video_path=prepared_video_path,
                work_directory=demo_directory,
                atlas_path=atlas_path,
                slam_setting_path=slam_setting_path,
                imu_json_path=imu_json_path,
                trajectory_csv_path=trajectory_csv_path,
                mask_path=mask_path,
                stdout_path=stdout_log_path,
                stderr_path=stderr_log_path,
                timeout_seconds=timeout_seconds,
            )
            status = (
                "success" if result.returncode in ALLOWED_RETURN_CODES else "failed"
            )
            error_message = None
            if status == "failed":
                error_message = (
                    f"SLAM extraction exited with code {result.returncode}. "
                    "Inspect the stderr log for details."
                )
        except subprocess.TimeoutExpired:
            logger.exception(
                "Trajectory extraction timed out for demo %s",
                demo_directory.name,
            )
            status = "timeout"
            error_message = "SLAM extraction timed out."
        except OPAIWorkflowError as exc:
            logger.exception(
                "Trajectory extraction failed for demo %s",
                demo_directory.name,
            )
            status = "failed"
            error_message = exc.message

        processed_videos.append(
            {
                "demo_id": demo_directory.name,
                "video_path": str(demo_video_path),
                "prepared_video_path": str(prepared_video_path),
                "trajectory_csv": str(trajectory_csv_path),
                "stdout_log": str(stdout_log_path),
                "stderr_log": str(stderr_log_path),
                "status": status,
                "error_message": error_message,
            }
        )

    logger.info(
        "Completed trajectory extraction batch for session %s. Processed %s demo videos",
        ctx.name,
        len(processed_videos),
    )
    return {
        "processed_videos": processed_videos,
        "total_processed": len(processed_videos),
    }


def _clear_mapping_outputs(work_directory: Path) -> None:
    logger.info("Clearing mapping outputs in %s", work_directory)
    for filename in (
        "imu_data.json",
        "slam_mask.png",
        PREPARED_MAPPING_VIDEO_FILENAME,
        "map_atlas.osa",
        "mapping_camera_trajectory.csv",
        "slam_stdout.txt",
        "slam_stderr.txt",
    ):
        (work_directory / filename).unlink(missing_ok=True)


###################
### NOTE: The following functions are adapted directly from the official universal-manipulation-interface repository with minimal modifications.
### Will be updated to a more generalized version in the near-future.
###################


def draw_predefined_mask(
    img, color=(0, 0, 0), mirror=True, gripper=True, finger=True, use_aa=False
):
    all_coords = list()
    if mirror:
        all_coords.extend(get_mirror_canonical_polygon())
    if gripper:
        all_coords.extend(get_gripper_canonical_polygon())
    if finger:
        all_coords.extend(get_finger_canonical_polygon())

    for coords in all_coords:
        image_size = (img.shape[1], img.shape[0])
        pts = canonical_to_pixel_coords(coords, image_size)
        pts = np.round(pts).astype(np.int32)
        flag = cv2.LINE_AA if use_aa else cv2.LINE_8
        cv2.fillPoly(img, [pts], color=color, lineType=flag)
    return img


def get_mirror_canonical_polygon():
    left_pts = [
        [540, 1700],
        [680, 1450],
        [590, 1070],
        [290, 1130],
        [290, 1770],
        [550, 1770],
    ]
    left_coords = pixel_coords_to_canonical(left_pts, DEFAULT_RESOLUTION)
    right_coords = left_coords.copy()
    right_coords[:, 0] *= -1
    coords = np.stack([left_coords, right_coords])
    return coords


def get_gripper_canonical_polygon():
    left_pts = [
        [1352, 1730],
        [1100, 1700],
        [650, 1500],
        [0, 1350],
        [0, 2028],
        [1352, 2028],
    ]
    left_coords = pixel_coords_to_canonical(left_pts, DEFAULT_RESOLUTION)
    right_coords = left_coords.copy()
    right_coords[:, 0] *= -1
    coords = np.stack([left_coords, right_coords])
    return coords


def get_finger_canonical_polygon(
    height=0.37, top_width=0.25, bottom_width=1.4, resolution=DEFAULT_RESOLUTION
):
    # image size
    img_w, img_h = resolution

    # calculate coordinates
    top_y = 1.0 - height
    bottom_y = 1.0
    width = img_w / img_h
    middle_x = width / 2.0
    top_left_x = middle_x - top_width / 2.0
    top_right_x = middle_x + top_width / 2.0
    bottom_left_x = middle_x - bottom_width / 2.0
    bottom_right_x = middle_x + bottom_width / 2.0

    top_y *= img_h
    bottom_y *= img_h
    top_left_x *= img_h
    top_right_x *= img_h
    bottom_left_x *= img_h
    bottom_right_x *= img_h

    # create polygon points for opencv API
    points = [
        [
            [bottom_left_x, bottom_y],
            [top_left_x, top_y],
            [top_right_x, top_y],
            [bottom_right_x, bottom_y],
        ]
    ]
    coords = pixel_coords_to_canonical(points, img_shape=resolution)
    return coords


def pixel_coords_to_canonical(pts, img_shape=DEFAULT_RESOLUTION):
    coords = (np.asarray(pts) - np.array(img_shape) * 0.5) / img_shape[1]
    return coords


def canonical_to_pixel_coords(coords, img_shape=DEFAULT_RESOLUTION):
    pts = np.asarray(coords) * img_shape[1] + np.array(img_shape) * 0.5
    return pts
