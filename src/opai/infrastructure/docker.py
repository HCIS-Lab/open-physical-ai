from __future__ import annotations

import os
import subprocess
from pathlib import Path

from opai.core.exceptions import OPAIWorkflowError
from opai.infrastructure.logger import get_logger

ALLOWED_RETURN_CODES = [
    0,
    139,
]  # NOTE: This is a noticible segfault, returncode of 139, in the chicheng/orb_slam3 image. We'll need to investigate this further.
DEFAULT_SLAM_CONTAINER_SETTINGS_PATH = Path("/slam_settings.yaml")
DEFAULT_SLAM_CONTAINER_VOCABULARY_PATH = Path("/ORB_SLAM3/Vocabulary/ORBvoc.txt")
DEFAULT_SLAM_MAPPING_COMMAND = "/ORB_SLAM3/Examples/Monocular-Inertial/gopro_slam"

logger = get_logger(__name__)


def pull_docker_image(docker_image: str) -> None:
    logger.info("Pulling Docker image: %s", docker_image)
    result = subprocess.run(
        ["docker", "pull", docker_image],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise OPAIWorkflowError(
            f"Failed to pull Docker image: {docker_image}",
            details={
                "docker_image": docker_image,
                "stderr": result.stderr.strip(),
            },
        )


def run_mapping_container(
    *,
    docker_image: str,
    prepared_video_path: Path,
    work_directory: Path,
    slam_setting_path: str,
    stdout_path: Path,
    stderr_path: Path,
    enable_gui: bool = True,
) -> None:
    workspace_target = Path("/workspace")
    video_mount_target = workspace_target
    command = ["docker", "run"]
    command.extend(
        [
            "--volume",
            f"{work_directory.resolve()}:{workspace_target}",
        ]
    )
    command.extend(
        [
            "--volume",
            f"{slam_setting_path}:{DEFAULT_SLAM_CONTAINER_SETTINGS_PATH}:ro",
        ]
    )

    if prepared_video_path.parent.resolve() != work_directory.resolve():
        video_mount_target = Path("/input")
        command.extend(
            [
                "--volume",
                f"{prepared_video_path.parent.resolve()}:{video_mount_target}:ro",
            ]
        )

    if enable_gui:
        display_env = os.environ.get("DISPLAY")
        if not display_env:
            raise OPAIWorkflowError(
                "DISPLAY environment variable is not set. GUI forwarding requires X11."
            )
        command.extend(["--volume", "/tmp/.X11-unix:/tmp/.X11-unix"])
        command.extend(["--env", f"DISPLAY={display_env}"])
        xdg_runtime_dir = os.environ.get("XDG_RUNTIME_DIR")
        if xdg_runtime_dir and os.path.exists(xdg_runtime_dir):
            command.extend(["--volume", f"{xdg_runtime_dir}:{xdg_runtime_dir}"])
            command.extend(["--env", f"XDG_RUNTIME_DIR={xdg_runtime_dir}"])

        xauthority_file = os.environ.get(
            "XAUTHORITY",
            os.path.expanduser("~/.Xauthority"),
        )
        if os.path.exists(xauthority_file):
            command.extend(["--volume", f"{xauthority_file}:{xauthority_file}"])
            command.extend(["--env", f"XAUTHORITY={xauthority_file}"])
        command.extend(["--env", "LIBGL_ALWAYS_SOFTWARE=1"])
        command.extend(["--ipc", "host"])

    command.extend(
        [
            docker_image,
            DEFAULT_SLAM_MAPPING_COMMAND,
            "--vocabulary",
            str(DEFAULT_SLAM_CONTAINER_VOCABULARY_PATH),
            "--setting",
            str(DEFAULT_SLAM_CONTAINER_SETTINGS_PATH),
            "--input_video",
            str(video_mount_target / prepared_video_path.name),
            "--input_imu_json",
            str(workspace_target / "imu_data.json"),
            "--output_trajectory_csv",
            str(workspace_target / "mapping_camera_trajectory.csv"),
            "--save_map",
            str(workspace_target / "map_atlas.osa"),
            "--mask_img",
            str(workspace_target / "slam_mask.png"),
        ]
    )

    if enable_gui:
        command.extend(["--enable_gui"])

    logger.info("Running Docker command: %s", " ".join(command))

    with (
        stdout_path.open("w", encoding="utf-8") as stdout_file,
        stderr_path.open("w", encoding="utf-8") as stderr_file,
    ):
        process = subprocess.Popen(
            command,
            cwd=str(work_directory.resolve()),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for line in iter(process.stdout.readline, ""):
            logger.info("SUBPROCESS STDOUT: %s", line.strip())
            stdout_file.write(line)
            stdout_file.flush()
        for line in iter(process.stderr.readline, ""):
            logger.warning("SUBPROCESS STDERR: %s", line.strip())
            stderr_file.write(line)
            stderr_file.flush()

        process.wait()

    if process.returncode not in ALLOWED_RETURN_CODES:
        raise OPAIWorkflowError(
            "SLAM mapping failed. Inspect the generated stdout/stderr logs for details.",
            details={
                "docker_image": docker_image,
                "return_code": process.returncode,
                "stdout_log": str(stdout_path),
                "stderr_log": str(stderr_path),
                "work_directory": str(work_directory),
            },
        )


def run_trajectory_extraction_container(
    *,
    docker_image: str,
    prepared_video_path: Path,
    work_directory: Path,
    atlas_path: Path,
    slam_setting_path: Path,
    imu_json_path: Path,
    trajectory_csv_path: Path,
    mask_path: Path,
    stdout_path: Path,
    stderr_path: Path,
    timeout_seconds: float | None = None,
) -> subprocess.CompletedProcess:
    workspace_target = Path("/workspace")
    atlas_mount_target = Path("/map")
    video_mount_target = workspace_target
    command = ["docker", "run", "--rm"]
    command.extend(
        [
            "--volume",
            f"{work_directory.resolve()}:{workspace_target}",
        ]
    )
    command.extend(
        [
            "--volume",
            f"{atlas_path.parent.resolve()}:{atlas_mount_target}:ro",
        ]
    )
    command.extend(
        [
            "--volume",
            f"{slam_setting_path.resolve()}:{DEFAULT_SLAM_CONTAINER_SETTINGS_PATH}:ro",
        ]
    )

    if prepared_video_path.parent.resolve() != work_directory.resolve():
        video_mount_target = Path("/input")
        command.extend(
            [
                "--volume",
                f"{prepared_video_path.parent.resolve()}:{video_mount_target}:ro",
            ]
        )

    command.extend(
        [
            docker_image,
            DEFAULT_SLAM_MAPPING_COMMAND,
            "--vocabulary",
            str(DEFAULT_SLAM_CONTAINER_VOCABULARY_PATH),
            "--setting",
            str(DEFAULT_SLAM_CONTAINER_SETTINGS_PATH),
            "--input_video",
            str(video_mount_target / prepared_video_path.name),
            "--input_imu_json",
            str(workspace_target / imu_json_path.name),
            "--output_trajectory_csv",
            str(workspace_target / trajectory_csv_path.name),
            "--load_map",
            str(atlas_mount_target / atlas_path.name),
            "--mask_img",
            str(workspace_target / mask_path.name),
        ]
    )

    logger.info("Running Docker command: %s", " ".join(command))

    with (
        stdout_path.open("w", encoding="utf-8") as stdout_file,
        stderr_path.open("w", encoding="utf-8") as stderr_file,
    ):
        return subprocess.run(
            command,
            cwd=str(work_directory.resolve()),
            stdout=stdout_file,
            stderr=stderr_file,
            timeout=timeout_seconds,
            check=False,
        )
