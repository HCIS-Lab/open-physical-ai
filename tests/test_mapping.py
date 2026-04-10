from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import opai
from opai.application import slam as slam_module
from opai.core.exceptions import OPAIContextError, OPAIWorkflowError
from opai.domain.context import Context
from opai.infrastructure import context_store


def test_run_mapping_requires_context(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(context_store, "_ACTIVE_CONTEXT", None)

    with pytest.raises(OPAIContextError, match="Call opai.init"):
        opai.run_mapping(slam_settings_file=tmp_path / "slam_settings.yaml")


def test_add_mapping_video_replaces_active_mapping(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    opai.init("session-001")
    mapping_a = tmp_path / "mapping-a.mp4"
    mapping_b = tmp_path / "mapping-b.mp4"
    mapping_a.write_bytes(b"first")
    mapping_b.write_bytes(b"second")

    first = opai.add_mapping_video(mapping_a)
    second = opai.add_mapping_video(mapping_b)

    assert first.original_filename == "mapping-a.mp4"
    assert second.original_filename == "mapping-b.mp4"
    mapping_dir = (
        tmp_path / "sessions" / "session-001" / "captures" / "mapping" / "current"
    )
    assert sorted(path.name for path in mapping_dir.iterdir()) == ["mapping-b.mp4"]


def test_run_mapping_requires_registered_mapping_video(tmp_path) -> None:
    session_directory = tmp_path / "sessions" / "session-001"
    session_directory.mkdir(parents=True)
    ctx = Context(
        name="session-001",
        session_directory=session_directory,
        manifest_path=session_directory / "session.json",
    )
    settings_path = tmp_path / "slam_settings.yaml"
    settings_path.write_text("settings", encoding="utf-8")

    with pytest.raises(OPAIWorkflowError, match="No mapping video found"):
        slam_module.run_mapping(ctx, slam_settings_file=settings_path)


def test_clear_mapping_outputs_logs_directory(tmp_path, caplog) -> None:
    work_directory = tmp_path / "slam" / "mapping"
    work_directory.mkdir(parents=True)
    stale_output = work_directory / "imu_data.json"
    stale_output.write_text("stale", encoding="utf-8")
    caplog.set_level(logging.INFO, logger="opai")

    slam_module._clear_mapping_outputs(work_directory)

    assert not stale_output.exists()
    assert f"Clearing mapping outputs in {work_directory}" in caplog.text


def test_run_mapping_always_reruns_and_overwrites_outputs(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    ctx = opai.init("session-001")
    mapping_video = tmp_path / "mapping.mp4"
    mapping_video.write_bytes(b"mapping")
    opai.add_mapping_video(mapping_video)
    settings_path = tmp_path / "slam_settings.yaml"
    settings_path.write_text("settings", encoding="utf-8")

    events: list[str] = []
    container_runs: list[int] = []
    prepared_video_existed_before_convert: list[bool] = []
    config_dir = tmp_path / "configs" / "slam"
    config_dir.mkdir(parents=True)
    config_path = config_dir / settings_path.name
    config_path.write_text("settings", encoding="utf-8")

    def fake_get_slam_config_dir() -> Path:
        return config_dir

    def fake_get_video_fps(video_path: Path) -> float:
        assert video_path.name == "mapping.mp4"
        return 120.0

    def fake_convert_video_to_fps(
        video_path: Path,
        output_path: Path,
        target_fps: int,
    ) -> Path:
        events.append(f"convert:{target_fps}")
        prepared_video_existed_before_convert.append(output_path.exists())
        assert video_path.name == "mapping.mp4"
        assert target_fps == slam_module.TARGET_SLAM_FPS
        output_path.write_text("prepared", encoding="utf-8")
        return output_path

    def fake_get_video_resolution(video_path: Path) -> tuple[int, int]:
        assert video_path.name == slam_module.PREPARED_MAPPING_VIDEO_FILENAME
        return slam_module.DEFAULT_RESOLUTION

    def fake_extract(video_path: Path, dest: Path) -> Path:
        events.append("extract")
        assert video_path.name == "mapping.mp4"
        dest.write_text("imu", encoding="utf-8")
        return dest

    def fake_draw_mask(mask, **_kwargs):
        events.append("mask")
        return mask

    def fake_imwrite(path: str, _image) -> bool:
        Path(path).write_text("mask", encoding="utf-8")
        return True

    def fake_pull(docker_image: str) -> None:
        events.append(f"pull:{docker_image}")

    def fake_run(
        *,
        docker_image: str,
        prepared_video_path: Path,
        work_directory: Path,
        settings_file: Path,
        stdout_path: Path,
        stderr_path: Path,
        enable_gui: bool,
    ) -> None:
        events.append(f"run:{docker_image}")
        container_runs.append(len(container_runs) + 1)
        assert prepared_video_path.name == slam_module.PREPARED_MAPPING_VIDEO_FILENAME
        assert settings_file == config_path
        assert enable_gui is False
        (work_directory / "map_atlas.osa").write_text(
            f"map-{container_runs[-1]}",
            encoding="utf-8",
        )
        (work_directory / "mapping_camera_trajectory.csv").write_text(
            f"trajectory-{container_runs[-1]}",
            encoding="utf-8",
        )
        stdout_path.write_text(f"stdout-{container_runs[-1]}", encoding="utf-8")
        stderr_path.write_text(f"stderr-{container_runs[-1]}", encoding="utf-8")

    monkeypatch.setattr(slam_module, "get_slam_config_dir", fake_get_slam_config_dir)
    monkeypatch.setattr(slam_module, "get_video_fps", fake_get_video_fps)
    monkeypatch.setattr(slam_module, "convert_video_to_fps", fake_convert_video_to_fps)
    monkeypatch.setattr(slam_module, "get_video_resolution", fake_get_video_resolution)
    monkeypatch.setattr(slam_module, "extract_imu_from_video", fake_extract)
    monkeypatch.setattr(slam_module, "draw_predefined_mask", fake_draw_mask)
    monkeypatch.setattr(slam_module.cv2, "imwrite", fake_imwrite)
    monkeypatch.setattr(slam_module, "pull_docker_image", fake_pull)
    monkeypatch.setattr(slam_module, "run_mapping_container", fake_run)

    first = slam_module.run_mapping(
        ctx,
        slam_settings_file=settings_path.name,
    )
    second = slam_module.run_mapping(
        ctx,
        slam_settings_file=settings_path.name,
    )

    assert len(container_runs) == 2
    assert prepared_video_existed_before_convert == [False, False]
    assert events.count(f"convert:{slam_module.TARGET_SLAM_FPS}") == 2
    assert events.count("extract") == 2
    assert events.count("mask") == 2
    assert sum(event.startswith("pull:") for event in events) == 2
    assert sum(event.startswith("run:") for event in events) == 2
    assert first.input_video_path.name == "mapping.mp4"
    assert first.map_path.read_text(encoding="utf-8") == "map-2"
    assert second.map_path.read_text(encoding="utf-8") == "map-2"
    assert second.trajectory_csv_path.read_text(encoding="utf-8") == "trajectory-2"


def test_run_mapping_uses_original_input_when_video_is_already_60fps(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    ctx = opai.init("session-001")
    mapping_video = tmp_path / "mapping.mp4"
    mapping_video.write_bytes(b"mapping")
    opai.add_mapping_video(mapping_video)
    settings_path = tmp_path / "slam_settings.yaml"
    settings_path.write_text("settings", encoding="utf-8")

    config_dir = tmp_path / "configs" / "slam"
    config_dir.mkdir(parents=True)
    config_path = config_dir / settings_path.name
    config_path.write_text("settings", encoding="utf-8")
    events: list[str] = []

    def fake_get_slam_config_dir() -> Path:
        return config_dir

    def fake_get_video_fps(video_path: Path) -> float:
        assert video_path.name == "mapping.mp4"
        return 60.0

    def fake_convert_video_to_fps(*_args, **_kwargs) -> Path:
        raise AssertionError("60 fps input should not be converted")

    def fake_get_video_resolution(video_path: Path) -> tuple[int, int]:
        assert video_path.name == "mapping.mp4"
        return slam_module.DEFAULT_RESOLUTION

    def fake_extract(video_path: Path, dest: Path) -> Path:
        events.append("extract")
        assert video_path.name == "mapping.mp4"
        dest.write_text("imu", encoding="utf-8")
        return dest

    def fake_draw_mask(mask, **_kwargs):
        events.append("mask")
        return mask

    def fake_imwrite(path: str, _image) -> bool:
        Path(path).write_text("mask", encoding="utf-8")
        return True

    def fake_pull(_docker_image: str) -> None:
        events.append("pull")

    def fake_run(
        *,
        docker_image: str,
        prepared_video_path: Path,
        work_directory: Path,
        settings_file: Path,
        stdout_path: Path,
        stderr_path: Path,
        enable_gui: bool,
    ) -> None:
        events.append("run")
        assert docker_image == slam_module.DEFAULT_DOCKER_IMAGE
        assert prepared_video_path.name == "mapping.mp4"
        assert settings_file == config_path
        assert work_directory == ctx.session_directory / "slam" / "mapping"
        assert enable_gui is False
        stdout_path.write_text("stdout", encoding="utf-8")
        stderr_path.write_text("stderr", encoding="utf-8")

    monkeypatch.setattr(slam_module, "get_slam_config_dir", fake_get_slam_config_dir)
    monkeypatch.setattr(slam_module, "get_video_fps", fake_get_video_fps)
    monkeypatch.setattr(slam_module, "convert_video_to_fps", fake_convert_video_to_fps)
    monkeypatch.setattr(slam_module, "get_video_resolution", fake_get_video_resolution)
    monkeypatch.setattr(slam_module, "extract_imu_from_video", fake_extract)
    monkeypatch.setattr(slam_module, "draw_predefined_mask", fake_draw_mask)
    monkeypatch.setattr(slam_module.cv2, "imwrite", fake_imwrite)
    monkeypatch.setattr(slam_module, "pull_docker_image", fake_pull)
    monkeypatch.setattr(slam_module, "run_mapping_container", fake_run)

    result = slam_module.run_mapping(
        ctx,
        slam_settings_file=settings_path.name,
    )

    assert result.input_video_path.name == "mapping.mp4"
    assert events == ["extract", "mask", "pull", "run"]


def test_run_mapping_raises_if_120fps_conversion_fails(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    ctx = opai.init("session-001")
    mapping_video = tmp_path / "mapping.mp4"
    mapping_video.write_bytes(b"mapping")
    opai.add_mapping_video(mapping_video)
    settings_path = tmp_path / "slam_settings.yaml"
    settings_path.write_text("settings", encoding="utf-8")

    config_dir = tmp_path / "configs" / "slam"
    config_dir.mkdir(parents=True)
    config_path = config_dir / settings_path.name
    config_path.write_text("settings", encoding="utf-8")
    container_started = False
    image_pulled = False

    def fake_get_slam_config_dir() -> Path:
        return config_dir

    def fake_get_video_fps(video_path: Path) -> float:
        assert video_path.name == "mapping.mp4"
        return 120.0

    def fake_convert_video_to_fps(*_args, **_kwargs) -> Path:
        raise OPAIWorkflowError("conversion failed")

    def fake_pull(_docker_image: str) -> None:
        nonlocal image_pulled
        image_pulled = True

    def fake_run(**_kwargs) -> None:
        nonlocal container_started
        container_started = True

    monkeypatch.setattr(slam_module, "get_slam_config_dir", fake_get_slam_config_dir)
    monkeypatch.setattr(slam_module, "get_video_fps", fake_get_video_fps)
    monkeypatch.setattr(slam_module, "convert_video_to_fps", fake_convert_video_to_fps)
    monkeypatch.setattr(slam_module, "pull_docker_image", fake_pull)
    monkeypatch.setattr(slam_module, "run_mapping_container", fake_run)

    with pytest.raises(OPAIWorkflowError, match="conversion failed"):
        slam_module.run_mapping(
            ctx,
            slam_settings_file=config_path.name,
        )

    assert image_pulled is False
    assert container_started is False


def test_draw_predefined_mask_uses_width_height_image_order() -> None:
    image_width, image_height = slam_module.DEFAULT_RESOLUTION
    mask = np.zeros((image_height, image_width), dtype=np.uint8)

    slam_module.draw_predefined_mask(
        mask, color=255, mirror=False, gripper=False, finger=True
    )

    probe_row = int(image_height * 0.7)
    probe_col = image_width // 2
    assert mask[probe_row, probe_col] == 255


def test_gripper_polygon_round_trips_within_default_resolution() -> None:
    image_width, image_height = slam_module.DEFAULT_RESOLUTION
    left_gripper = slam_module.get_gripper_canonical_polygon()[0]
    points = slam_module.canonical_to_pixel_coords(
        left_gripper, slam_module.DEFAULT_RESOLUTION
    )

    assert np.all(points[:, 0] >= 0)
    assert np.all(points[:, 0] <= image_width)
    assert np.all(points[:, 1] >= 0)
    assert np.all(points[:, 1] <= image_height)


def test_run_extract_trajectories_batch_requires_map_atlas(tmp_path) -> None:
    session_directory = tmp_path / "sessions" / "session-001"
    session_directory.mkdir(parents=True)
    ctx = Context(
        name="session-001",
        session_directory=session_directory,
        manifest_path=session_directory / "session.json",
    )

    with pytest.raises(OPAIWorkflowError, match="map atlas file"):
        slam_module.run_extract_trajectories_batch(ctx)


def test_run_extract_trajectories_batch_processes_registered_demos(
    tmp_path,
    monkeypatch,
) -> None:
    session_directory = tmp_path / "sessions" / "session-001"
    atlas_path = session_directory / "slam" / "mapping" / "map_atlas.osa"
    atlas_path.parent.mkdir(parents=True, exist_ok=True)
    atlas_path.write_text("atlas", encoding="utf-8")
    settings_path = session_directory / "slam_settings.yaml"
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text("settings", encoding="utf-8")

    demo_a_directory = session_directory / "captures" / "demos" / "demo-0001"
    demo_b_directory = session_directory / "captures" / "demos" / "demo-0002"
    demo_a_directory.mkdir(parents=True, exist_ok=True)
    demo_b_directory.mkdir(parents=True, exist_ok=True)
    demo_a_video = demo_a_directory / "demo-a.mp4"
    demo_b_video = demo_b_directory / "demo-b.mp4"
    demo_a_video.write_bytes(b"demo-a")
    demo_b_video.write_bytes(b"demo-b")
    (demo_b_directory / "camera_trajectory.csv").write_text(
        "existing",
        encoding="utf-8",
    )
    ctx = Context(
        name="session-001",
        session_directory=session_directory,
        manifest_path=session_directory / "session.json",
    )

    events: list[str] = []

    def fake_get_demo_videos(_ctx: Context) -> list[Path]:
        return [demo_a_video, demo_b_video]

    def fake_pull(docker_image: str) -> None:
        events.append(f"pull:{docker_image}")

    def fake_get_video_fps(video_path: Path) -> float:
        assert video_path == demo_a_video
        return 120.0

    def fake_convert_video_to_fps(
        video_path: Path,
        output_path: Path,
        target_fps: int,
    ) -> Path:
        events.append(f"convert:{target_fps}")
        assert video_path == demo_a_video
        output_path.write_text("prepared", encoding="utf-8")
        return output_path

    def fake_get_video_resolution(video_path: Path) -> tuple[int, int]:
        assert video_path.name == slam_module.PREPARED_TRAJECTORY_VIDEO_FILENAME
        return slam_module.DEFAULT_RESOLUTION

    def fake_extract(video_path: Path, destination: Path) -> Path:
        events.append("extract")
        assert video_path == demo_a_video
        destination.write_text("imu", encoding="utf-8")
        return destination

    def fake_draw_mask(mask, **_kwargs):
        events.append("mask")
        return mask

    def fake_imwrite(path: str, _image) -> bool:
        Path(path).write_text("mask", encoding="utf-8")
        return True

    def fake_get_video_duration_seconds(video_path: Path) -> float:
        assert video_path.name == slam_module.PREPARED_TRAJECTORY_VIDEO_FILENAME
        return 12.5

    def fake_run_trajectory_extraction_container(
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
        timeout_seconds: float | None,
    ) -> SimpleNamespace:
        events.append("run")
        assert docker_image == slam_module.DEFAULT_DOCKER_IMAGE
        assert prepared_video_path == (
            demo_a_directory / slam_module.PREPARED_TRAJECTORY_VIDEO_FILENAME
        )
        assert work_directory == demo_a_directory
        assert atlas_path == session_directory / "slam" / "mapping" / "map_atlas.osa"
        assert slam_setting_path == settings_path
        assert imu_json_path == demo_a_directory / "imu_data.json"
        assert trajectory_csv_path == demo_a_directory / "camera_trajectory.csv"
        assert mask_path == demo_a_directory / "slam_mask.png"
        assert timeout_seconds == (
            12.5 * slam_module.DEFAULT_TRAJECTORY_TIMEOUT_MULTIPLE
        )
        trajectory_csv_path.write_text("trajectory", encoding="utf-8")
        stdout_path.write_text("stdout", encoding="utf-8")
        stderr_path.write_text("stderr", encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(slam_module, "get_demo_videos", fake_get_demo_videos)
    monkeypatch.setattr(slam_module, "pull_docker_image", fake_pull)
    monkeypatch.setattr(slam_module, "get_video_fps", fake_get_video_fps)
    monkeypatch.setattr(slam_module, "convert_video_to_fps", fake_convert_video_to_fps)
    monkeypatch.setattr(slam_module, "get_video_resolution", fake_get_video_resolution)
    monkeypatch.setattr(slam_module, "extract_imu_from_video", fake_extract)
    monkeypatch.setattr(slam_module, "draw_predefined_mask", fake_draw_mask)
    monkeypatch.setattr(slam_module.cv2, "imwrite", fake_imwrite)
    monkeypatch.setattr(
        slam_module,
        "get_video_duration_seconds",
        fake_get_video_duration_seconds,
    )
    monkeypatch.setattr(
        slam_module,
        "run_trajectory_extraction_container",
        fake_run_trajectory_extraction_container,
    )

    result = slam_module.run_extract_trajectories_batch(ctx)

    assert events == [
        f"pull:{slam_module.DEFAULT_DOCKER_IMAGE}",
        f"convert:{slam_module.TARGET_SLAM_FPS}",
        "extract",
        "mask",
        "run",
    ]
    assert result["total_processed"] == 1
    assert result["processed_videos"] == [
        {
            "demo_id": "demo-0001",
            "video_path": str(demo_a_video),
            "prepared_video_path": str(
                demo_a_directory / slam_module.PREPARED_TRAJECTORY_VIDEO_FILENAME
            ),
            "trajectory_csv": str(demo_a_directory / "camera_trajectory.csv"),
            "stdout_log": str(demo_a_directory / "slam_stdout.txt"),
            "stderr_log": str(demo_a_directory / "slam_stderr.txt"),
            "status": "success",
            "error_message": None,
        }
    ]


def test_run_extract_trajectories_batch_records_timeouts_and_failures(
    tmp_path,
    monkeypatch,
) -> None:
    session_directory = tmp_path / "sessions" / "session-001"
    atlas_path = session_directory / "slam" / "mapping" / "map_atlas.osa"
    atlas_path.parent.mkdir(parents=True, exist_ok=True)
    atlas_path.write_text("atlas", encoding="utf-8")
    settings_path = session_directory / "slam_settings.yaml"
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text("settings", encoding="utf-8")

    demo_a_directory = session_directory / "captures" / "demos" / "demo-0001"
    demo_b_directory = session_directory / "captures" / "demos" / "demo-0002"
    demo_a_directory.mkdir(parents=True, exist_ok=True)
    demo_b_directory.mkdir(parents=True, exist_ok=True)
    demo_a_video = demo_a_directory / "demo-a.mp4"
    demo_b_video = demo_b_directory / "demo-b.mp4"
    demo_a_video.write_bytes(b"demo-a")
    demo_b_video.write_bytes(b"demo-b")
    ctx = Context(
        name="session-001",
        session_directory=session_directory,
        manifest_path=session_directory / "session.json",
    )

    def fake_get_demo_videos(_ctx: Context) -> list[Path]:
        return [demo_a_video, demo_b_video]

    def fake_pull(_docker_image: str) -> None:
        return None

    def fake_get_video_fps(_video_path: Path) -> float:
        return 60.0

    def fake_get_video_resolution(_video_path: Path) -> tuple[int, int]:
        return slam_module.DEFAULT_RESOLUTION

    def fake_extract(_video_path: Path, destination: Path) -> Path:
        destination.write_text("imu", encoding="utf-8")
        return destination

    def fake_draw_mask(mask, **_kwargs):
        return mask

    def fake_imwrite(path: str, _image) -> bool:
        Path(path).write_text("mask", encoding="utf-8")
        return True

    def fake_get_video_duration_seconds(_video_path: Path) -> float:
        return 5.0

    def fake_run_trajectory_extraction_container(
        *,
        work_directory: Path,
        stdout_path: Path,
        stderr_path: Path,
        **_kwargs,
    ) -> SimpleNamespace:
        stdout_path.write_text("stdout", encoding="utf-8")
        stderr_path.write_text("stderr", encoding="utf-8")
        if work_directory == demo_a_directory:
            raise subprocess.TimeoutExpired(cmd=["docker"], timeout=50.0)
        return SimpleNamespace(returncode=2)

    monkeypatch.setattr(slam_module, "get_demo_videos", fake_get_demo_videos)
    monkeypatch.setattr(slam_module, "pull_docker_image", fake_pull)
    monkeypatch.setattr(slam_module, "get_video_fps", fake_get_video_fps)
    monkeypatch.setattr(slam_module, "get_video_resolution", fake_get_video_resolution)
    monkeypatch.setattr(slam_module, "extract_imu_from_video", fake_extract)
    monkeypatch.setattr(slam_module, "draw_predefined_mask", fake_draw_mask)
    monkeypatch.setattr(slam_module.cv2, "imwrite", fake_imwrite)
    monkeypatch.setattr(
        slam_module,
        "get_video_duration_seconds",
        fake_get_video_duration_seconds,
    )
    monkeypatch.setattr(
        slam_module,
        "run_trajectory_extraction_container",
        fake_run_trajectory_extraction_container,
    )

    result = slam_module.run_extract_trajectories_batch(ctx)

    assert result["total_processed"] == 2
    assert [entry["status"] for entry in result["processed_videos"]] == [
        "timeout",
        "failed",
    ]
    assert (
        result["processed_videos"][0]["error_message"] == "SLAM extraction timed out."
    )
    assert result["processed_videos"][1]["error_message"] == (
        "SLAM extraction exited with code 2. Inspect the stderr log for details."
    )
