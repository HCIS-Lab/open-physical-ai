from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from opai.core.exceptions import OPAIWorkflowError
from opai.infrastructure import docker as docker_module


def test_run_mapping_container_places_docker_flags_before_image(
    tmp_path,
    monkeypatch,
) -> None:
    work_directory = tmp_path / "work"
    input_directory = tmp_path / "input"
    settings_file = tmp_path / "settings.yaml"
    stdout_path = tmp_path / "stdout.txt"
    stderr_path = tmp_path / "stderr.txt"
    work_directory.mkdir()
    input_directory.mkdir()
    settings_file.write_text("settings", encoding="utf-8")
    prepared_video_path = input_directory / "raw_video.mp4"
    prepared_video_path.write_bytes(b"video")

    xdg_runtime_dir = tmp_path / "xdg"
    xauthority_file = tmp_path / ".Xauthority"
    xdg_runtime_dir.mkdir()
    xauthority_file.write_text("cookie", encoding="utf-8")
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(xdg_runtime_dir))
    monkeypatch.setenv("XAUTHORITY", str(xauthority_file))
    monkeypatch.setattr(docker_module.os, "getuid", lambda: 1000)
    monkeypatch.setattr(docker_module.os, "getgid", lambda: 1000)

    captured_command: list[str] = []

    def fake_run(command: list[str], **_kwargs) -> SimpleNamespace:
        captured_command[:] = command
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(docker_module.subprocess, "run", fake_run)

    docker_module.run_mapping_container(
        docker_image="chicheng/orb_slam3:latest",
        prepared_video_path=prepared_video_path,
        work_directory=work_directory,
        settings_file=settings_file,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )

    image_index = captured_command.index("chicheng/orb_slam3:latest")
    assert captured_command[:3] == ["docker", "run", "--rm"]
    assert (
        captured_command[image_index + 1] == docker_module.DEFAULT_SLAM_MAPPING_COMMAND
    )
    user_index = captured_command.index("--user")
    assert captured_command[user_index : user_index + 2] == ["--user", "1000:1000"]
    assert user_index < image_index
    ipc_index = captured_command.index("--ipc")
    assert captured_command[ipc_index : ipc_index + 2] == ["--ipc", "host"]
    assert ipc_index < image_index
    assert "/tmp/.X11-unix:/tmp/.X11-unix" in captured_command
    assert f"{input_directory.resolve()}:/input:ro" in captured_command
    assert "DISPLAY=:0" in captured_command
    assert f"XDG_RUNTIME_DIR={xdg_runtime_dir}" in captured_command
    assert f"XAUTHORITY={xauthority_file}" in captured_command


def test_run_mapping_container_requires_display_for_gui(
    tmp_path,
    monkeypatch,
) -> None:
    work_directory = tmp_path / "work"
    settings_file = tmp_path / "settings.yaml"
    stdout_path = tmp_path / "stdout.txt"
    stderr_path = tmp_path / "stderr.txt"
    work_directory.mkdir()
    settings_file.write_text("settings", encoding="utf-8")
    prepared_video_path = work_directory / "raw_video.mp4"
    prepared_video_path.write_bytes(b"video")
    monkeypatch.delenv("DISPLAY", raising=False)

    with pytest.raises(OPAIWorkflowError, match="DISPLAY"):
        docker_module.run_mapping_container(
            docker_image="chicheng/orb_slam3:latest",
            prepared_video_path=prepared_video_path,
            work_directory=work_directory,
            settings_file=settings_file,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )
