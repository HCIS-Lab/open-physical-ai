from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np

from opai.core.exceptions import OPAIWorkflowError


def sample_video_frames(
    video_path: str | Path,
    frame_sample_step: int,
) -> tuple[np.ndarray, ...]:
    path = Path(video_path).expanduser()
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise OPAIWorkflowError(
            f"Unable to open video for calibration sampling: {path}",
            details={"path": str(path)},
        )

    sampled_frames: list[np.ndarray] = []
    frame_index = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % frame_sample_step == 0:
                sampled_frames.append(frame)
            frame_index += 1
    finally:
        capture.release()

    return tuple(sampled_frames)


def get_video_fps(video_path: str | Path) -> float:
    path = Path(video_path).expanduser()
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise OPAIWorkflowError(
            f"Unable to open video for FPS detection: {path}",
            details={"path": str(path)},
        )

    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()

    if fps <= 0:
        raise OPAIWorkflowError(
            f"Unable to detect a positive FPS value for video: {path}",
            details={"path": str(path), "fps": fps},
        )
    return fps


def get_video_resolution(video_path: str | Path) -> tuple[int, int]:
    path = Path(video_path).expanduser()
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise OPAIWorkflowError(
            f"Unable to open video for resolution detection: {path}",
            details={"path": str(path)},
        )

    try:
        return (
            int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    finally:
        capture.release()


def get_video_duration_seconds(video_path: str | Path) -> float:
    path = Path(video_path).expanduser()
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise OPAIWorkflowError(
            f"Unable to open video for duration detection: {path}",
            details={"path": str(path)},
        )

    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()

    if fps <= 0:
        raise OPAIWorkflowError(
            f"Unable to detect a positive FPS value for video: {path}",
            details={"path": str(path), "fps": fps},
        )
    if frame_count <= 0:
        return 0.0
    return frame_count / fps


def convert_video_to_fps(
    video_path: str | Path,
    output_path: str | Path,
    target_fps: int,
) -> Path:
    input_path = Path(video_path).expanduser()
    destination_path = Path(output_path).expanduser()
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        "ffmpeg",
        "-i",
        str(input_path),
        "-map_metadata",
        "0",
        "-movflags",
        "+faststart+use_metadata_tags",
        "-vf",
        f"fps={target_fps}",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "23",
        "-c:a",
        "copy",
        "-y",
        str(destination_path),
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise OPAIWorkflowError(
            f"Failed to convert video to {target_fps} fps: {input_path}",
            details={
                "input_path": str(input_path),
                "output_path": str(destination_path),
                "target_fps": target_fps,
                "stderr": result.stderr.strip(),
            },
        )
    return destination_path
