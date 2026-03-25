from __future__ import annotations

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
