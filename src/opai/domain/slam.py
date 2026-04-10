from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MappingRunResult:
    input_video_path: Path
    imu_json_path: Path
    mask_path: Path
    map_path: Path
    trajectory_csv_path: Path
    stdout_log_path: Path
    stderr_log_path: Path
