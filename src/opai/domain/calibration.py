from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CalibrationIntrinsics:
    aspect_ratio: float
    focal_length: float
    principal_pt_x: float
    principal_pt_y: float
    radial_distortion_1: float
    radial_distortion_2: float
    radial_distortion_3: float
    radial_distortion_4: float
    skew: float


@dataclass
class CalibrationResult:
    mse_reproj_error: float
    image_height: int
    image_width: int
    intrinsic_type: str
    intrinsics: CalibrationIntrinsics
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
