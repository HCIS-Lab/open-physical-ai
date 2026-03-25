from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from opai.core.exceptions import OPAIValidationError

DEFAULT_CHARUCO_DICTIONARY = "DICT_5X5_100"
DEFAULT_CHARUCO_SQUARES_X = 11
DEFAULT_CHARUCO_SQUARES_Y = 8
DEFAULT_CHARUCO_SQUARE_LENGTH = 0.03
DEFAULT_CHARUCO_MARKER_LENGTH = 0.022
DEFAULT_CHARUCO_IMAGE_WIDTH_PX = 2000
DEFAULT_CHARUCO_IMAGE_HEIGHT_PX = 1400
DEFAULT_CHARUCO_MARGIN_SIZE_PX = 20


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


@dataclass(frozen=True)
class CalibrationVerificationFrame:
    sampled_frame_index: int
    detected_corner_count: int
    mse_reproj_error: float


@dataclass(frozen=True)
class CalibrationVerificationResult:
    requested_check_image_count: int
    sampled_image_count: int
    verified_image_count: int
    skipped_image_count: int
    total_detected_corner_count: int
    mse_reproj_error: float
    frame_results: tuple[CalibrationVerificationFrame, ...]


@dataclass(frozen=True)
class CharucoBoardConfig:
    dictionary: str
    squares_x: int
    squares_y: int
    square_length: float
    marker_length: float
    image_width_px: int
    image_height_px: int
    margin_size_px: int


@dataclass(frozen=True)
class CharucoBoardArtifacts:
    image_path: Path
    config_path: Path
    config: CharucoBoardConfig


def validate_charuco_board_config(config: CharucoBoardConfig) -> None:
    if not isinstance(config.dictionary, str) or not config.dictionary.strip():
        raise OPAIValidationError(
            "dictionary must be a non-empty ArUco dictionary name.",
            details={"dictionary": config.dictionary},
        )
    if config.squares_x <= 1 or config.squares_y <= 1:
        raise OPAIValidationError(
            "squares_x and squares_y must both be greater than 1.",
            details={
                "squares_x": config.squares_x,
                "squares_y": config.squares_y,
            },
        )
    if config.square_length <= 0 or config.marker_length <= 0:
        raise OPAIValidationError(
            "square_length and marker_length must both be positive.",
            details={
                "square_length": config.square_length,
                "marker_length": config.marker_length,
            },
        )
    if config.marker_length >= config.square_length:
        raise OPAIValidationError(
            "marker_length must be smaller than square_length.",
            details={
                "square_length": config.square_length,
                "marker_length": config.marker_length,
            },
        )
    if config.image_width_px <= 0 or config.image_height_px <= 0:
        raise OPAIValidationError(
            "image_width_px and image_height_px must both be positive.",
            details={
                "image_width_px": config.image_width_px,
                "image_height_px": config.image_height_px,
            },
        )
    if config.margin_size_px < 0:
        raise OPAIValidationError(
            "margin_size_px must be greater than or equal to 0.",
            details={"margin_size_px": config.margin_size_px},
        )
