from __future__ import annotations

import logging

import numpy as np

from opai.domain.calibration import CalibrationIntrinsics, CalibrationResult
from opai.infrastructure.logger import get_logger
from opai.infrastructure.persistence import write_calibration_result


def test_get_logger_uses_opai_namespace() -> None:
    logger = get_logger("infrastructure.persistence")

    assert logger.name == "opai.infrastructure.persistence"


def test_write_calibration_result_emits_native_log_record(tmp_path, caplog) -> None:
    caplog.set_level(logging.INFO, logger="opai")

    result = CalibrationResult(
        mse_reproj_error=0.12,
        image_height=1080,
        image_width=1920,
        intrinsic_type="fisheye",
        intrinsics=CalibrationIntrinsics(
            aspect_ratio=1.0,
            focal_length=500.0,
            principal_pt_x=960.0,
            principal_pt_y=540.0,
            radial_distortion_1=0.1,
            radial_distortion_2=0.01,
            radial_distortion_3=0.001,
            radial_distortion_4=0.0001,
            skew=0.0,
        ),
        camera_matrix=np.eye(3),
        dist_coeffs=np.zeros((4, 1)),
    )

    output_path = write_calibration_result(tmp_path, result)

    assert output_path.exists()
    assert f"Wrote calibration result to {output_path}" in caplog.text
