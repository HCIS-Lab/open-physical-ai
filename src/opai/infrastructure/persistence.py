import json
from pathlib import Path

from opai.domain.calibration import CalibrationResult


def write_calibration_result(
    session_directory: Path,
    result: CalibrationResult,
    filename: str = "calibration.json",
) -> Path:
    payload = {
        "mse_reproj_error": result.mse_reproj_error,
        "image_height": result.image_height,
        "image_width": result.image_width,
        "intrinsic_type": result.intrinsic_type,
        "intrinsics": {
            "aspect_ratio": result.intrinsics.aspect_ratio,
            "focal_length": result.intrinsics.focal_length,
            "principal_pt_x": result.intrinsics.principal_pt_x,
            "principal_pt_y": result.intrinsics.principal_pt_y,
            "radial_distortion_1": result.intrinsics.radial_distortion_1,
            "radial_distortion_2": result.intrinsics.radial_distortion_2,
            "radial_distortion_3": result.intrinsics.radial_distortion_3,
            "radial_distortion_4": result.intrinsics.radial_distortion_4,
            "skew": result.intrinsics.skew,
        },
    }

    output_path = session_directory / filename
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path
