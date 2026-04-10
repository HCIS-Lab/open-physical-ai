from __future__ import annotations

import json
from pathlib import Path

from py_gpmf_parser.gopro_telemetry_extractor import GoProTelemetryExtractor

from opai.core.exceptions import OPAIWorkflowError
from opai.domain.imu import (
    DEFAULT_IMU_RECORDING_ID,
    IMUPayload,
    IMURecording,
    IMUSample,
    IMUStream,
)
from opai.infrastructure.logger import get_logger

SECS_TO_MS = 1e3
DEFAULT_STREAM_TYPES = [
    "ACCL",
    "GYRO",
    "GPS5",
    "GPSP",
    "GPSU",
    "GPSF",
    "GRAV",
    "MAGN",
    "CORI",
    "IORI",
    "TMPC",
]

logger = get_logger(__name__)


def extract_imu_from_video(
    video_path: Path,
    dest: Path,
    stream_types: list[str] = DEFAULT_STREAM_TYPES,
) -> Path:
    """Extract IMU data from a video using py_gpmf_parser."""

    logger.info("Extracting IMU streams from %s to %s", video_path, dest)
    extractor = GoProTelemetryExtractor(str(video_path))
    try:
        extractor.open_source()

        streams: dict[str, IMUStream] = {}
        for stream in stream_types:
            payload = extractor.extract_data(stream)
            if payload and len(payload[0]) > 0:
                logger.info(
                    "Extracted %s IMU samples for stream %s",
                    len(payload[0]),
                    stream,
                )
                streams[stream] = IMUStream(
                    samples=tuple(
                        IMUSample(
                            value=data.tolist(),
                            cts=_coerce_timestamp_millis(ts),
                        )
                        for data, ts in zip(*payload)
                    )
                )

        output = IMUPayload(
            recording_id=DEFAULT_IMU_RECORDING_ID,
            recording=IMURecording(streams=streams),
            frames_per_second=0.0,
        )
        dest.write_text(json.dumps(output.to_dict(), indent=2), encoding="utf-8")
        logger.info("Wrote IMU payload to %s", dest)

        return dest

    except Exception as exc:
        logger.exception("Failed to extract IMU data from %s", video_path)
        raise OPAIWorkflowError(f"Error processing {video_path}: {exc}") from exc

    finally:
        extractor.close_source()


def _coerce_timestamp_millis(timestamp: object) -> int | float:
    millis = timestamp * SECS_TO_MS
    if hasattr(millis, "tolist"):
        millis = millis.tolist()
    if isinstance(millis, list):
        if len(millis) != 1:
            raise OPAIWorkflowError(
                "IMU timestamp payload must contain exactly one scalar value.",
                details={"timestamp": str(millis)},
            )
        return millis[0]
    return millis
