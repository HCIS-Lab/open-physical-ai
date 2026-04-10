from __future__ import annotations

import json
import logging

import numpy as np
import pytest

import opai.application.imu as imu_module
from opai.core.exceptions import OPAIWorkflowError
from opai.domain.imu import IMUPayload, IMURecording, IMUSample, IMUStream


def test_imu_payload_to_dict_matches_wire_shape() -> None:
    payload = IMUPayload(
        recording=IMURecording(
            streams={
                "ACCL": IMUStream(
                    samples=(
                        IMUSample(value=[1.0, 2.0, 3.0], cts=125.0),
                        IMUSample(value=4.0, cts=250.0),
                    )
                )
            }
        ),
        frames_per_second=30.0,
    )

    assert payload.to_dict() == {
        "1": {
            "streams": {
                "ACCL": {
                    "samples": [
                        {"value": [1.0, 2.0, 3.0], "cts": 125.0},
                        {"value": 4.0, "cts": 250.0},
                    ]
                }
            }
        },
        "frames/second": 30.0,
    }


def test_extract_imu_from_video_writes_typed_payload_json(
    tmp_path, monkeypatch
) -> None:
    telemetry = {
        "ACCL": (
            [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])],
            np.array([0.125, 0.25]),
        ),
        "GYRO": (
            [np.array([7.0, 8.0, 9.0])],
            np.array([0.5]),
        ),
    }
    monkeypatch.setattr(
        imu_module,
        "GoProTelemetryExtractor",
        lambda video_path: _FakeExtractor(video_path, telemetry=telemetry),
    )

    output_path = imu_module.extract_imu_from_video(
        tmp_path / "input.mp4",
        tmp_path / "imu_data.json",
        stream_types=["ACCL", "GYRO"],
    )

    assert output_path == tmp_path / "imu_data.json"
    assert json.loads(output_path.read_text(encoding="utf-8")) == {
        "1": {
            "streams": {
                "ACCL": {
                    "samples": [
                        {"value": [1.0, 2.0, 3.0], "cts": 125.0},
                        {"value": [4.0, 5.0, 6.0], "cts": 250.0},
                    ]
                },
                "GYRO": {
                    "samples": [
                        {"value": [7.0, 8.0, 9.0], "cts": 500.0},
                    ]
                },
            }
        },
        "frames/second": 0.0,
    }


def test_extract_imu_from_video_omits_empty_streams(tmp_path, monkeypatch) -> None:
    telemetry = {
        "ACCL": (
            [np.array([1.0, 2.0, 3.0])],
            np.array([0.125]),
        ),
        "GYRO": (
            [],
            np.array([]),
        ),
    }
    monkeypatch.setattr(
        imu_module,
        "GoProTelemetryExtractor",
        lambda video_path: _FakeExtractor(video_path, telemetry=telemetry),
    )

    output_path = imu_module.extract_imu_from_video(
        tmp_path / "input.mp4",
        tmp_path / "imu_data.json",
        stream_types=["ACCL", "GYRO"],
    )

    assert json.loads(output_path.read_text(encoding="utf-8")) == {
        "1": {
            "streams": {
                "ACCL": {
                    "samples": [
                        {"value": [1.0, 2.0, 3.0], "cts": 125.0},
                    ]
                }
            }
        },
        "frames/second": 0.0,
    }


def test_extract_imu_from_video_logs_output_path(
    tmp_path,
    monkeypatch,
    caplog,
) -> None:
    telemetry = {
        "ACCL": (
            [np.array([1.0, 2.0, 3.0])],
            np.array([0.125]),
        )
    }
    monkeypatch.setattr(
        imu_module,
        "GoProTelemetryExtractor",
        lambda video_path: _FakeExtractor(video_path, telemetry=telemetry),
    )
    caplog.set_level(logging.INFO, logger="opai")

    output_path = imu_module.extract_imu_from_video(
        tmp_path / "input.mp4",
        tmp_path / "imu_data.json",
        stream_types=["ACCL"],
    )

    assert f"Wrote IMU payload to {output_path}" in caplog.text


def test_extract_imu_from_video_wraps_extractor_failures(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        imu_module,
        "GoProTelemetryExtractor",
        lambda video_path: _RaisingExtractor(video_path, "explode"),
    )
    video_path = tmp_path / "input.mp4"
    output_path = tmp_path / "imu_data.json"

    with pytest.raises(OPAIWorkflowError, match="Error processing") as exc_info:
        imu_module.extract_imu_from_video(video_path, output_path)

    assert str(video_path) in str(exc_info.value)
    assert "explode" in str(exc_info.value)


class _FakeExtractor:
    def __init__(
        self,
        video_path: str,
        *,
        telemetry: dict[str, tuple[list[np.ndarray], np.ndarray]],
    ) -> None:
        self.video_path = video_path
        self.telemetry = telemetry
        self.opened = False
        self.closed = False

    def open_source(self) -> None:
        self.opened = True

    def extract_data(
        self, stream_type: str
    ) -> tuple[list[np.ndarray], np.ndarray] | None:
        return self.telemetry.get(stream_type)

    def close_source(self) -> None:
        self.closed = True


class _RaisingExtractor:
    def __init__(self, video_path: str, message: str) -> None:
        self.video_path = video_path
        self.message = message

    def open_source(self) -> None:
        raise RuntimeError(self.message)

    def close_source(self) -> None:
        return None
