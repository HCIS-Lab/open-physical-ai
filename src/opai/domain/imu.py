from __future__ import annotations

from dataclasses import dataclass

DEFAULT_IMU_RECORDING_ID = "1"
IMUNumeric = int | float
IMUValue = IMUNumeric | list[IMUNumeric]


@dataclass(frozen=True)
class IMUSample:
    value: IMUValue
    cts: IMUNumeric

    def to_dict(self) -> dict[str, IMUValue | IMUNumeric]:
        return {
            "value": self.value,
            "cts": self.cts,
        }


@dataclass(frozen=True)
class IMUStream:
    samples: tuple[IMUSample, ...]

    def to_dict(self) -> dict[str, list[dict[str, IMUValue | IMUNumeric]]]:
        return {
            "samples": [sample.to_dict() for sample in self.samples],
        }


@dataclass(frozen=True)
class IMURecording:
    streams: dict[str, IMUStream]

    def to_dict(self) -> dict[str, dict[str, dict[str, list[dict[str, object]]]]]:
        return {
            "streams": {
                stream_name: stream.to_dict()
                for stream_name, stream in self.streams.items()
            }
        }


@dataclass(frozen=True)
class IMUPayload:
    recording: IMURecording
    frames_per_second: float = 0.0
    recording_id: str = DEFAULT_IMU_RECORDING_ID

    def to_dict(self) -> dict[str, object]:
        return {
            self.recording_id: self.recording.to_dict(),
            "frames/second": self.frames_per_second,
        }
