from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from opai.core.exceptions import OPAIValidationError


@dataclass
class Context:
    name: str
    session_directory: Path
    manifest_path: Path | None = None
    gopro_socket_address: str | None = None

    def set_gopro_socket_address(self, socket_address: str) -> None:
        if self.gopro_socket_address is not None:
            raise OPAIValidationError(
                "GoPro socket address is already set.",
                details={"socket_address": self.gopro_socket_address},
            )
        self.gopro_socket_address = socket_address
