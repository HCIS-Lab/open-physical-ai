from __future__ import annotations


class OPAIError(Exception):
    """Base exception for repo-defined errors with stable error codes."""

    default_error_code = "opai_error"

    def __init__(
        self,
        message: str,
        error_code: str | None = None,
        details: dict[str, str | int | float | bool | None] | None = None,
    ) -> None:
        self.message = message
        self.error_code = error_code or self.default_error_code
        self.details = details or {}
        super().__init__(message)

    def to_dict(
        self,
    ) -> dict[str, str | dict[str, str | int | float | bool | None]]:
        return {
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
        }


class OPAIValidationError(OPAIError, ValueError):
    """Raised when user-provided inputs fail validation."""

    default_error_code = "validation_error"


class OPAIContextError(OPAIError, RuntimeError):
    """Raised when notebook/session context is missing or invalid."""

    default_error_code = "context_error"


class OPAIDependencyError(OPAIError, RuntimeError):
    """Raised when required runtime dependencies are unavailable."""

    default_error_code = "dependency_error"


class OPAIWorkflowError(OPAIError, RuntimeError):
    """Raised when a workflow fails after input validation."""

    default_error_code = "workflow_error"


class OPAIGoProRegistrationError(OPAIError, RuntimeError):
    """Raised when a workflow fails after input validation."""

    default_error_code = "gopro_registration_error"


class OPAIGoProNotConnectedError(OPAIError, RuntimeError):
    """Raised when a workflow fails after input validation."""

    default_error_code = "gopro_not_connected_error"
