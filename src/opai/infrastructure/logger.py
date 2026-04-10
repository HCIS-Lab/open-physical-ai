from __future__ import annotations

import logging

PACKAGE_LOGGER_NAME = "opai"
DEFAULT_LOG_FORMAT = "%(levelname)s %(name)s: %(message)s"

_package_logger = logging.getLogger(PACKAGE_LOGGER_NAME)
if not any(
    isinstance(handler, logging.NullHandler) for handler in _package_logger.handlers
):
    _package_logger.addHandler(logging.NullHandler())


def get_logger(name: str | None = None) -> logging.Logger:
    if name is None or not name.strip():
        return logging.getLogger(PACKAGE_LOGGER_NAME)
    if name == PACKAGE_LOGGER_NAME or name.startswith(f"{PACKAGE_LOGGER_NAME}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{PACKAGE_LOGGER_NAME}.{name.lstrip('.')}")


def configure_logging(level: int | str = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    default_handler = next(
        (
            handler
            for handler in logger.handlers
            if getattr(handler, "_opai_default_handler", False)
        ),
        None,
    )
    if default_handler is None:
        default_handler = logging.StreamHandler()
        default_handler.setFormatter(logging.Formatter(DEFAULT_LOG_FORMAT))
        setattr(default_handler, "_opai_default_handler", True)
        logger.addHandler(default_handler)

    logger.setLevel(level)
    logger.propagate = False
    return logger
