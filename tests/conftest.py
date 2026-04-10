from __future__ import annotations

import pytest

from opai.infrastructure import context_store


@pytest.fixture(autouse=True)
def reset_active_context() -> None:
    context_store._ACTIVE_CONTEXT = None
    yield
    context_store._ACTIVE_CONTEXT = None
