from __future__ import annotations

import pytest

from opai.core.exceptions import OPAIWorkflowError
from opai.domain.context import Context
from opai.domain.session import DemoAsset, MappingAsset, SessionManifest
from opai.infrastructure.context_store import get_demo_videos, get_mapping_video
from opai.infrastructure.persistence import write_session_manifest


def test_get_mapping_video_returns_absolute_stored_path(tmp_path) -> None:
    ctx = _build_context(tmp_path)
    mapping_path = (
        ctx.session_directory / "captures" / "mapping" / "current" / "map.mp4"
    )
    mapping_path.parent.mkdir(parents=True)
    mapping_path.write_bytes(b"mapping")
    write_session_manifest(
        ctx.manifest_path,
        SessionManifest(
            session_name=ctx.name,
            demos=(),
            mapping=MappingAsset(
                source_path="/tmp/source-map.mp4",
                stored_path="captures/mapping/current/map.mp4",
                original_filename="map.mp4",
            ),
        ),
    )

    assert get_mapping_video(ctx) == mapping_path.resolve()


def test_get_mapping_video_returns_none_when_manifest_has_no_mapping(tmp_path) -> None:
    ctx = _build_context(tmp_path)
    write_session_manifest(
        ctx.manifest_path,
        SessionManifest(session_name=ctx.name, demos=(), mapping=None),
    )

    assert get_mapping_video(ctx) is None


def test_get_mapping_video_raises_when_stored_file_is_missing(tmp_path) -> None:
    ctx = _build_context(tmp_path)
    write_session_manifest(
        ctx.manifest_path,
        SessionManifest(
            session_name=ctx.name,
            demos=(),
            mapping=MappingAsset(
                source_path="/tmp/source-map.mp4",
                stored_path="captures/mapping/current/missing.mp4",
                original_filename="missing.mp4",
            ),
        ),
    )

    with pytest.raises(OPAIWorkflowError, match="mapping video is missing") as exc_info:
        get_mapping_video(ctx)

    assert exc_info.value.details == {
        "session_name": "session-001",
        "asset_kind": "mapping",
        "stored_path": "captures/mapping/current/missing.mp4",
        "resolved_path": str(
            (
                ctx.session_directory
                / "captures"
                / "mapping"
                / "current"
                / "missing.mp4"
            ).resolve()
        ),
    }


def test_get_demo_videos_returns_absolute_paths_in_manifest_order(tmp_path) -> None:
    ctx = _build_context(tmp_path)
    demo_a = ctx.session_directory / "captures" / "demos" / "demo-0002" / "b.mp4"
    demo_b = ctx.session_directory / "captures" / "demos" / "demo-0001" / "a.mp4"
    demo_a.parent.mkdir(parents=True)
    demo_b.parent.mkdir(parents=True)
    demo_a.write_bytes(b"b")
    demo_b.write_bytes(b"a")
    write_session_manifest(
        ctx.manifest_path,
        SessionManifest(
            session_name=ctx.name,
            demos=(
                DemoAsset(
                    demo_id="demo-0002",
                    source_path="/tmp/b.mp4",
                    stored_path="captures/demos/demo-0002/b.mp4",
                    original_filename="b.mp4",
                ),
                DemoAsset(
                    demo_id="demo-0001",
                    source_path="/tmp/a.mp4",
                    stored_path="captures/demos/demo-0001/a.mp4",
                    original_filename="a.mp4",
                ),
            ),
            mapping=None,
        ),
    )

    assert get_demo_videos(ctx) == [demo_a.resolve(), demo_b.resolve()]


def test_get_demo_videos_returns_empty_list_when_manifest_has_no_demos(
    tmp_path,
) -> None:
    ctx = _build_context(tmp_path)
    write_session_manifest(
        ctx.manifest_path,
        SessionManifest(session_name=ctx.name, demos=(), mapping=None),
    )

    assert get_demo_videos(ctx) == []


def test_get_demo_videos_raises_when_any_stored_file_is_missing(tmp_path) -> None:
    ctx = _build_context(tmp_path)
    demo_path = ctx.session_directory / "captures" / "demos" / "demo-0001" / "a.mp4"
    demo_path.parent.mkdir(parents=True)
    demo_path.write_bytes(b"a")
    write_session_manifest(
        ctx.manifest_path,
        SessionManifest(
            session_name=ctx.name,
            demos=(
                DemoAsset(
                    demo_id="demo-0001",
                    source_path="/tmp/a.mp4",
                    stored_path="captures/demos/demo-0001/a.mp4",
                    original_filename="a.mp4",
                ),
                DemoAsset(
                    demo_id="demo-0002",
                    source_path="/tmp/missing.mp4",
                    stored_path="captures/demos/demo-0002/missing.mp4",
                    original_filename="missing.mp4",
                ),
            ),
            mapping=None,
        ),
    )

    with pytest.raises(OPAIWorkflowError, match="demo video is missing") as exc_info:
        get_demo_videos(ctx)

    assert exc_info.value.details == {
        "session_name": "session-001",
        "asset_kind": "demo",
        "stored_path": "captures/demos/demo-0002/missing.mp4",
        "resolved_path": str(
            (
                ctx.session_directory
                / "captures"
                / "demos"
                / "demo-0002"
                / "missing.mp4"
            ).resolve()
        ),
        "asset_id": "demo-0002",
    }


def _build_context(tmp_path) -> Context:
    session_directory = tmp_path / "sessions" / "session-001"
    session_directory.mkdir(parents=True)
    return Context(
        name="session-001",
        session_directory=session_directory,
        manifest_path=session_directory / "session.json",
    )
