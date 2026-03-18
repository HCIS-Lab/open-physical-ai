from __future__ import annotations

import json
import sys
from builtins import __import__ as builtin_import
from types import SimpleNamespace

import numpy as np
import pytest

import opai
from opai.application import calibration as calibration_module
from opai.core.exceptions import (
    OPAIContextError,
    OPAIDependencyError,
    OPAIValidationError,
)
from opai.infrastructure import context_store


def test_calibrate_requires_context() -> None:
    with pytest.raises(OPAIContextError, match="Call opai.init"):
        opai.calibrate([], 3, 3, 1.0, 0.5, "DICT_4X4_50")


def test_init_creates_context_directory(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    ctx = opai.init("session-001")

    assert ctx.name == "session-001"
    assert ctx.session_directory.exists()
    assert (
        ctx.manifest_path
        == tmp_path / ".opai_sessions" / "session-001" / "session.json"
    )
    assert ctx.manifest_path.exists()
    assert (ctx.session_directory / "captures" / "demos").exists()
    assert (ctx.session_directory / "captures" / "mapping").exists()


def test_init_resumes_existing_session_manifest(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    session_dir = tmp_path / ".opai_sessions" / "session-001"
    session_dir.mkdir(parents=True)
    manifest_path = session_dir / "session.json"
    manifest_path.write_text(
        json.dumps(
            {
                "session_name": "session-001",
                "demos": [
                    {
                        "demo_id": "demo-0001",
                        "source_path": "/tmp/source.mp4",
                        "stored_path": "captures/demos/demo-0001/source.mp4",
                        "original_filename": "source.mp4",
                    }
                ],
                "mapping": None,
            }
        ),
        encoding="utf-8",
    )

    ctx = opai.init("session-001")

    payload = json.loads(ctx.manifest_path.read_text(encoding="utf-8"))
    assert payload["demos"][0]["demo_id"] == "demo-0001"


def test_init_rejects_invalid_session_name() -> None:
    with pytest.raises(OPAIValidationError, match="may only contain"):
        opai.init("../bad-session")


def test_calibrate_writes_artifact(tmp_path, monkeypatch) -> None:
    fake_cv2 = _build_fake_cv2()
    monkeypatch.setattr(calibration_module, "cv2", fake_cv2)
    monkeypatch.chdir(tmp_path)
    opai.init("session-001")

    frame = np.zeros((10, 12, 3), dtype=np.uint8)
    result = opai.calibrate(
        [frame],
        3,
        3,
        1.0,
        0.5,
        "DICT_4X4_50",
    )

    output_path = tmp_path / ".opai_sessions" / "session-001" / "calibration.json"
    assert output_path.exists()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["intrinsic_type"] == "FISHEYE"
    assert payload["image_height"] == 10
    assert payload["image_width"] == 12
    assert result.intrinsic_type == "FISHEYE"


def test_add_demos_requires_context(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(context_store, "_ACTIVE_CONTEXT", None)
    demo_path = tmp_path / "demo.mp4"
    demo_path.write_bytes(b"demo")

    with pytest.raises(OPAIContextError, match="Call opai.init"):
        opai.add_demos([demo_path])


def test_add_demos_copies_files_and_preserves_order(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    opai.init("session-001")
    demo_a = tmp_path / "demo-a.mp4"
    demo_b = tmp_path / "demo-b.mp4"
    demo_a.write_bytes(b"a")
    demo_b.write_bytes(b"b")

    assets = opai.add_demos([demo_a, demo_b])

    assert [asset.demo_id for asset in assets] == ["demo-0001", "demo-0002"]
    payload = json.loads(
        (tmp_path / ".opai_sessions" / "session-001" / "session.json").read_text(
            encoding="utf-8"
        )
    )
    assert [entry["original_filename"] for entry in payload["demos"]] == [
        "demo-a.mp4",
        "demo-b.mp4",
    ]
    assert (
        tmp_path
        / ".opai_sessions"
        / "session-001"
        / "captures"
        / "demos"
        / "demo-0001"
        / "demo-a.mp4"
    ).read_bytes() == b"a"


def test_add_mapping_replaces_active_mapping(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    opai.init("session-001")
    mapping_a = tmp_path / "mapping-a.mp4"
    mapping_b = tmp_path / "mapping-b.mp4"
    mapping_a.write_bytes(b"first")
    mapping_b.write_bytes(b"second")

    first = opai.add_mapping(mapping_a)
    second = opai.add_mapping(mapping_b)

    assert first.original_filename == "mapping-a.mp4"
    assert second.original_filename == "mapping-b.mp4"
    mapping_dir = (
        tmp_path / ".opai_sessions" / "session-001" / "captures" / "mapping" / "current"
    )
    assert sorted(path.name for path in mapping_dir.iterdir()) == ["mapping-b.mp4"]
    payload = json.loads(
        (tmp_path / ".opai_sessions" / "session-001" / "session.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["mapping"]["original_filename"] == "mapping-b.mp4"


def test_list_sessions_returns_names(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    opai.init("session-b")
    opai.init("session-a")
    recorder = _install_fake_rich(monkeypatch)

    assert opai.list_sessions() == ["session-a", "session-b"]
    tree = recorder["prints"][0][0]
    assert tree.label.startswith("[bold].opai_sessions[/]")
    assert [child.label for child in tree.children] == [
        "[bold cyan]session-a[/] [dim](current, demos=0, mapping=no, files=1)[/]",
        "[bold cyan]session-b[/] [dim](demos=0, mapping=no, files=1)[/]",
    ]


def test_list_sessions_shows_empty_state(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    recorder = _install_fake_rich(monkeypatch)

    assert opai.list_sessions() == []
    tree = recorder["prints"][0][0]
    assert tree.label.startswith("[bold].opai_sessions[/]")
    assert [child.label for child in tree.children] == ["[yellow]No sessions found[/]"]


def test_list_sessions_requires_rich(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("rich"):
            raise ModuleNotFoundError("No module named 'rich'")
        return builtin_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", fake_import)

    with pytest.raises(OPAIDependencyError, match="rich"):
        opai.list_sessions()


def test_browse_session_returns_files_without_changing_active_context(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    ctx = opai.init("session-001")
    demo_path = tmp_path / "demo.mp4"
    demo_path.write_bytes(b"demo")
    opai.add_demos([demo_path])
    recorder = _install_fake_rich(monkeypatch)

    files = opai.browse_session("session-001")

    assert "captures/demos/demo-0001/demo.mp4" in files
    assert opai.get_context().name == ctx.name
    tree = recorder["prints"][0][0]
    assert tree.label.startswith("[bold].opai_sessions[/]")
    session_branch = tree.children[0]
    assert session_branch.label == (
        "[bold magenta]session-001[/] "
        "[dim](path=session-001, demos=1, mapping=no, files=2)[/]"
    )
    assert (
        session_branch.children[0].label
        == f"[dim]path:[/] [cyan]{ctx.session_directory}[/]"
    )
    assert [child.label for child in session_branch.children[1:]] == [
        "[bold blue]captures/[/]",
        "[green]session.json[/]",
    ]


def test_browse_session_requires_rich(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    opai.init("session-001")

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("rich"):
            raise ModuleNotFoundError("No module named 'rich'")
        return builtin_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", fake_import)

    with pytest.raises(OPAIDependencyError, match="rich"):
        opai.browse_session("session-001")


def _build_fake_cv2() -> SimpleNamespace:
    board = SimpleNamespace(
        getChessboardCorners=lambda: np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
    )

    aruco = SimpleNamespace(
        DICT_4X4_50=1,
        getPredefinedDictionary=lambda _: "dictionary",
        CharucoBoard=lambda *args, **kwargs: board,
        detectMarkers=lambda *args, **kwargs: (
            [np.zeros((4, 1, 2), dtype=np.float32)],
            np.array([[0]], dtype=np.int32),
            None,
        ),
        interpolateCornersCharuco=lambda **kwargs: (
            4,
            np.array(
                [
                    [[1.0, 1.0]],
                    [[2.0, 1.0]],
                    [[1.0, 2.0]],
                    [[2.0, 2.0]],
                ],
                dtype=np.float32,
            ),
            np.array([[0], [1], [2], [3]], dtype=np.int32),
        ),
        calibrateCameraCharuco=lambda **kwargs: (
            0.1,
            np.array([[10.0, 0.0, 5.0], [0.0, 20.0, 6.0], [0.0, 0.0, 1.0]]),
            np.array([0.1, 0.2, 0.3, 0.4]),
            [np.zeros((3, 1), dtype=np.float32)],
            [np.zeros((3, 1), dtype=np.float32)],
        ),
    )

    return SimpleNamespace(
        aruco=aruco,
        COLOR_BGR2GRAY=1,
        cvtColor=lambda frame, _: frame[:, :, 0],
        projectPoints=lambda **kwargs: (kwargs["objectPoints"][:, :, :2], None),
    )


def _install_fake_rich(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, list[tuple[object, ...]]]:
    recorder: dict[str, list[tuple[object, ...]]] = {"prints": []}

    class FakeTree:
        def __init__(self, label: str, **kwargs) -> None:
            self.label = label
            self.kwargs = kwargs
            self.children = []

        def add(self, label: str):
            child = FakeTree(label)
            self.children.append(child)
            return child

    class FakeConsole:
        def print(self, *_args, **_kwargs) -> None:
            recorder["prints"].append(_args)
            return None

    monkeypatch.setitem(sys.modules, "rich", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules, "rich.console", SimpleNamespace(Console=FakeConsole)
    )
    monkeypatch.setitem(sys.modules, "rich.tree", SimpleNamespace(Tree=FakeTree))
    return recorder
