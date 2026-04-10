from __future__ import annotations

import json
import sys
from builtins import __import__ as builtin_import
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

import opai
from opai.application import calibration as calibration_module
from opai.application import slam as slam_module
from opai.core.exceptions import (
    OPAIContextError,
    OPAIDependencyError,
    OPAIValidationError,
    OPAIWorkflowError,
)
from opai.infrastructure import context_store


def test_calibrate_requires_context() -> None:
    with pytest.raises(OPAIContextError, match="Call opai.init"):
        opai.calibrate([], 3, 3, 1.0, 0.5, "DICT_4X4_50")


def test_calibrate_with_video_requires_context(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    video_path = tmp_path / "demo.mp4"
    video_path.write_bytes(b"demo")

    with pytest.raises(OPAIContextError, match="Call opai.init"):
        opai.calibrate_with_video(video_path, 2, 3, 3, 1.0, 0.5, "DICT_4X4_50")


def test_verify_calibrated_parameters_requires_context(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(context_store, "_ACTIVE_CONTEXT", None)
    video_path = tmp_path / "demo.mp4"
    video_path.write_bytes(b"demo")

    with pytest.raises(OPAIContextError, match="Call opai.init"):
        opai.verify_calibrated_parameters(
            video_path=video_path,
            n_check_imgs=2,
            charuco_config_json={},
            intrinsics_json={},
        )


def test_run_extract_trajectories_batch_requires_context(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(context_store, "_ACTIVE_CONTEXT", None)

    with pytest.raises(OPAIContextError, match="Call opai.init"):
        opai.run_extract_trajectories_batch()


def test_run_extract_trajectories_batch_uses_active_context(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    ctx = opai.init("session-001")
    expected = {"processed_videos": [], "total_processed": 0}

    def fake_run_extract_trajectories_batch(*, ctx):
        assert ctx == opai.get_context()
        return expected

    monkeypatch.setattr(
        slam_module,
        "run_extract_trajectories_batch",
        fake_run_extract_trajectories_batch,
    )

    assert opai.run_extract_trajectories_batch() == expected
    assert opai.get_context() == ctx


def test_init_creates_context_directory(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    ctx = opai.init("session-001")

    assert ctx.name == "session-001"
    assert ctx.session_directory.exists()
    assert ctx.manifest_path == tmp_path / "sessions" / "session-001" / "session.json"
    assert ctx.manifest_path.exists()
    assert (ctx.session_directory / "captures" / "demos").exists()
    assert (ctx.session_directory / "captures" / "mapping").exists()
    assert (ctx.session_directory / "gopro_thumbnails").exists()


def test_init_resumes_existing_session_manifest(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    session_dir = tmp_path / "sessions" / "session-001"
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


def test_calibrate_writes_artifact_without_plotting_by_default(
    tmp_path, monkeypatch
) -> None:
    fake_cv2 = _build_fake_cv2()
    monkeypatch.setattr(calibration_module, "cv2", fake_cv2)
    monkeypatch.chdir(tmp_path)
    opai.init("session-001")
    monkeypatch.setattr(
        calibration_module,
        "plot_frames",
        lambda *_args, **_kwargs: pytest.fail(
            "plot_frames should not run when plot_result=False"
        ),
    )

    frame = np.zeros((10, 12, 3), dtype=np.uint8)
    result = opai.calibrate(
        [frame],
        3,
        3,
        1.0,
        0.5,
        "DICT_4X4_50",
    )

    output_path = tmp_path / "sessions" / "session-001" / "calibration.json"
    assert output_path.exists()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["intrinsic_type"] == "FISHEYE"
    assert payload["image_height"] == 10
    assert payload["image_width"] == 12
    assert result.intrinsic_type == "FISHEYE"


def test_calibrate_accepts_custom_plot_grid(tmp_path, monkeypatch) -> None:
    fake_cv2 = _build_fake_cv2()
    monkeypatch.setattr(calibration_module, "cv2", fake_cv2)
    monkeypatch.chdir(tmp_path)
    opai.init("session-001")
    pyplot = _build_fake_pyplot(nrows=1, ncols=5)
    monkeypatch.setitem(sys.modules, "matplotlib", SimpleNamespace(pyplot=pyplot))
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", pyplot)

    frame = np.zeros((10, 12, 3), dtype=np.uint8)
    opai.calibrate(
        [frame],
        3,
        3,
        1.0,
        0.5,
        "DICT_4X4_50",
        plot_result=True,
        plot_nrows=1,
        plot_ncols=5,
    )

    assert pyplot.subplots_calls == [((1, 5), {"figsize": (16.0, 4.5)})]


def test_calibrate_with_video_writes_artifact_without_plotting_by_default(
    tmp_path,
    monkeypatch,
) -> None:
    fake_cv2 = _build_fake_cv2()
    monkeypatch.setattr(calibration_module, "cv2", fake_cv2)
    monkeypatch.chdir(tmp_path)
    opai.init("session-001")

    frames = tuple(
        np.full((10, 12, 3), fill_value=index, dtype=np.uint8) for index in range(5)
    )
    video_path = tmp_path / "demo.mp4"
    video_path.write_bytes(b"demo")
    monkeypatch.setattr(
        calibration_module,
        "sample_video_frames",
        lambda **kwargs: frames,
    )
    monkeypatch.setattr(
        calibration_module,
        "plot_frames",
        lambda *_args, **_kwargs: pytest.fail(
            "plot_frames should not run when plot_result=False"
        ),
    )

    result = opai.calibrate_with_video(
        video_path=video_path,
        frame_sample_step=2,
        row_count=3,
        col_count=3,
        square_length=1.0,
        marker_length=0.5,
        dictionary="DICT_4X4_50",
        plot_result=False,
        plot_nrows=1,
        plot_ncols=5,
    )

    output_path = tmp_path / "sessions" / "session-001" / "calibration.json"
    assert output_path.exists()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["image_height"] == 10
    assert payload["image_width"] == 12
    assert result.intrinsic_type == "FISHEYE"


def test_calibrate_with_video_plots_detected_corners_when_enabled(
    tmp_path,
    monkeypatch,
) -> None:
    fake_cv2 = _build_fake_cv2()
    monkeypatch.setattr(calibration_module, "cv2", fake_cv2)
    monkeypatch.chdir(tmp_path)
    opai.init("session-001")

    frames = tuple(
        np.full((10, 12, 3), fill_value=index, dtype=np.uint8) for index in range(5)
    )
    video_path = tmp_path / "demo.mp4"
    video_path.write_bytes(b"demo")
    monkeypatch.setattr(
        calibration_module,
        "sample_video_frames",
        lambda **kwargs: frames,
    )
    pyplot = _build_fake_pyplot(nrows=1, ncols=5)
    monkeypatch.setitem(sys.modules, "matplotlib", SimpleNamespace(pyplot=pyplot))
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", pyplot)

    result = opai.calibrate_with_video(
        video_path=video_path,
        frame_sample_step=2,
        row_count=3,
        col_count=3,
        square_length=1.0,
        marker_length=0.5,
        dictionary="DICT_4X4_50",
        plot_result=True,
        plot_nrows=1,
        plot_ncols=5,
    )

    output_path = tmp_path / "sessions" / "session-001" / "calibration.json"
    assert output_path.exists()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["image_height"] == 10
    assert payload["image_width"] == 12
    assert result.intrinsic_type == "FISHEYE"
    assert pyplot.subplots_calls == [((1, 5), {"figsize": (16.0, 4.5)})]
    assert pyplot.show_count == 1
    assert pyplot.close_calls == [pyplot.figure]
    assert np.array_equal(pyplot.axes[0].images[0], frames[0][..., ::-1])


def test_verify_calibrated_parameters_uses_session_relative_json_paths(
    tmp_path,
    monkeypatch,
) -> None:
    fake_cv2 = _build_fake_cv2()
    monkeypatch.setattr(calibration_module, "cv2", fake_cv2)
    monkeypatch.chdir(tmp_path)
    ctx = opai.init("session-001")

    frames = tuple(
        np.full((10, 12, 3), fill_value=index, dtype=np.uint8) for index in range(3)
    )
    video_path = tmp_path / "demo.mp4"
    video_path.write_bytes(b"demo")
    monkeypatch.setattr(
        calibration_module,
        "sample_video_frames",
        lambda **_kwargs: frames,
    )
    plot_calls: list[tuple[tuple[np.ndarray, ...], dict[str, object]]] = []

    def fake_plot_frames(
        plotted_frames: tuple[np.ndarray, ...] | list[np.ndarray],
        **kwargs,
    ) -> None:
        plot_calls.append((tuple(plotted_frames), kwargs))

    monkeypatch.setattr(calibration_module, "plot_frames", fake_plot_frames)

    charuco_path = ctx.session_directory / "charuco_config.json"
    charuco_path.write_text(
        json.dumps(
            {
                "dictionary": "DICT_4X4_50",
                "squares_x": 3,
                "squares_y": 3,
                "square_length": 1.0,
                "marker_length": 0.5,
                "image_width_px": 1200,
                "image_height_px": 800,
                "margin_size_px": 20,
                "board_image_path": "charuco_board.png",
            }
        ),
        encoding="utf-8",
    )
    intrinsics_path = ctx.session_directory / "calibration.json"
    intrinsics_path.write_text(
        json.dumps(
            {
                "mse_reproj_error": 0.1,
                "image_height": 10,
                "image_width": 12,
                "intrinsic_type": "FISHEYE",
                "intrinsics": {
                    "aspect_ratio": 0.5,
                    "focal_length": 10.0,
                    "principal_pt_x": 5.0,
                    "principal_pt_y": 6.0,
                    "radial_distortion_1": 0.1,
                    "radial_distortion_2": 0.2,
                    "radial_distortion_3": 0.3,
                    "radial_distortion_4": 0.4,
                    "skew": 0.0,
                },
            }
        ),
        encoding="utf-8",
    )

    result = opai.verify_calibrated_parameters(
        video_path=video_path,
        n_check_imgs=2,
        charuco_config_json="charuco_config.json",
        intrinsics_json="calibration.json",
        plot_result=True,
        plot_nrows=1,
        plot_ncols=2,
    )

    assert result.sampled_image_count == 2
    assert result.verified_image_count == 2
    assert result.total_detected_corner_count == 8
    assert len(plot_calls) == 1
    assert len(plot_calls[0][0]) == 2
    assert plot_calls[0][1] == {"nrows": 1, "ncols": 2, "frames_are_bgr": True}
    output_path = (
        tmp_path / "sessions" / "session-001" / "calibration_verification.json"
    )
    assert output_path.exists()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["verified_image_count"] == 2


def test_calibrate_with_video_requires_matplotlib_for_detected_corner_plotting(
    tmp_path,
    monkeypatch,
) -> None:
    fake_cv2 = _build_fake_cv2()
    monkeypatch.setattr(calibration_module, "cv2", fake_cv2)
    monkeypatch.chdir(tmp_path)
    opai.init("session-001")

    frames = tuple(
        np.full((10, 12, 3), fill_value=index, dtype=np.uint8) for index in range(2)
    )
    video_path = tmp_path / "demo.mp4"
    video_path.write_bytes(b"demo")
    monkeypatch.setattr(
        calibration_module,
        "sample_video_frames",
        lambda **kwargs: frames,
    )

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "matplotlib":
            raise ModuleNotFoundError("No module named 'matplotlib'")
        return builtin_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", fake_import)

    with pytest.raises(OPAIDependencyError, match="matplotlib"):
        opai.calibrate_with_video(
            video_path=video_path,
            frame_sample_step=2,
            row_count=3,
            col_count=3,
            square_length=1.0,
            marker_length=0.5,
            dictionary="DICT_4X4_50",
            plot_result=True,
        )


def test_plot_video_frames_uses_auto_grid_defaults_without_context(
    tmp_path,
    monkeypatch,
) -> None:
    frames = tuple(
        np.full((10, 12, 3), fill_value=index, dtype=np.uint8) for index in range(5)
    )
    video_path = tmp_path / "demo.mp4"
    video_path.write_bytes(b"demo")
    monkeypatch.setattr(context_store, "_ACTIVE_CONTEXT", None)
    monkeypatch.setattr(
        calibration_module,
        "sample_video_frames",
        lambda **kwargs: frames,
    )
    pyplot = _build_fake_pyplot(nrows=2, ncols=3)
    monkeypatch.setitem(sys.modules, "matplotlib", SimpleNamespace(pyplot=pyplot))
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", pyplot)

    opai.plot_video_frames(video_path, frame_sample_step=2)

    assert pyplot.subplots_calls == [((2, 3), {"figsize": (12.0, 6.0)})]
    assert pyplot.show_count == 1
    assert pyplot.close_calls == [pyplot.figure]
    assert pyplot.figure.tight_layout_calls == 1
    assert np.array_equal(
        pyplot.axes[0].images[0],
        frames[0][..., ::-1],
    )
    assert np.shares_memory(pyplot.axes[0].images[0], frames[0])
    assert all(axis.axis_off_calls == 1 for axis in pyplot.axes[:5])
    assert pyplot.axes[5].axis_off_calls == 1


def test_plot_video_frames_accepts_custom_grid(tmp_path, monkeypatch) -> None:
    frames = tuple(
        np.full((10, 12, 3), fill_value=index, dtype=np.uint8) for index in range(5)
    )
    video_path = tmp_path / "demo.mp4"
    video_path.write_bytes(b"demo")
    monkeypatch.setattr(
        calibration_module,
        "sample_video_frames",
        lambda **kwargs: frames,
    )
    pyplot = _build_fake_pyplot(nrows=1, ncols=5)
    monkeypatch.setitem(sys.modules, "matplotlib", SimpleNamespace(pyplot=pyplot))
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", pyplot)

    opai.plot_video_frames(video_path, frame_sample_step=2, nrows=1, ncols=5)

    assert pyplot.subplots_calls[0][0] == (1, 5)
    assert pyplot.subplots_calls[0][1]["figsize"] == pytest.approx((16.0, 4.5))
    assert pyplot.show_count == 1
    assert pyplot.close_calls == [pyplot.figure]
    assert all(axis.axis_off_calls == 1 for axis in pyplot.axes)


def test_plot_video_frames_requires_matplotlib(tmp_path, monkeypatch) -> None:
    video_path = tmp_path / "demo.mp4"
    video_path.write_bytes(b"demo")
    monkeypatch.setattr(
        calibration_module,
        "sample_video_frames",
        lambda **kwargs: (np.zeros((4, 4, 3), dtype=np.uint8),),
    )

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "matplotlib":
            raise ModuleNotFoundError("No module named 'matplotlib'")
        return builtin_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", fake_import)

    with pytest.raises(OPAIDependencyError, match="matplotlib"):
        opai.plot_video_frames(video_path, frame_sample_step=2)


def test_plot_video_frames_propagates_sampling_failure(tmp_path, monkeypatch) -> None:
    video_path = tmp_path / "demo.mp4"
    video_path.write_bytes(b"demo")
    monkeypatch.setattr(
        calibration_module,
        "sample_video_frames",
        lambda **kwargs: (_ for _ in ()).throw(
            OPAIWorkflowError("Calibration failed: video sampling produced no frames.")
        ),
    )

    with pytest.raises(OPAIWorkflowError, match="produced no frames"):
        opai.plot_video_frames(video_path, frame_sample_step=2)


def test_plot_video_frames_downsamples_oversized_frames_before_plotting(
    tmp_path,
    monkeypatch,
) -> None:
    frame = np.arange(4000 * 6000 * 3, dtype=np.uint8).reshape(4000, 6000, 3)
    video_path = tmp_path / "demo.mp4"
    video_path.write_bytes(b"demo")
    monkeypatch.setattr(
        calibration_module,
        "sample_video_frames",
        lambda **kwargs: (frame,),
    )
    pyplot = _build_fake_pyplot(nrows=1, ncols=1)
    monkeypatch.setitem(sys.modules, "matplotlib", SimpleNamespace(pyplot=pyplot))
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", pyplot)

    opai.plot_video_frames(video_path, frame_sample_step=2)

    plotted = pyplot.axes[0].images[0]
    assert plotted.shape == (2000, 3000, 3)
    assert np.shares_memory(plotted, frame)


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
        (tmp_path / "sessions" / "session-001" / "session.json").read_text(
            encoding="utf-8"
        )
    )
    assert [entry["original_filename"] for entry in payload["demos"]] == [
        "demo-a.mp4",
        "demo-b.mp4",
    ]
    assert (
        tmp_path
        / "sessions"
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
        tmp_path / "sessions" / "session-001" / "captures" / "mapping" / "current"
    )
    assert sorted(path.name for path in mapping_dir.iterdir()) == ["mapping-b.mp4"]
    payload = json.loads(
        (tmp_path / "sessions" / "session-001" / "session.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["mapping"]["original_filename"] == "mapping-b.mp4"


def test_list_sessions(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    opai.init("session-b")
    opai.init("session-a")
    recorder = _install_fake_rich(monkeypatch)

    opai.list_sessions()
    tree = recorder["prints"][0][0]

    assert tree.label.startswith("[bold]sessions[/]")
    assert [child.label for child in tree.children] == [
        "[bold cyan]session-a[/] [dim](current, demos=0, mapping=no, files=1)[/]",
        "[bold cyan]session-b[/] [dim](demos=0, mapping=no, files=1)[/]",
    ]


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
    assert tree.label.startswith("[bold]sessions[/]")
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
    def fake_circle(
        image: np.ndarray,
        center: tuple[int, int],
        _radius: int,
        color: tuple[int, int, int],
        _thickness: int,
    ) -> np.ndarray:
        x, y = center
        if 0 <= y < image.shape[0] and 0 <= x < image.shape[1]:
            image[y, x] = np.array(color, dtype=image.dtype)
        return image

    def fake_arrowed_line(
        image: np.ndarray,
        start: tuple[int, int],
        end: tuple[int, int],
        color: tuple[int, int, int],
        _thickness: int,
        *,
        tipLength: float = 0.0,
    ) -> np.ndarray:
        del tipLength
        midpoint = (
            int(round((start[0] + end[0]) / 2)),
            int(round((start[1] + end[1]) / 2)),
        )
        for x, y in (start, midpoint, end):
            if 0 <= y < image.shape[0] and 0 <= x < image.shape[1]:
                image[y, x] = np.array(color, dtype=image.dtype)
        return image

    def fake_fisheye_calibrate(
        object_points,
        image_points,
        image_size,
        camera_matrix,
        dist_coeffs,
        *,
        flags,
        criteria,
    ):
        frame_count = len(object_points)
        assert len(image_points) == frame_count
        assert image_size == (12, 10)
        assert camera_matrix.shape == (3, 3)
        assert dist_coeffs.shape == (4, 1)
        assert flags == 7
        assert criteria == (3, 100, 1e-6)
        return (
            0.1,
            np.array([[10.0, 0.0, 5.0], [0.0, 20.0, 6.0], [0.0, 0.0, 1.0]]),
            np.array([[0.1], [0.2], [0.3], [0.4]]),
            [np.zeros((3, 1), dtype=np.float32) for _ in range(frame_count)],
            [np.zeros((3, 1), dtype=np.float32) for _ in range(frame_count)],
        )

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

    detected_corners = np.array(
        [
            [[1.0, 1.0]],
            [[2.0, 1.0]],
            [[1.0, 2.0]],
            [[2.0, 2.0]],
        ],
        dtype=np.float32,
    )
    detected_ids = np.array([[0], [1], [2], [3]], dtype=np.int32)

    class FakeCharucoDetector:
        def __init__(self, _board) -> None:
            self.board = _board

        def detectBoard(self, _image):
            return detected_corners, detected_ids, None, None

    def fake_cvt_color(frame: np.ndarray, code: int) -> np.ndarray:
        if code == 1:
            return frame[:, :, 0]
        if code == 2:
            return np.repeat(frame[:, :, None], 3, axis=2)
        raise AssertionError(f"Unexpected color conversion code: {code}")

    aruco = SimpleNamespace(
        DICT_4X4_50=1,
        getPredefinedDictionary=lambda _: "dictionary",
        CharucoBoard=lambda *args, **kwargs: board,
        CharucoDetector=FakeCharucoDetector,
        drawDetectedCornersCharuco=lambda **kwargs: kwargs["image"],
    )
    fisheye = SimpleNamespace(
        CALIB_RECOMPUTE_EXTRINSIC=1,
        CALIB_CHECK_COND=2,
        CALIB_FIX_SKEW=4,
        calibrate=fake_fisheye_calibrate,
        projectPoints=lambda object_points, *_args: (object_points[:, :, :2], None),
    )

    return SimpleNamespace(
        aruco=aruco,
        fisheye=fisheye,
        TERM_CRITERIA_EPS=1,
        TERM_CRITERIA_MAX_ITER=2,
        COLOR_BGR2GRAY=1,
        COLOR_GRAY2BGR=2,
        cvtColor=fake_cvt_color,
        arrowedLine=fake_arrowed_line,
        circle=fake_circle,
        solvePnP=lambda *_args, **_kwargs: (
            True,
            np.zeros((3, 1), dtype=np.float64),
            np.zeros((3, 1), dtype=np.float64),
        ),
        waitKey=lambda _delay: 0,
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

    monkeypatch.setitem(sys.modules, "rich", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules, "rich.console", SimpleNamespace(Console=FakeConsole)
    )
    monkeypatch.setitem(sys.modules, "rich.tree", SimpleNamespace(Tree=FakeTree))
    return recorder


def _build_fake_pyplot(*, nrows: int, ncols: int):
    class FakeAxis:
        def __init__(self) -> None:
            self.images: list[np.ndarray] = []
            self.axis_off_calls = 0

        def imshow(self, image: np.ndarray) -> None:
            self.images.append(image)

        def set_axis_off(self) -> None:
            self.axis_off_calls += 1

    class FakeFigure:
        def __init__(self) -> None:
            self.tight_layout_calls = 0

        def tight_layout(self) -> None:
            self.tight_layout_calls += 1

    class FakePyplot:
        def __init__(self) -> None:
            self.figure = FakeFigure()
            self.axes = [FakeAxis() for _ in range(nrows * ncols)]
            self.subplots_calls: list[tuple[tuple[int, int], dict[str, object]]] = []
            self.show_count = 0
            self.close_calls: list[FakeFigure] = []

        def subplots(self, rows: int, cols: int, **kwargs):
            self.subplots_calls.append(((rows, cols), kwargs))
            return self.figure, np.array(self.axes, dtype=object).reshape(rows, cols)

        def show(self) -> None:
            self.show_count += 1

        def close(self, figure) -> None:
            self.close_calls.append(figure)

    return FakePyplot()


def _install_fake_ipywidgets(
    monkeypatch: pytest.MonkeyPatch,
    *,
    selected_indexes: list[int],
    tracker: dict[str, object] | None = None,
) -> None:
    ipywidgets_module = ModuleType("ipywidgets")
    ipywidgets_module.Layout = _FakeLayout
    ipywidgets_module.HTML = _FakeHTMLWidget
    ipywidgets_module.Image = _FakeImageWidget
    ipywidgets_module.Checkbox = _FakeCheckboxWidget
    ipywidgets_module.Button = _FakeButtonWidget
    ipywidgets_module.VBox = _FakeBoxWidget
    ipywidgets_module.HBox = _FakeBoxWidget
    ipywidgets_module.Box = _FakeBoxWidget

    ipython_module = ModuleType("IPython")

    display_module = ModuleType("IPython.display")

    def fake_display(browser) -> None:
        if tracker is not None:
            tracker["browser"] = browser
            tracker["browser_was_open_during_display"] = not browser.closed
        cards_box = browser.children[2]
        for index in selected_indexes:
            cards_box.children[index].children[2].set_value(True)
        browser.children[3].children[0].click()

    display_module.display = fake_display

    monkeypatch.setitem(sys.modules, "ipywidgets", ipywidgets_module)
    monkeypatch.setitem(sys.modules, "IPython", ipython_module)
    monkeypatch.setitem(sys.modules, "IPython.display", display_module)


class _FakeLayout:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


class _FakeHTMLWidget:
    def __init__(self, value: str = "", layout=None) -> None:
        self.value = value
        self.layout = layout


class _FakeImageWidget:
    def __init__(self, value: bytes, format: str, layout=None) -> None:
        self.value = value
        self.format = format
        self.layout = layout


class _FakeCheckboxWidget:
    def __init__(self, value: bool, description: str, indent: bool) -> None:
        self.value = value
        self.description = description
        self.indent = indent
        self._observers = []

    def observe(self, callback, names: str) -> None:
        self._observers.append((callback, names))

    def set_value(self, value: bool) -> None:
        self.value = value
        for callback, _names in self._observers:
            callback({"new": value})


class _FakeButtonWidget:
    def __init__(self, description: str, button_style: str = "") -> None:
        self.description = description
        self.button_style = button_style
        self._callbacks = []

    def on_click(self, callback) -> None:
        self._callbacks.append(callback)

    def click(self) -> None:
        for callback in self._callbacks:
            callback(self)


class _FakeBoxWidget:
    def __init__(self, children, layout=None) -> None:
        self.children = list(children)
        self.layout = layout
        self.closed = False

    def close(self) -> None:
        self.closed = True
