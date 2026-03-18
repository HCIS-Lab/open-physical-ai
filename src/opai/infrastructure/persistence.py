from __future__ import annotations

import json
import shutil
from collections.abc import Sequence
from pathlib import Path

from opai.domain.calibration import CalibrationResult
from opai.domain.gopro import GPThumbnail, GPThumbnailIndex
from opai.domain.session import DemoAsset, MappingAsset, SessionManifest


def write_calibration_result(
    session_directory: Path,
    result: CalibrationResult,
    filename: str = "calibration.json",
) -> Path:
    payload = {
        "mse_reproj_error": result.mse_reproj_error,
        "image_height": result.image_height,
        "image_width": result.image_width,
        "intrinsic_type": result.intrinsic_type,
        "intrinsics": {
            "aspect_ratio": result.intrinsics.aspect_ratio,
            "focal_length": result.intrinsics.focal_length,
            "principal_pt_x": result.intrinsics.principal_pt_x,
            "principal_pt_y": result.intrinsics.principal_pt_y,
            "radial_distortion_1": result.intrinsics.radial_distortion_1,
            "radial_distortion_2": result.intrinsics.radial_distortion_2,
            "radial_distortion_3": result.intrinsics.radial_distortion_3,
            "radial_distortion_4": result.intrinsics.radial_distortion_4,
            "skew": result.intrinsics.skew,
        },
    }

    output_path = session_directory / filename
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def load_session_manifest(manifest_path: Path, session_name: str) -> SessionManifest:
    if not manifest_path.exists():
        return SessionManifest(session_name=session_name, demos=(), mapping=None)

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    demos = tuple(
        DemoAsset(
            demo_id=entry["demo_id"],
            source_path=entry["source_path"],
            stored_path=entry["stored_path"],
            original_filename=entry["original_filename"],
        )
        for entry in payload.get("demos", [])
    )
    mapping_payload = payload.get("mapping")
    mapping = None
    if mapping_payload is not None:
        mapping = MappingAsset(
            source_path=mapping_payload["source_path"],
            stored_path=mapping_payload["stored_path"],
            original_filename=mapping_payload["original_filename"],
        )
    return SessionManifest(
        session_name=payload.get("session_name", session_name),
        demos=demos,
        mapping=mapping,
    )


def write_session_manifest(manifest_path: Path, manifest: SessionManifest) -> Path:
    payload = {
        "session_name": manifest.session_name,
        "demos": [
            {
                "demo_id": demo.demo_id,
                "source_path": demo.source_path,
                "stored_path": demo.stored_path,
                "original_filename": demo.original_filename,
            }
            for demo in manifest.demos
        ],
        "mapping": None,
    }
    if manifest.mapping is not None:
        payload["mapping"] = {
            "source_path": manifest.mapping.source_path,
            "stored_path": manifest.mapping.stored_path,
            "original_filename": manifest.mapping.original_filename,
        }

    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path


def load_gopro_thumbnail_index(index_path: Path) -> GPThumbnailIndex:
    if not index_path.exists():
        return GPThumbnailIndex(items=[])

    payload = json.loads(index_path.read_text(encoding="utf-8"))
    return GPThumbnailIndex(
        items=[
            GPThumbnail(
                media_path=entry["media_path"],
                source_directory=entry["source_directory"],
                source_filename=entry["source_filename"],
                thumbnail_path=entry["thumbnail_path"],
            )
            for entry in payload.get("items", [])
        ]
    )


def write_gopro_thumbnail_index(index_path: Path, index: GPThumbnailIndex) -> Path:
    payload = {
        "items": [
            {
                "media_path": item.media_path,
                "source_directory": item.source_directory,
                "source_filename": item.source_filename,
                "thumbnail_path": item.thumbnail_path,
            }
            for item in index.items
        ]
    }
    index_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return index_path


def copy_demo_assets(
    session_directory: Path,
    current_demos: Sequence[DemoAsset],
    source_paths: Sequence[Path],
) -> tuple[DemoAsset, ...]:
    demo_root = session_directory / "captures" / "demos"
    next_index = len(current_demos) + 1
    copied_assets: list[DemoAsset] = []
    for source_path in source_paths:
        demo_id = f"demo-{next_index:04d}"
        next_index += 1
        destination_directory = demo_root / demo_id
        destination_directory.mkdir(parents=True, exist_ok=False)
        destination_path = destination_directory / source_path.name
        shutil.copy2(source_path, destination_path)
        copied_assets.append(
            DemoAsset(
                demo_id=demo_id,
                source_path=str(source_path.resolve()),
                stored_path=str(destination_path.relative_to(session_directory)),
                original_filename=source_path.name,
            )
        )
    return tuple(copied_assets)


def copy_mapping_asset(session_directory: Path, source_path: Path) -> MappingAsset:
    mapping_root = session_directory / "captures" / "mapping" / "current"
    if mapping_root.exists():
        shutil.rmtree(mapping_root)
    mapping_root.mkdir(parents=True, exist_ok=True)
    destination_path = mapping_root / source_path.name
    shutil.copy2(source_path, destination_path)
    return MappingAsset(
        source_path=str(source_path.resolve()),
        stored_path=str(destination_path.relative_to(session_directory)),
        original_filename=source_path.name,
    )


def list_relative_paths(session_directory: Path) -> list[str]:
    entries: list[str] = []
    for path in sorted(
        session_directory.rglob("*"),
        key=lambda candidate: candidate.relative_to(session_directory).as_posix(),
    ):
        if path.is_file():
            entries.append(path.relative_to(session_directory).as_posix())
    return entries


def build_file_tree(session_directory: Path) -> dict[str, dict]:
    tree: dict[str, dict] = {}
    for relative_path in list_relative_paths(session_directory):
        parts = relative_path.split("/")
        cursor = tree
        for part in parts:
            cursor = cursor.setdefault(part, {})
    return tree
