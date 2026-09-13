"""Build a frozen, deterministic view manifest for local NVS evaluation.

The source dataset manifest is the same newline-delimited list of camera JSON
files consumed by :class:`uttt_nvs.data.loader.NVSDataset`.  A view manifest stores
the exact, ordered frame indices selected for every scene so every checkpoint
is evaluated on identical inputs and targets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Sequence


VIEW_MANIFEST_SCHEMA_VERSION = 1
DEFAULT_NUM_VIEWS = 24
DEFAULT_VIEW_SEED = 9595
SELECTION_METHOD = "per_scene_seeded_sample"


class ViewManifestError(ValueError):
    """Raised when a dataset or frozen view manifest is invalid."""


def _require_local_path(path: str | Path, *, kind: str) -> Path:
    value = str(path)
    if "://" in value:
        raise ViewManifestError(
            f"{kind} must be local for reproducible evaluation, got {value!r}"
        )
    return Path(value).expanduser().resolve()


def read_local_dataset_manifest(path: str | Path) -> list[Path]:
    """Read and resolve the camera JSON paths in a local dataset manifest."""

    manifest_path = _require_local_path(path, kind="dataset manifest")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"dataset manifest does not exist: {manifest_path}")

    camera_paths: list[Path] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            value = line.strip()
            if not value:
                continue
            if "://" in value:
                raise ViewManifestError(
                    f"{manifest_path}:{line_number}: expected a local camera JSON "
                    f"path, got {value!r}"
                )
            camera_path = Path(value).expanduser()
            if not camera_path.is_absolute():
                camera_path = manifest_path.parent / camera_path
            camera_paths.append(camera_path.resolve())

    if not camera_paths:
        raise ViewManifestError(f"dataset manifest is empty: {manifest_path}")
    return camera_paths


def frame_id(frame: dict[str, Any], frame_index: int) -> str:
    """Return a stable, human-readable ID for one camera JSON frame."""

    for key in ("frame_id", "id", "image_id"):
        if key in frame and frame[key] is not None:
            return str(frame[key])
    if frame.get("file_path") is not None:
        return str(frame["file_path"])
    return str(frame_index)


def scene_id(camera_payload: dict[str, Any], camera_path: Path) -> str:
    """Return the scene ID, falling back to the camera file's parent folder."""

    for key in ("scene_id", "id", "name"):
        if key in camera_payload and camera_payload[key] not in (None, ""):
            return str(camera_payload[key])

    generic_camera_names = {
        "camera",
        "cameras",
        "metadata",
        "transforms",
        "transforms_train",
        "transforms_test",
    }
    camera_stem = camera_path.stem.lower()
    is_generic_camera_file = (
        camera_stem in generic_camera_names
        or camera_stem.startswith("opencv_camera")
    )
    if is_generic_camera_file and camera_path.parent.name:
        # GSO stores its camera JSON below ``<scene>/kai/`` whereas DL3DV
        # stores it directly below ``<scene>/``. ``kai`` is a renderer
        # namespace rather than the scene identifier.
        if camera_path.parent.name.lower() == "kai":
            return camera_path.parent.parent.name
        return camera_path.parent.name
    return camera_path.stem


def select_ordered_frame_indices(
    num_frames: int,
    *,
    num_views: int,
    seed: int,
    scene_identifier: str,
) -> list[int]:
    """Select ``num_views`` unique indices in a frozen random-sample order.

    A stable per-scene seed makes selection independent of dataset iteration
    order.  We intentionally retain ``random.sample``'s returned order: the
    existing evaluation protocol interprets View 1..23 as progressively more
    random context, not as a temporal sequence.
    """

    if num_views <= 0:
        raise ViewManifestError(f"num_views must be positive, got {num_views}")
    if num_frames < num_views:
        raise ViewManifestError(
            f"scene {scene_identifier!r} has {num_frames} frames, but "
            f"{num_views} ordered views are required"
        )
    digest = hashlib.sha256(
        f"{seed}\0{scene_identifier}".encode("utf-8")
    ).digest()
    per_scene_seed = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return random.Random(per_scene_seed).sample(range(num_frames), num_views)


def _read_camera_json(camera_path: Path) -> tuple[dict[str, Any], str]:
    if not camera_path.is_file():
        raise FileNotFoundError(f"camera JSON does not exist: {camera_path}")
    raw = camera_path.read_bytes()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ViewManifestError(f"invalid camera JSON {camera_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ViewManifestError(
            f"camera JSON must contain an object at the top level: {camera_path}"
        )
    frames = payload.get("frames")
    if not isinstance(frames, list):
        raise ViewManifestError(
            f"camera JSON is missing a list-valued 'frames': {camera_path}"
        )
    return payload, hashlib.sha256(raw).hexdigest()


def build_view_manifest(
    dataset_manifest: str | Path,
    *,
    dataset: str,
    num_views: int = DEFAULT_NUM_VIEWS,
    seed: int = DEFAULT_VIEW_SEED,
) -> dict[str, Any]:
    """Build (but do not write) a deterministic view manifest dictionary."""

    dataset_manifest_path = _require_local_path(
        dataset_manifest, kind="dataset manifest"
    )
    camera_paths = read_local_dataset_manifest(dataset_manifest_path)
    source_sha256 = hashlib.sha256(dataset_manifest_path.read_bytes()).hexdigest()

    scenes: list[dict[str, Any]] = []
    seen_scene_ids: dict[str, Path] = {}
    for scene_index, camera_path in enumerate(camera_paths):
        payload, camera_sha256 = _read_camera_json(camera_path)
        identifier = scene_id(payload, camera_path)
        if identifier in seen_scene_ids:
            raise ViewManifestError(
                f"duplicate scene_id {identifier!r}: {seen_scene_ids[identifier]} "
                f"and {camera_path}"
            )
        seen_scene_ids[identifier] = camera_path

        frames = payload["frames"]
        selected_indices = select_ordered_frame_indices(
            len(frames),
            num_views=num_views,
            seed=seed,
            scene_identifier=identifier,
        )
        selected_ids = [
            frame_id(frames[index], index) for index in selected_indices
        ]
        scenes.append(
            {
                "scene_index": scene_index,
                "scene_id": identifier,
                "camera_path": str(camera_path),
                "camera_sha256": camera_sha256,
                "frame_indices": selected_indices,
                "frame_ids": selected_ids,
            }
        )

    return {
        "schema_version": VIEW_MANIFEST_SCHEMA_VERSION,
        "dataset": str(dataset),
        "seed": int(seed),
        "num_views": int(num_views),
        "selection_method": SELECTION_METHOD,
        "source_manifest": str(dataset_manifest_path),
        "source_manifest_sha256": source_sha256,
        "view_semantics": {
            "input": 0,
            "targets": list(range(1, num_views)),
        },
        "scenes": scenes,
    }


def validate_view_manifest(
    manifest: dict[str, Any],
    *,
    expected_num_views: int | None = None,
) -> dict[str, Any]:
    """Validate a loaded view manifest and return it unchanged."""

    if not isinstance(manifest, dict):
        raise ViewManifestError("view manifest must be a JSON object")
    if manifest.get("schema_version") != VIEW_MANIFEST_SCHEMA_VERSION:
        raise ViewManifestError(
            "unsupported view manifest schema_version "
            f"{manifest.get('schema_version')!r}; expected "
            f"{VIEW_MANIFEST_SCHEMA_VERSION}"
        )

    num_views = manifest.get("num_views")
    if not isinstance(num_views, int) or isinstance(num_views, bool) or num_views <= 0:
        raise ViewManifestError(
            f"view manifest num_views must be a positive integer, got {num_views!r}"
        )
    if expected_num_views is not None and num_views != expected_num_views:
        raise ViewManifestError(
            f"view manifest contains {num_views} views; expected "
            f"{expected_num_views}"
        )

    semantics = manifest.get("view_semantics")
    expected_targets = list(range(1, num_views))
    if not isinstance(semantics, dict) or semantics.get("input") != 0:
        raise ViewManifestError("view manifest must designate View 0 as input")
    if semantics.get("targets") != expected_targets:
        raise ViewManifestError(
            f"view manifest targets must be View 1..{num_views - 1}"
        )

    scenes = manifest.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        raise ViewManifestError("view manifest must contain at least one scene")

    seen_scene_ids: set[str] = set()
    for expected_scene_index, entry in enumerate(scenes):
        prefix = f"scenes[{expected_scene_index}]"
        if not isinstance(entry, dict):
            raise ViewManifestError(f"{prefix} must be an object")
        if entry.get("scene_index") != expected_scene_index:
            raise ViewManifestError(
                f"{prefix}.scene_index must be {expected_scene_index}, got "
                f"{entry.get('scene_index')!r}"
            )
        identifier = entry.get("scene_id")
        if not isinstance(identifier, str) or not identifier:
            raise ViewManifestError(f"{prefix}.scene_id must be a non-empty string")
        if identifier in seen_scene_ids:
            raise ViewManifestError(f"duplicate scene_id {identifier!r}")
        seen_scene_ids.add(identifier)

        indices = entry.get("frame_indices")
        if (
            not isinstance(indices, list)
            or len(indices) != num_views
            or any(
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
                for value in indices
            )
        ):
            raise ViewManifestError(
                f"{prefix}.frame_indices must contain {num_views} "
                "non-negative integers"
            )
        if len(set(indices)) != num_views:
            raise ViewManifestError(
                f"{prefix}.frame_indices must be unique"
            )

        ids = entry.get("frame_ids")
        if (
            not isinstance(ids, list)
            or len(ids) != num_views
            or any(not isinstance(value, str) for value in ids)
        ):
            raise ViewManifestError(
                f"{prefix}.frame_ids must contain {num_views} strings"
            )

    return manifest


def load_view_manifest(
    path: str | Path,
    *,
    expected_num_views: int | None = None,
) -> dict[str, Any]:
    """Load and validate a frozen view manifest from disk."""

    manifest_path = _require_local_path(path, kind="view manifest")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"view manifest does not exist: {manifest_path}")
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ViewManifestError(
            f"invalid view manifest JSON {manifest_path}: {exc}"
        ) from exc
    return validate_view_manifest(
        manifest, expected_num_views=expected_num_views
    )


def write_view_manifest(
    manifest: dict[str, Any],
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Validate and atomically write a view manifest."""

    validate_view_manifest(manifest)
    path = _require_local_path(output_path, kind="view manifest output")
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite existing view manifest: {path}; "
            "pass overwrite=True/--force to replace it"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary_path.replace(path)
    return path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze deterministic View 0..23 selections for NVS eval."
    )
    parser.add_argument(
        "--dataset-manifest",
        required=True,
        help="Local newline-delimited camera JSON manifest.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset label recorded in the output (for example gso or dl3dv).",
    )
    parser.add_argument("--output", required=True, help="Output JSON path.")
    parser.add_argument("--num-views", type=int, default=DEFAULT_NUM_VIEWS)
    parser.add_argument("--seed", type=int, default=DEFAULT_VIEW_SEED)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing output manifest.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    manifest = build_view_manifest(
        args.dataset_manifest,
        dataset=args.dataset,
        num_views=args.num_views,
        seed=args.seed,
    )
    output_path = write_view_manifest(
        manifest, args.output, overwrite=args.force
    )
    print(
        f"Wrote {len(manifest['scenes'])} scenes x "
        f"{manifest['num_views']} views to {output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
