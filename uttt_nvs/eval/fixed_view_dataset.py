"""Deterministic local dataset for the 24-view NVS evaluation protocol."""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping

from uttt_nvs.eval.build_view_manifest import (
    DEFAULT_NUM_VIEWS,
    ViewManifestError,
    frame_id,
    load_view_manifest,
    read_local_dataset_manifest,
    scene_id,
)


MODEL_BATCH_KEYS = ("fxfycxcy", "c2w", "image", "index")


def as_model_batch(sample_or_batch: Mapping[str, Any]) -> dict[str, Any]:
    """Strip evaluation metadata before calling an existing LVSM model.

    Current LVSM model implementations slice every value in ``data_batch``.
    Keeping this helper at the dataset boundary prevents string metadata such
    as ``scene_id`` and ``frame_ids`` from entering ``model.forward``.
    """

    missing = [key for key in MODEL_BATCH_KEYS if key not in sample_or_batch]
    if missing:
        raise KeyError(f"sample is missing model batch keys: {missing}")
    return {key: sample_or_batch[key] for key in MODEL_BATCH_KEYS}


def _mapping_get(container: Any, key: str, default: Any = None) -> Any:
    if container is None:
        return default
    if isinstance(container, Mapping):
        return container.get(key, default)
    getter = getattr(container, "get", None)
    if getter is not None:
        return getter(key, default)
    return getattr(container, key, default)


class FixedViewNVSDataset:
    """Load exactly the frozen View 0..23 selection for every local scene.

    The standard tensor keys have the same shapes and semantics as
    ``uttt_nvs.data.loader.NVSDataset``:

    * ``fxfycxcy``: ``[24, 4]``
    * ``c2w``: ``[24, 4, 4]``
    * ``image``: ``[24, 3, H, W]``
    * ``index``: ``[24, 2]`` (source frame index, dataset scene index)

    View 0 is the input and View 1..23 are metric-bearing targets.  Evaluation
    metadata is returned alongside the four model keys.  Pass the result
    through :func:`as_model_batch` before invoking an existing LVSM model.

    Unlike ``NVSDataset``, a read/parse/decode error is fatal for that sample.
    It is never hidden by recursively substituting a random scene.
    """

    def __init__(
        self,
        config: Any = None,
        *,
        dataset_manifest: str | Path | None = None,
        view_manifest: str | Path | None = None,
        image_size: int | tuple[int, int] | None = None,
        num_views: int = DEFAULT_NUM_VIEWS,
        max_workers: int = 8,
        verify_camera_sha256: bool = True,
    ) -> None:
        if isinstance(config, (str, os.PathLike)):
            if dataset_manifest is not None:
                raise TypeError(
                    "dataset manifest was supplied both positionally and by keyword"
                )
            dataset_manifest = config
            config = None

        training = _mapping_get(config, "training")
        model = _mapping_get(config, "model")
        if dataset_manifest is None:
            dataset_manifest = _mapping_get(training, "dataset_path")
        if view_manifest is None:
            view_manifest = _mapping_get(training, "view_manifest")
        if view_manifest is None:
            view_manifest = _mapping_get(config, "view_manifest")
        if image_size is None:
            image_size = _mapping_get(model, "image_size", 256)

        if dataset_manifest is None:
            raise TypeError("dataset_manifest (or config.training.dataset_path) is required")
        if view_manifest is None:
            raise TypeError("view_manifest (or config.training.view_manifest) is required")
        if num_views != DEFAULT_NUM_VIEWS:
            raise ValueError(
                f"the evaluation protocol requires {DEFAULT_NUM_VIEWS} views, "
                f"got {num_views}"
            )
        if isinstance(image_size, int):
            self.image_size = (image_size, image_size)
        else:
            if (
                not isinstance(image_size, (tuple, list))
                or len(image_size) != 2
                or any(
                    not isinstance(value, int) or value <= 0
                    for value in image_size
                )
            ):
                raise ValueError(
                    "image_size must be a positive integer or (height, width)"
                )
            self.image_size = (int(image_size[0]), int(image_size[1]))
        if max_workers <= 0:
            raise ValueError(f"max_workers must be positive, got {max_workers}")

        self.dataset_manifest_path = Path(dataset_manifest).expanduser().resolve()
        self.view_manifest_path = Path(view_manifest).expanduser().resolve()
        self.camera_paths = read_local_dataset_manifest(self.dataset_manifest_path)
        self.view_manifest = load_view_manifest(
            self.view_manifest_path, expected_num_views=num_views
        )
        self.scenes = self.view_manifest["scenes"]
        if len(self.camera_paths) != len(self.scenes):
            raise ViewManifestError(
                f"dataset manifest has {len(self.camera_paths)} scenes, but frozen "
                f"view manifest has {len(self.scenes)} scenes"
            )

        self.num_views = num_views
        self.max_workers = max_workers
        self.verify_camera_sha256 = verify_camera_sha256
        self.dataset = str(self.view_manifest.get("dataset", ""))

    def __len__(self) -> int:
        return len(self.scenes)

    def metadata(self, index: int) -> dict[str, Any]:
        """Return frozen metadata without reading images."""

        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        entry = self.scenes[index]
        return {
            "dataset": self.dataset,
            "scene_index": index,
            "scene_id": entry["scene_id"],
            "camera_path": str(self.camera_paths[index]),
            "frame_indices": list(entry["frame_indices"]),
            "frame_ids": list(entry["frame_ids"]),
        }

    def _read_and_validate_camera(
        self, index: int
    ) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
        entry = self.scenes[index]
        camera_path = self.camera_paths[index]
        if not camera_path.is_file():
            raise FileNotFoundError(
                f"scene {entry['scene_id']!r}: camera JSON does not exist: "
                f"{camera_path}"
            )

        raw = camera_path.read_bytes()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ViewManifestError(
                f"scene {entry['scene_id']!r}: invalid camera JSON "
                f"{camera_path}: {exc}"
            ) from exc
        if not isinstance(payload, dict) or not isinstance(
            payload.get("frames"), list
        ):
            raise ViewManifestError(
                f"scene {entry['scene_id']!r}: {camera_path} must contain a "
                "list-valued 'frames'"
            )

        actual_scene_id = scene_id(payload, camera_path)
        if actual_scene_id != entry["scene_id"]:
            raise ViewManifestError(
                f"scene index {index}: frozen scene_id is {entry['scene_id']!r}, "
                f"but local camera JSON resolves to {actual_scene_id!r}"
            )
        if self.verify_camera_sha256 and entry.get("camera_sha256"):
            actual_sha256 = hashlib.sha256(raw).hexdigest()
            if actual_sha256 != entry["camera_sha256"]:
                raise ViewManifestError(
                    f"scene {entry['scene_id']!r}: camera JSON changed after view "
                    f"selection (expected sha256 {entry['camera_sha256']}, got "
                    f"{actual_sha256})"
                )

        frames = payload["frames"]
        selected_indices = entry["frame_indices"]
        largest_frame_index = max(selected_indices)
        if largest_frame_index >= len(frames):
            raise ViewManifestError(
                f"scene {entry['scene_id']!r}: frozen frame index "
                f"{largest_frame_index} is outside local frame count {len(frames)}"
            )
        actual_frame_ids = [
            frame_id(frames[frame_index], frame_index)
            for frame_index in selected_indices
        ]
        if actual_frame_ids != entry["frame_ids"]:
            raise ViewManifestError(
                f"scene {entry['scene_id']!r}: local frame IDs do not match the "
                "frozen view manifest"
            )
        return camera_path, payload, frames

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)

        entry = self.scenes[index]
        try:
            camera_path, _payload, frames = self._read_and_validate_camera(index)

            # Import project/runtime dependencies lazily.  This leaves the
            # manifest builder usable on login nodes without the CUDA env.
            import torch
            import torchvision.transforms as transforms
            from PIL import Image

            from uttt_nvs.data.loader import resize_and_crop

            data_point_base_dir = camera_path.parent
            selected_indices = entry["frame_indices"]
            to_tensor = transforms.ToTensor()

            def load_single_view(frame_index: int) -> tuple[Any, list[float], Any]:
                info = frames[frame_index]
                required = ("fx", "fy", "cx", "cy", "w2c", "file_path")
                missing = [key for key in required if key not in info]
                if missing:
                    raise KeyError(
                        f"scene {entry['scene_id']!r}, frame {frame_index}: "
                        f"missing camera fields {missing}"
                    )

                intrinsics = [
                    float(info["fx"]),
                    float(info["fy"]),
                    float(info["cx"]),
                    float(info["cy"]),
                ]
                w2c = torch.tensor(info["w2c"], dtype=torch.float32)
                if w2c.shape != (4, 4):
                    raise ValueError(
                        f"scene {entry['scene_id']!r}, frame {frame_index}: "
                        f"w2c must be 4x4, got {tuple(w2c.shape)}"
                    )
                c2w = torch.linalg.inv(w2c)

                raw_image_path = str(info["file_path"])
                if "://" in raw_image_path:
                    raise ViewManifestError(
                        f"scene {entry['scene_id']!r}, frame {frame_index}: "
                        f"evaluation requires a local image, got {raw_image_path!r}"
                    )
                image_path = Path(raw_image_path).expanduser()
                if not image_path.is_absolute():
                    image_path = data_point_base_dir / image_path
                image_path = image_path.resolve()
                if not image_path.is_file():
                    raise FileNotFoundError(
                        f"scene {entry['scene_id']!r}, frame {frame_index}: "
                        f"image does not exist: {image_path}"
                    )

                with image_path.open("rb") as handle:
                    with Image.open(handle) as image:
                        image.load()
                        image, intrinsics = resize_and_crop(
                            image, self.image_size, intrinsics
                        )
                        if image.mode == "RGBA":
                            rgb_image = Image.new(
                                "RGB", image.size, (255, 255, 255)
                            )
                            rgb_image.paste(image, mask=image.split()[-1])
                            image = rgb_image
                        elif image.mode != "RGB":
                            image = image.convert("RGB")
                        image_tensor = to_tensor(image)
                return c2w, intrinsics, image_tensor

            with ThreadPoolExecutor(
                max_workers=min(self.max_workers, self.num_views)
            ) as executor:
                results = list(
                    executor.map(load_single_view, selected_indices)
                )
        except Exception as exc:
            raise RuntimeError(
                f"failed to load fixed NVS scene index {index} "
                f"({entry['scene_id']!r}); no fallback scene was substituted: "
                f"{exc}"
            ) from exc

        c2w_list, fxfycxcy_list, image_list = zip(*results)
        frame_indices = torch.tensor(selected_indices, dtype=torch.long)
        scene_indices = torch.full_like(frame_indices, index)
        data_indices = torch.stack((frame_indices, scene_indices), dim=-1)

        return {
            "fxfycxcy": torch.tensor(fxfycxcy_list, dtype=torch.float32),
            "c2w": torch.stack(c2w_list),
            "image": torch.stack(image_list),
            "index": data_indices,
            "dataset": self.dataset,
            "scene_id": entry["scene_id"],
            "frame_ids": tuple(entry["frame_ids"]),
            "frame_indices": frame_indices,
            "camera_path": str(camera_path),
        }


FixedViewDataset = FixedViewNVSDataset
