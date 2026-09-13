"""Check a local NVS camera manifest before starting distributed training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def validate_manifest(path: Path, min_views: int = 24, check_images: bool = False) -> dict:
    if min_views < 1:
        raise ValueError("min_views must be positive")
    entries = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not entries:
        raise ValueError("Manifest is empty")
    errors, image_count = [], 0
    for camera_path in dict.fromkeys(entries):
        try:
            if camera_path.startswith(("s3://", "gs://")):
                raise ValueError("local checker cannot validate cloud objects; stage this scene locally first")
            camera = Path(camera_path)
            frames = json.loads(camera.read_text(encoding="utf-8"))["frames"]
            if not isinstance(frames, list) or len(frames) < min_views:
                raise ValueError(f"frames must be a list with at least {min_views} views")
            for index, frame in enumerate(frames):
                intrinsics = np.asarray([frame[key] for key in ("fx", "fy", "cx", "cy")], dtype=float)
                if not np.isfinite(intrinsics).all() or (intrinsics[:2] <= 0).any():
                    raise ValueError(f"frame {index}: intrinsics must be finite with positive fx/fy")
                matrix = np.asarray(frame["w2c"], dtype=float)
                if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
                    raise ValueError(f"frame {index}: w2c must be finite and 4x4")
                if not np.allclose(matrix[3], [0, 0, 0, 1]) or abs(np.linalg.det(matrix)) < 1e-12:
                    raise ValueError(f"frame {index}: w2c must be an invertible homogeneous transform")
                image_path = camera.parent / frame["file_path"]
                if not image_path.is_file():
                    raise ValueError(f"frame {index}: image not found: {image_path}")
                if check_images:
                    with Image.open(image_path) as image:
                        image.verify()
                    with Image.open(image_path) as image:
                        image.load()
                image_count += 1
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append(f"{camera_path}: {exc}")
    return {"entries": len(entries), "unique_scenes": len(set(entries)),
            "duplicate_entries": len(entries) - len(set(entries)),
            "images_checked": image_count, "decoded_images": check_images,
            "errors": errors}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--min-views", type=int, default=24)
    parser.add_argument("--check-images", action="store_true")
    args = parser.parse_args()
    try:
        report = validate_manifest(args.manifest, args.min_views, args.check_images)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2))
    return int(bool(report["errors"]))


if __name__ == "__main__":
    raise SystemExit(main())
