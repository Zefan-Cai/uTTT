"""Build a tiny synthetic scene set, so the training loop can be run before any
real data is in place.

The scenes are procedural: a coloured backdrop with a few shapes, photographed
from cameras on a circular orbit. They carry no useful 3D signal, and a model
trained on them learns nothing worth keeping. What they do give you is a
complete, valid dataset in the layout of :mod:`uttt_nvs.data.loader` -- camera
JSONs, images, and a manifest -- so a first run exercises the loader, the
model, the optimizer and checkpointing end to end.

    python -m uttt_nvs.data.make_toy_dataset --out /tmp/uttt_toy

Then point `uttt_nvs/datasets.yaml` at the manifest it prints.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
from PIL import Image


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """World-to-camera matrix for a camera at `eye` looking at `target`."""
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    camera_down = np.cross(forward, right)

    w2c = np.eye(4)
    w2c[:3, :3] = np.stack([right, camera_down, forward])
    w2c[:3, 3] = -w2c[:3, :3] @ eye
    return w2c


def render_view(rng: np.random.Generator, size: int, scene_hue: np.ndarray,
                angle: float) -> np.ndarray:
    """A cheap procedural image that varies smoothly with the view angle."""
    ys, xs = np.mgrid[0:size, 0:size] / size
    backdrop = (0.35 + 0.3 * np.sin(2 * np.pi * (xs + angle / (2 * np.pi))))[..., None] * scene_hue

    image = backdrop.copy()
    for cx, cy, radius, colour in rng.random((3, 4)) @ np.diag([1, 1, 0.18, 1]):
        blob = ((xs - cx) ** 2 + (ys - cy) ** 2) < radius ** 2
        image[blob] = np.clip(scene_hue * (0.5 + colour), 0, 1)

    return (np.clip(image, 0, 1) * 255).astype(np.uint8)


def build(out_dir: str, scenes: int, views: int, size: int, seed: int,
          repeat: int) -> str:
    for name, value in (("scenes", scenes), ("views", views), ("size", size),
                        ("repeat", repeat)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    out_dir = os.path.abspath(os.path.expanduser(out_dir))
    rng = np.random.default_rng(seed)
    focal = 0.7 * size
    manifest_lines = []

    for scene in range(scenes):
        scene_dir = os.path.join(out_dir, f"scene_{scene:03d}")
        os.makedirs(os.path.join(scene_dir, "images"), exist_ok=True)
        scene_hue = 0.4 + 0.6 * rng.random(3)

        frames = []
        for view in range(views):
            angle = 2 * np.pi * view / views
            eye = np.array([2.5 * np.cos(angle), 0.6, 2.5 * np.sin(angle)])
            w2c = look_at(eye, np.zeros(3), np.array([0.0, 1.0, 0.0]))

            image = render_view(rng, size, scene_hue, angle)
            Image.fromarray(image).save(os.path.join(scene_dir, "images", f"{view:03d}.png"))

            frames.append({
                "fx": focal, "fy": focal, "cx": size / 2, "cy": size / 2,
                "w2c": w2c.tolist(),
                "file_path": f"images/{view:03d}.png",
            })

        camera_json = os.path.join(scene_dir, "opencv_cameras.json")
        with open(camera_json, "w", encoding="utf-8") as handle:
            json.dump({"frames": frames}, handle)
        manifest_lines.append(camera_json)

    # One line per sample, not per scene: a global batch of 128 has to be drawn
    # from somewhere, and rendering 128 distinct scenes is slower than reusing a
    # few. Repetition is fine here -- these scenes are a pipeline check, not data.
    manifest = os.path.join(out_dir, "manifest.txt")
    with open(manifest, "w", encoding="utf-8") as handle:
        handle.write("\n".join(manifest_lines * repeat) + "\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, help="Directory to build the scenes in")
    parser.add_argument("--scenes", type=int, default=16, help="Number of scenes")
    parser.add_argument("--views", type=int, default=24,
                        help="Views per scene; must cover both num_views and eval_num_views")
    parser.add_argument("--size", type=int, default=256,
                        help="Image resolution, matching model.image_size")
    parser.add_argument("--repeat", type=int, default=16,
                        help="Times each scene is listed in the manifest, so that a "
                             "full global batch can be drawn (default: 16)")
    parser.add_argument("--seed", type=int, default=9595)
    args = parser.parse_args()

    try:
        manifest = build(args.out, args.scenes, args.views, args.size, args.seed,
                         args.repeat)
    except ValueError as exc:
        parser.error(str(exc))
    print(f"Wrote {args.scenes} scenes to {args.out}, "
          f"listed {args.repeat}x each ({args.scenes * args.repeat} manifest entries)")
    print("\nPoint uttt_nvs/datasets.yaml at it:\n")
    print("obj:")
    print(f"  train: {manifest}")
    print(f"  eval:  {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
