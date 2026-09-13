import json
from pathlib import Path
import subprocess
import sys

import pytest
import numpy as np

from tools.check_manifest import validate_manifest
from uttt_nvs.data.build_manifest import scan_local
from uttt_nvs.data.make_toy_dataset import build, look_at


def test_portable_toy_manifest_and_reproducible_images(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    manifest = Path(build("relative", scenes=2, views=24, size=8, seed=9595, repeat=2))
    assert manifest.is_absolute()
    report = validate_manifest(manifest, check_images=True)
    assert report == {"entries": 4, "unique_scenes": 2, "duplicate_entries": 2,
                      "images_checked": 48, "decoded_images": True, "errors": []}
    build("second", scenes=2, views=24, size=8, seed=9595, repeat=1)
    assert (tmp_path / "relative/scene_000/images/000.png").read_bytes() == (tmp_path / "second/scene_000/images/000.png").read_bytes()


def test_toy_camera_is_right_handed_opencv():
    eye = np.array([0.0, 0.0, 2.5])
    transform = look_at(eye, np.zeros(3), np.array([0.0, 1.0, 0.0]))
    rotation = transform[:3, :3]
    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3))
    assert np.linalg.det(rotation) == pytest.approx(1.0)
    np.testing.assert_allclose(transform @ np.append(eye, 1.0), [0, 0, 0, 1])
    assert (transform @ np.array([0, 0, 0, 1]))[2] > 0


@pytest.mark.parametrize("field", ["scenes", "views", "size", "repeat"])
def test_toy_rejects_empty_dimensions(tmp_path, field):
    arguments = dict(scenes=1, views=2, size=8, seed=42, repeat=1)
    arguments[field] = 0
    with pytest.raises(ValueError, match=field):
        build(str(tmp_path / "absent"), **arguments)
    assert not (tmp_path / "absent").exists()


def test_default_toy_has_enough_eval_views(tmp_path):
    result = subprocess.run([sys.executable, "-m", "uttt_nvs.data.make_toy_dataset",
                             "--out", str(tmp_path), "--scenes", "1", "--size", "8", "--repeat", "1"],
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert not validate_manifest(tmp_path / "manifest.txt", min_views=24)["errors"]


def test_scan_local_returns_absolute_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "scene").mkdir()
    (tmp_path / "scene/opencv_cameras.json").write_text("{}")
    assert scan_local("scene", "*.json") == [str(tmp_path / "scene/opencv_cameras.json")]
    with pytest.raises(FileNotFoundError):
        scan_local("missing", "*.json")


@pytest.mark.parametrize("failure", ["too_few_views", "missing_image", "singular_camera", "nonfinite_intrinsics"])
def test_manifest_rejects_bad_scene(tmp_path, failure):
    manifest = Path(build(str(tmp_path), 1, 24, 8, 42, 1))
    camera = tmp_path / "scene_000/opencv_cameras.json"
    data = json.loads(camera.read_text())
    if failure == "too_few_views":
        data["frames"] = data["frames"][:2]
    elif failure == "missing_image":
        data["frames"][0]["file_path"] = "missing.png"
    elif failure == "singular_camera":
        data["frames"][0]["w2c"] = [[0] * 4 for index in range(4)]
    else:
        data["frames"][0]["fx"] = float("nan")
    camera.write_text(json.dumps(data))
    assert validate_manifest(manifest)["errors"]


def test_empty_and_cloud_manifests_are_not_reported_as_valid(tmp_path):
    manifest = tmp_path / "manifest.txt"
    manifest.write_text("\n")
    with pytest.raises(ValueError, match="empty"):
        validate_manifest(manifest)
    manifest.write_text("s3://example/camera.json\n")
    assert "cannot validate cloud" in validate_manifest(manifest)["errors"][0]


def test_negative_manifest_limit_does_not_write_output(tmp_path):
    output = tmp_path / "manifest.txt"
    result = subprocess.run([sys.executable, "-m", "uttt_nvs.data.build_manifest",
                             "--root", str(tmp_path), "--out", str(output), "--limit", "-1"],
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 2
    assert "non-negative" in result.stderr
    assert not output.exists()


def test_image_content_validation_is_explicit(tmp_path):
    manifest = Path(build(str(tmp_path), 1, 24, 8, 42, 1))
    (tmp_path / "scene_000/images/000.png").write_bytes(b"not an image")
    assert not validate_manifest(manifest, check_images=False)["errors"]
    assert validate_manifest(manifest, check_images=True)["errors"]
