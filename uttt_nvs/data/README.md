# Data

This repository ships **code, not data**. You describe your own copy of the data
in `uttt_nvs/datasets.yaml`, once:

```bash
cp uttt_nvs/datasets.example.yaml uttt_nvs/datasets.yaml
```

```yaml
obj:
  train: /path/to/objaverse_train_manifest.txt
  eval:  /path/to/gso_test_manifest.txt
dl3dv:
  train: /path/to/dl3dv_train_manifest.txt
  eval:  /path/to/dl3dv_benchmark_manifest.txt
```

Every config carries `training.dataset: obj` or `dl3dv`, which selects the block
above — so this one file wires up all 64 configs. A config can still override it
by uncommenting its own `dataset_path` / `eval_dataset_path`.

## Manifests

The loader never takes a dataset directory. It takes a **manifest**: a plain
text file with one camera-JSON path per line.

```
/data/objaverse/000-001/abc123/opencv_cameras.json
/data/objaverse/000-001/def456/opencv_cameras.json
...
```

Each line is opened with `open_file()`, which dispatches on the prefix:

| Prefix | Backend | Credentials |
|---|---|---|
| `/…` or `./…` | local filesystem | none |
| `s3://…` | Amazon S3 (unsigned/anonymous) | none for public buckets |
| `gs://…` | Google Cloud Storage | `GOOGLE_APPLICATION_CREDENTIALS` |

Use a backend-matched manifest: local scene paths for a local manifest, S3
paths for S3, and GCS paths for GCS. Client initialization in the current loader
depends on the manifest's backend; arbitrary mixed-backend manifests are not
validated. See the [data contracts guide](../../docs/data-contracts.md) for
portable paths, geometry conventions and validation boundaries.

## Camera JSON format

Each entry in a manifest is a JSON file shaped like:

```json
{
  "frames": [
    {
      "fx": 512.0, "fy": 512.0, "cx": 256.0, "cy": 256.0,
      "w2c": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 2.5], [0, 0, 0, 1]],
      "file_path": "images/000.png"
    }
  ]
}
```

`file_path` is resolved **relative to the directory containing the JSON**, so a
scene directory is self-contained and can be relocated freely.

The example shows one frame only; real scenes need at least the requested
view count (12 for standard training, 24 for in-training/full-test evaluation).
Intrinsics are in pixels and `w2c` is world-to-camera in OpenCV convention.

## Validate before training

```bash
python -m tools.check_manifest /data/manifests/train.txt --min-views 12 --check-images
python -m tools.check_manifest /data/manifests/test.txt --min-views 24 --check-images
```

The CPU checker validates local scenes, view counts, calibration matrices and
image paths; the optional flag also verifies image files. It reports cloud
entries as unvalidated rather than silently passing them. Duplicate manifest
entries are counted explicitly.

For a dataset-free plumbing check, generate 16 synthetic scenes with 24 views:

```bash
python -m uttt_nvs.data.make_toy_dataset --out /tmp/uttt_toy
python -m tools.check_manifest /tmp/uttt_toy/manifest.txt --check-images
```

The synthetic manifest repeats scenes to provide complete batches. It is not
appropriate for reporting quality or train/test generalization.

## Building a manifest

```bash
# local tree
python -m uttt_nvs.data.build_manifest \
    --root /path/to/objaverse \
    --out manifests/objaverse_train.txt

# S3 prefix (anonymous)
python -m uttt_nvs.data.build_manifest \
    --root s3://your-bucket/your/prefix \
    --out manifests/train.txt

# GCS prefix
python -m uttt_nvs.data.build_manifest \
    --root gs://your-bucket/your/prefix \
    --out manifests/train.txt
```

Use `--pattern` if your camera files are not named `opencv_cameras*.json`, and
`--limit` to build a small manifest for a quick trial run.

## Then wire it into a config

```yaml
training:
  dataset_path: manifests/objaverse_train.txt
eval_dataset_path: manifests/gso_test.txt
```

## Obtaining the datasets

The paper uses three datasets. Get them from their official sources and respect
their licences — this repository redistributes none of them and endorses no
particular mirror.

| Dataset | Used for | Source |
|---|---|---|
| Objaverse | object-level training | https://objaverse.allenai.org/ |
| Google Scanned Objects | object-level evaluation | https://research.google/resources/datasets/scanned-objects/ |
| DL3DV-10K | scene-level training and evaluation | https://dl3dv-10k.github.io/DL3DV-10K/ (gated; request access) |

Renderings must be produced in the camera-JSON layout above. Source images may
be any size: the loader resizes each one to cover `model.image_size` and then
centre-crops it to a square, adjusting the camera intrinsics to match. The
shipped configs train at 256x256 with 12 input and 24 evaluation views, on
Objaverse for the object models and on the DL3DV 11K-scene training split with
140 held-out test scenes for the scene models. The larger-model configs step
through a resolution ladder instead (128 to 512 on DL3DV, 256 to 1024 on
Objaverse).
