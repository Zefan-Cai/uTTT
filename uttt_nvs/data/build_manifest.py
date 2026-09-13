"""Build a dataset manifest for :class:`uttt_nvs.data.loader.NVSDataset`.

A manifest is a plain text file with one camera-JSON path per line. The loader
opens each line with :func:`uttt_nvs.data.loader.open_file`, which dispatches on
the path prefix, so a manifest may point at local files, ``s3://`` objects or
``gs://`` blobs. Use a backend-matched manifest; the current training loader's
client initialization is tied to the manifest backend.

This indirection is why the repository ships code but no data: point the
manifest wherever your copy of the dataset lives.

Examples
--------
Scan a local dataset tree::

    python -m uttt_nvs.data.build_manifest \
        --root /path/to/objaverse \
        --out manifests/objaverse_train.txt

Scan an S3 prefix (anonymous access, no credentials needed for public buckets)::

    python -m uttt_nvs.data.build_manifest \
        --root s3://your-bucket/your/prefix \
        --out manifests/train.txt

Scan a GCS prefix (requires GOOGLE_APPLICATION_CREDENTIALS)::

    python -m uttt_nvs.data.build_manifest \
        --root gs://your-bucket/your/prefix \
        --out manifests/train.txt

Each line the loader reads must be a JSON file with a ``frames`` list, where
every frame carries ``fx``, ``fy``, ``cx``, ``cy``, ``w2c`` and ``file_path``.
``file_path`` is resolved relative to the directory holding the JSON.
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import sys
from pathlib import Path


DEFAULT_PATTERN = "opencv_cameras*.json"


def scan_local(root: str, pattern: str) -> list[str]:
    root = os.path.abspath(os.path.expanduser(root))
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Dataset root is not a directory: {root}")
    matches: list[str] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            if fnmatch.fnmatch(name, pattern):
                matches.append(os.path.join(dirpath, name))
    return matches


def scan_s3(root: str, pattern: str) -> list[str]:
    try:
        import boto3
        from botocore import UNSIGNED
        from botocore.config import Config
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("boto3 is required to scan s3:// prefixes") from exc

    bucket, _, prefix = root[len("s3://"):].partition("/")
    client = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    paginator = client.get_paginator("list_objects_v2")
    matches: list[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if fnmatch.fnmatch(os.path.basename(key), pattern):
                matches.append(f"s3://{bucket}/{key}")
    return matches


def scan_gcs(root: str, pattern: str) -> list[str]:
    try:
        from google.cloud import storage
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("google-cloud-storage is required to scan gs:// prefixes") from exc

    bucket_name, _, prefix = root[len("gs://"):].partition("/")
    client = storage.Client()
    matches: list[str] = []
    for blob in client.list_blobs(bucket_name, prefix=prefix):
        if fnmatch.fnmatch(os.path.basename(blob.name), pattern):
            matches.append(f"gs://{bucket_name}/{blob.name}")
    return matches


def build(root: str, pattern: str) -> list[str]:
    if root.startswith("s3://"):
        return scan_s3(root, pattern)
    if root.startswith("gs://"):
        return scan_gcs(root, pattern)
    root = os.path.abspath(os.path.expanduser(root))
    if not os.path.isdir(root):
        raise SystemExit(f"Not a directory: {root}")
    return scan_local(root, pattern)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", required=True,
                        help="Directory, s3:// prefix or gs:// prefix to scan")
    parser.add_argument("--out", required=True, help="Manifest file to write")
    parser.add_argument("--pattern", default=DEFAULT_PATTERN,
                        help=f"Filename glob for camera JSONs (default: {DEFAULT_PATTERN})")
    parser.add_argument("--limit", type=int, default=0,
                        help="Keep at most this many entries (0 = no limit)")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be non-negative")

    print(f"Scanning {args.root} for {args.pattern} ...", file=sys.stderr)
    matches = sorted(build(args.root, args.pattern))
    if args.limit:
        matches = matches[: args.limit]

    if not matches:
        print(
            f"No files matching {args.pattern!r} under {args.root}.\n"
            f"Check the path, or pass a different --pattern.",
            file=sys.stderr,
        )
        return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(matches) + "\n", encoding="utf-8")
    print(f"Wrote {len(matches)} entries to {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
