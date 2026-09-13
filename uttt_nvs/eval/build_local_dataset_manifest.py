#!/usr/bin/env python3
"""Translate an object-store NVS manifest into verified local camera paths."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def translate_manifest(
    source_manifest: str | Path,
    *,
    source_prefix: str,
    local_root: str | Path,
    verify_exists: bool = True,
) -> list[Path]:
    source_path = Path(source_manifest).expanduser().resolve()
    root = Path(local_root).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"source manifest does not exist: {source_path}")
    if not source_prefix:
        raise ValueError("source_prefix must be non-empty")

    translated: list[Path] = []
    for line_number, line in enumerate(
        source_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        value = line.strip()
        if not value:
            continue
        if not value.startswith(source_prefix):
            raise ValueError(
                f"{source_path}:{line_number}: path does not start with "
                f"{source_prefix!r}: {value!r}"
            )
        relative = value[len(source_prefix) :].lstrip("/")
        if not relative:
            raise ValueError(
                f"{source_path}:{line_number}: source path has no relative suffix"
            )
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"{source_path}:{line_number}: translated path escapes "
                f"local root {root}: {candidate}"
            ) from exc
        if verify_exists and not candidate.is_file():
            raise FileNotFoundError(
                f"{source_path}:{line_number}: local camera JSON is missing: "
                f"{candidate}"
            )
        translated.append(candidate)

    if not translated:
        raise ValueError(f"source manifest has no non-empty entries: {source_path}")
    if len(translated) != len(set(translated)):
        raise ValueError("translated manifest contains duplicate camera paths")
    return translated


def write_manifest(
    paths: list[Path],
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    output = Path(output_path).expanduser().resolve()
    if output.exists() and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite {output}; pass --force to replace it"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.write_text(
        "".join(f"{path}\n" for path in paths),
        encoding="utf-8",
    )
    temporary.replace(output)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-prefix", required=True)
    parser.add_argument("--local-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-existence-check", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = translate_manifest(
        args.source_manifest,
        source_prefix=args.source_prefix,
        local_root=args.local_root,
        verify_exists=not args.skip_existence_check,
    )
    output = write_manifest(paths, args.output, overwrite=args.force)
    print(f"Wrote {len(paths)} local camera paths to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
