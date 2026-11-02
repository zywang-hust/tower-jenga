#!/usr/bin/env python3
"""Fetch and inspectably install the official starter; never execute its code.

jenga.server imports its wire codec (external/tower-starter/tower_protocol.py), so serving needs it. The archive is
pinned by SHA-256 (the version the final policy was served with); --archive installs a local copy of it instead
of downloading, e.g. if the challenge site is gone."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
from urllib.request import urlopen
import zipfile

URL = "https://challenge.tower-benchmark.com/docs/downloads/tower-policy-example.zip"
MANIFEST = ".starter-source.json"
MAX_BYTES = 16 * 1024 * 1024
DEFAULT_DEST = Path(__file__).resolve().parents[1] / "external" / "tower-starter"
ARCHIVE_SHA256 = "b0a4cfdd952064fa3fd3fa10c35bf4b119795b5a976dbeab77e8984f9f9f5ede"  # fetched 2026-10-22


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def unpack_checked(data: bytes) -> dict[str, bytes]:
    """Validate the complete archive before any filesystem mutation."""
    if len(data) > MAX_BYTES:
        raise ValueError("Archive exceeds download size limit")
    files: dict[str, bytes] = {}
    seen: set[str] = set()
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        if len(archive.infolist()) > 1000:
            raise ValueError("Too many archive entries")
        total = 0
        for item in archive.infolist():
            name = item.filename
            path = PurePosixPath(name)
            mode = item.external_attr >> 16
            if (
                not name
                or "\\" in name
                or ":" in name
                or path.is_absolute()
                or any(p in ("", ".", "..") for p in name.rstrip("/").split("/"))
                or path.as_posix() == MANIFEST
            ):
                raise ValueError(f"Unsafe archive path: {name!r}")
            if stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR):
                raise ValueError(f"Non-regular archive member: {name!r}")
            normalized = path.as_posix()
            if normalized in seen:
                raise ValueError(f"Duplicate archive path: {name!r}")
            seen.add(normalized)
            total += item.file_size
            if total > MAX_BYTES:
                raise ValueError("Expanded archive exceeds size limit")
            if not item.is_dir():
                files[normalized] = archive.read(item)
    for name in files:
        if any(parent.as_posix() in files for parent in PurePosixPath(name).parents):
            raise ValueError(f"Conflicting archive paths: {name!r}")
    if not {"server.py", "tower_protocol.py", "requirements.txt"}.issubset(files):
        raise ValueError("Archive does not contain the expected starter files")
    return files


def install_archive(data: bytes, destination: Path) -> dict:
    files = unpack_checked(data)
    record = {
        "source_url": URL,
        "archive_sha256": digest(data),
        "files": {name: digest(value) for name, value in sorted(files.items())},
    }
    files[MANIFEST] = (json.dumps(record, indent=2, sort_keys=True) + "\n").encode()
    destination = destination.absolute()
    for part in (destination, *destination.parents):
        if part.is_symlink():
            raise ValueError(f"Refusing symlink destination: {part}")
    if destination.exists():
        if not destination.is_dir():
            raise ValueError("Destination is not a directory")
        for name, value in files.items():
            target = destination / name
            for part in (target, *target.parents):
                if part == destination:
                    break
                if part.is_symlink():
                    raise ValueError(f"Refusing existing symlink: {part}")
            if not target.is_file() or target.read_bytes() != value:
                raise ValueError(f"Existing destination differs at {name}; use a new --dest")
        return record  # Untracked files and all existing bytes are preserved.
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".tower-starter-", dir=destination.parent))
    try:
        for name, value in files.items():
            target = staging / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(value)
        # Reserve the destination without replacing any concurrently created directory. A failed move leaves the
        # partial installation for inspection; user files are never removed.
        destination.mkdir()
        for child in staging.iterdir():
            child.rename(destination / child.name)
    finally:
        shutil.rmtree(staging)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DEST,
        help="Extraction directory (default: external/tower-starter)",
    )
    parser.add_argument("--archive", type=Path, help="install this local copy of the archive instead of downloading")
    parser.add_argument("--any-version", action="store_true", help="accept an archive that differs from the pinned one")
    args = parser.parse_args()
    if args.archive is not None:
        data = args.archive.read_bytes()
    else:
        with urlopen(URL, timeout=30) as response:
            if not response.geturl().startswith("https://challenge.tower-benchmark.com/"):
                raise ValueError("Unexpected download redirect")
            data = response.read(MAX_BYTES + 1)
    if digest(data) != ARCHIVE_SHA256 and not args.any_version:
        raise ValueError(f"Archive sha256 {digest(data)} differs from the pinned {ARCHIVE_SHA256} (--any-version)")
    record = install_archive(data, args.dest)
    print(json.dumps({"destination": str(args.dest.absolute()), **record}, indent=2))


if __name__ == "__main__":
    main()
