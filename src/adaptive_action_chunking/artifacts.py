"""Locate repository artifacts and verify their recorded SHA256 hashes."""

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def resolve_artifact(path, expected_sha256=None):
    """Return the artifact path (relative paths are repository-root relative), checking its hash if given."""
    path = Path(path)
    absolute = path if path.is_absolute() else ROOT / path
    if absolute.exists() and (expected_sha256 is None or
                              hashlib.sha256(absolute.read_bytes()).hexdigest() == expected_sha256):
        return absolute
    raise FileNotFoundError(f"Artifact unavailable or checksum mismatch: {path}")


def digest(path, expected_sha256=None):
    return hashlib.sha256(resolve_artifact(path, expected_sha256).read_bytes()).hexdigest()
