"""Zstd compression for trajectory blobs stored in ``rollouts.extra_compressed``.

Mirrors the ``extra_compressed`` pattern from affine-cortex: the structured
trajectory JSON is compressed and stored in the row body. Postgres TOAST
handles out-of-page storage transparently.
"""

from __future__ import annotations

import hashlib

import orjson
import zstandard as zstd

_DEFAULT_LEVEL = 9


def compress_json(payload: object, *, level: int = _DEFAULT_LEVEL) -> tuple[bytes, str]:
    """Serialize and compress payload. Returns (compressed_bytes, sha256_hex_of_compressed)."""
    raw = orjson.dumps(payload)
    compressed = zstd.ZstdCompressor(level=level).compress(raw)
    digest = hashlib.sha256(compressed).hexdigest()
    return compressed, digest


def decompress_json(blob: bytes) -> object:
    raw = zstd.ZstdDecompressor().decompress(blob)
    return orjson.loads(raw)
