"""Process-wide tokenizer cache, deduplicated by tokenizer file content.

Previously every evaluate call ran ``AutoTokenizer.from_pretrained`` anew;
a fast tokenizer for a 150k+ vocab holds ~100-250MB of RAM, so N in-flight
evaluates held N copies and the container's cgroup limit was routinely
blown. Miners are fine-tunes of the same base family and ship
byte-identical tokenizers, so keying a cache by repo would still keep one
copy per miner — instead we hash the repo's tokenizer files and share ONE
in-memory instance per distinct content. A repo that genuinely ships a
different tokenizer gets its own instance, keeping input_ids aligned with
the vocab its serving endpoint actually uses.
"""

from __future__ import annotations

import fnmatch
import hashlib
import os
import threading
from collections import OrderedDict

from distill_v2_eval.infrastructure.logging import get_logger

log = get_logger("forward.tokenizer_cache")

# Files that define tokenization behaviour — these (and only these) enter
# the content hash.
_HASH_PATTERNS = (
    "tokenizer*",
    "vocab*",
    "merges*",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template*",
    "*.model",
    "*.tiktoken",
    "*.py",  # trust_remote_code tokenizer implementations
)
# Needed on disk for AutoTokenizer to load, but excluded from the hash:
# config.json differs across fine-tunes even when tokenization is identical
# (AutoTokenizer only falls back to it when tokenizer_config.json lacks a
# tokenizer_class).
_LOAD_ONLY_PATTERNS = ("config.json",)

# Distinct tokenizer *contents* live at once. In practice there is exactly
# one (the shared base family); 2 covers a base-model transition. Kept low
# on purpose so adversarial miners shipping unique tokenizers can bloat the
# cache by at most one extra instance (eviction only costs a reload —
# in-flight evaluations keep their reference).
_MAX_ENTRIES = 2

_lock = threading.Lock()
_cache: OrderedDict[str, object] = OrderedDict()


def get_tokenizer(source: str):  # type: ignore[no-untyped-def]
    """Return a shared tokenizer for ``source`` (HF repo id or local dir).

    Falls back to an uncached ``from_pretrained`` (the old behaviour) if
    snapshotting/hashing fails, so cache problems can never fail an eval
    that would previously have succeeded.
    """
    try:
        local_dir = _materialize(source)
        key = _content_key(local_dir)
    except Exception as exc:  # noqa: BLE001
        log.warning("tokenizer_cache.fallback_uncached", source=source,
                    error=f"{type(exc).__name__}: {exc}")
        return _load(source)

    # The lock is held across from_pretrained on purpose: concurrent misses
    # for the same content must not instantiate duplicates — that is the
    # exact allocation storm this cache exists to prevent. Loading from the
    # local snapshot dir takes seconds and hits no network.
    with _lock:
        tok = _cache.get(key)
        if tok is None:
            tok = _load(local_dir)
            _cache[key] = tok
            while len(_cache) > _MAX_ENTRIES:
                evicted, _ = _cache.popitem(last=False)
                log.info("tokenizer_cache.evict", key=evicted[:16])
            log.info("tokenizer_cache.load", source=source, key=key[:16],
                     entries=len(_cache))
        else:
            _cache.move_to_end(key)
        return tok


def _materialize(source: str) -> str:
    """Resolve ``source`` to a local directory holding the tokenizer files."""
    if os.path.isdir(source):
        return source
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    from huggingface_hub import snapshot_download
    return snapshot_download(
        repo_id=source,
        allow_patterns=list(_HASH_PATTERNS + _LOAD_ONLY_PATTERNS),
    )


def _content_key(local_dir: str) -> str:
    """sha256 over (relpath, file sha256) of every hash-relevant file."""
    entries: list[tuple[str, str]] = []
    for root, _dirs, files in os.walk(local_dir, followlinks=True):
        for name in files:
            rel = os.path.relpath(os.path.join(root, name), local_dir)
            if not any(fnmatch.fnmatch(rel, p) for p in _HASH_PATTERNS):
                continue
            entries.append((rel, _file_sha256(os.path.join(root, name))))
    if not entries:
        raise ValueError(f"no tokenizer files found under {local_dir}")
    digest = hashlib.sha256()
    for rel, sha in sorted(entries):
        digest.update(f"{rel}:{sha}\n".encode())
    return digest.hexdigest()


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load(path_or_repo: str):  # type: ignore[no-untyped-def]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path_or_repo, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok
