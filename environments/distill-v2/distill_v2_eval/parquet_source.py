"""Fetch + decode a per-cell rollout shard from cortex's R2 layout.

Layout (matches cortex's ``publishing.publisher``):

    {base_url}/manifest.jsonl                # append-only index
    {base_url}/tasks/{task_idx:08d}.parquet  # immutable per-cell shard

A "cell" is (list_name, env_name, task_id, teacher_name) — one
sampling_progress row at ``collected >= target_samples``. All rollouts
in a shard share the same teacher, so within-cell advantage z-score
captures only reward signal, not style differences between teachers.

Manifest line schema:
    {"task_idx": int, "list_name": str, "env_name": str,
     "task_id": int, "teacher_name": str, "n_rollouts": int,
     "reward_mean": float, "reward_std": float,
     "object_uri": str, "object_key": str, "committed_at": iso}

Reproducibility: given the same ``base_url`` + ``task_idx``, this module
always returns the same set of rollouts. Producers commit each
task_idx exactly once; the corresponding parquet is never rewritten.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass

import httpx
import pyarrow.parquet as pq

from distill_v2_eval.domain.ids import (
    EnvName,
    RolloutId,
    TaskId,
    TeacherName,
)
from distill_v2_eval.domain.models import Rollout, RolloutStatus


@dataclass(frozen=True)
class FetchParams:
    base_url: str        # e.g. https://r2.cdn.example.com/distill-v2
    task_idx: int        # cortex-side task identifier
    timeout_s: float = 60.0


@dataclass(frozen=True)
class ManifestEntry:
    task_idx: int
    list_name: str
    env_name: str
    task_id: int
    teacher_name: str
    n_rollouts: int
    reward_mean: float
    reward_std: float
    object_uri: str
    object_key: str
    committed_at: str = ""    # ISO 8601 UTC
    mature_at: str = ""       # ISO 8601 UTC; entries before this are "fresh"


def _manifest_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/manifest.jsonl"


def _task_object_url(base_url: str, task_idx: int) -> str:
    return f"{base_url.rstrip('/')}/tasks/{task_idx:08d}.parquet"


def fetch_manifest(
    base_url: str, *,
    timeout_s: float = 60.0,
    include_immature: bool = True,
) -> list[ManifestEntry]:
    """Pull manifest.jsonl and parse every line.

    ``include_immature`` controls the maturation gate:
      * ``True`` (default) — return every entry. Used by the validator
        who needs to evaluate freshly-published cells right away.
      * ``False`` — drop entries whose ``mature_at`` is still in the
        future. Used by any downstream consumer (e.g. miner-facing
        public manifest) that must not see freshly-evaluated tasks.

    Entries without a ``mature_at`` field (legacy manifests from
    before the maturation hook) are treated as already mature.
    """
    from datetime import datetime, timezone
    payload = _http_get_bytes(_manifest_url(base_url), timeout_s)
    text = payload.decode("utf-8").strip()
    out: list[ManifestEntry] = []
    if not text:
        return out
    now = datetime.now(timezone.utc)
    for line in text.splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        entry = ManifestEntry(
            task_idx=int(obj["task_idx"]),
            list_name=str(obj["list_name"]),
            env_name=str(obj["env_name"]),
            task_id=int(obj["task_id"]),
            teacher_name=str(obj["teacher_name"]),
            n_rollouts=int(obj["n_rollouts"]),
            reward_mean=float(obj.get("reward_mean", 0.0)),
            reward_std=float(obj.get("reward_std", 0.0)),
            object_uri=str(obj["object_uri"]),
            object_key=str(obj["object_key"]),
            committed_at=str(obj.get("committed_at", "")),
            mature_at=str(obj.get("mature_at", "")),
        )
        if not include_immature and entry.mature_at:
            ma = _parse_iso(entry.mature_at)
            if ma is not None and ma > now:
                continue
        out.append(entry)
    return out


def _parse_iso(s: str):  # type: ignore[no-untyped-def]
    from datetime import datetime
    try:
        # Python 3.11+ accepts 'Z'; normalise for safety
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def fetch_task_rollouts(params: FetchParams) -> tuple[list[Rollout], ManifestEntry]:
    """Resolve task_idx -> object url via manifest, then GET the shard.

    Returns ``(rollouts, manifest_entry)``. The manifest_entry is
    surfaced back so the caller can log / persist the source
    coordinates alongside the score.
    """
    manifest = fetch_manifest(params.base_url, timeout_s=params.timeout_s)
    entry = _lookup(manifest, params.task_idx)
    object_url = _task_object_url(params.base_url, entry.task_idx)
    payload = _http_get_bytes(object_url, params.timeout_s)
    table = pq.read_table(io.BytesIO(payload))
    return _table_to_rollouts(table), entry


@dataclass(frozen=True)
class DatasetMetadata:
    """Thin pointer published alongside the manifest. Matches the
    SWE-Infinite ``metadata.json`` schema so a single client convention
    works across both data sources."""

    version: int
    last_updated: str
    staged_up_to: int      # total cells the publisher has written (private)
    completed_up_to: int   # total mature cells (public bucket only)


def fetch_metadata(base_url: str, *, timeout_s: float = 30.0) -> DatasetMetadata:
    """GET ``{base_url}/metadata.json`` and decode it.

    This is the cheap-discovery entry point — the file is < 1 KB so
    callers can poll it on every sample to enumerate the valid
    ``task_idx`` range without paying the manifest cost.
    """
    url = f"{base_url.rstrip('/')}/metadata.json"
    payload = _http_get_bytes(url, timeout_s)
    obj = json.loads(payload.decode("utf-8"))
    tasks = obj.get("tasks") or {}
    # ``staged_up_to`` is the source of truth on the PRIVATE bucket
    # (where the publisher writes); the PUBLIC bucket also has it
    # alongside ``completed_up_to``. Fall back to the legacy ``total``
    # key emitted by the publisher's bare metadata.
    staged = int(
        tasks.get("staged_up_to")
        if tasks.get("staged_up_to") is not None
        else tasks.get("total", 0)
    )
    completed = int(tasks.get("completed_up_to", staged))
    return DatasetMetadata(
        version=int(obj.get("version", 1)),
        last_updated=str(obj.get("last_updated", "")),
        staged_up_to=staged,
        completed_up_to=completed,
    )


def fetch_task_rollouts_direct(
    base_url: str, task_idx: int, *, timeout_s: float = 60.0,
) -> list[Rollout]:
    """Manifest-free path: trust the caller's task_idx, GET the parquet,
    return rows. Use after ``fetch_metadata`` has bounded the valid range.
    Skips the full manifest download (5 MB+ at scale).
    """
    object_url = _task_object_url(base_url, task_idx)
    payload = _http_get_bytes(object_url, timeout_s)
    table = pq.read_table(io.BytesIO(payload))
    return _table_to_rollouts(table)


def _lookup(manifest: list[ManifestEntry], task_idx: int) -> ManifestEntry:
    # The manifest is append-only with ``task_idx == line number``, so
    # in the common case indexing directly works. Fall back to linear
    # scan if the file's been rewritten or has gaps.
    if 0 <= task_idx < len(manifest) and manifest[task_idx].task_idx == task_idx:
        return manifest[task_idx]
    for e in manifest:
        if e.task_idx == task_idx:
            return e
    raise LookupError(f"task_idx={task_idx} not in manifest (len={len(manifest)})")


def _http_get_bytes(url: str, timeout_s: float) -> bytes:
    with httpx.Client(timeout=timeout_s, follow_redirects=True) as client:
        r = client.get(url)
        r.raise_for_status()
        return r.content


def _table_to_rollouts(table) -> list[Rollout]:  # type: ignore[no-untyped-def]
    cols = {name: table.column(name).to_pylist() for name in table.schema.names}
    n = table.num_rows
    out: list[Rollout] = []
    for i in range(n):
        out.append(Rollout(
            rollout_id=RolloutId(cols["rollout_id"][i]),
            env_name=EnvName(cols["env_name"][i]),
            task_id=TaskId(cols["task_id"][i]),
            teacher_name=TeacherName(cols["teacher_name"][i]),
            sample_idx=int(cols["sample_idx"][i]),
            temperature=float(cols["temperature"][i]),
            top_p=_opt_float(cols["top_p"][i]),
            seed=_opt_int(cols["seed"][i]),
            status=RolloutStatus(cols["status"][i]),
            reward=_opt_float(cols["reward"][i]),
            steps=_opt_int(cols["steps"][i]),
            tokens_in=_opt_int(cols["tokens_in"][i]),
            tokens_out=_opt_int(cols["tokens_out"][i]),
            schema_version=str(cols["schema_version"][i]),
            extra_compressed=cols["extra_compressed"][i],
            extra_sha256=str(cols["extra_sha256"][i]),
            producer_id="cortex-publisher",
            created_at=cols["created_at"][i],
        ))
    return out


def _opt_float(v) -> float | None:  # type: ignore[no-untyped-def]
    return None if v is None else float(v)


def _opt_int(v) -> int | None:  # type: ignore[no-untyped-def]
    return None if v is None else int(v)
