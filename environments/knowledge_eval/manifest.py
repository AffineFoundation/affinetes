"""Runtime loader for the build-time manifest.

The image ships ``/app/data/manifest.json`` plus one ``<task_type>.jsonl``
per benchmark. We load every jsonl into memory once at module import (a
few MB total) so per-call lookups are O(1) and the env stays stateless.
"""

import bisect
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

DATA_DIR = Path("/app/data")


class Manifest:
    def __init__(self, data_dir: Path = DATA_DIR) -> None:
        manifest_path = data_dir / "manifest.json"
        if not manifest_path.exists():
            raise RuntimeError(
                f"manifest.json not found at {manifest_path}. "
                "Did the docker build run preprocess.py?"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.total: int = manifest["total"]
        self.ranges: List[Dict[str, Any]] = manifest["ranges"]

        # Per task_type rows kept in memory.
        self._rows: Dict[str, List[Dict[str, Any]]] = {}
        self._by_type_count: Dict[str, int] = {}
        for entry in self.ranges:
            tt = entry["task_type"]
            path = data_dir / entry["file"]
            rows: List[Dict[str, Any]] = []
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
            if len(rows) != entry["count"]:
                raise RuntimeError(
                    f"manifest count mismatch for {tt}: "
                    f"expected {entry['count']}, loaded {len(rows)}"
                )
            self._rows[tt] = rows
            self._by_type_count[tt] = len(rows)

        # Sorted starts list for bisect-based global -> (task_type, local) routing.
        self._starts: List[int] = [e["start"] for e in self.ranges]

    @property
    def task_types(self) -> List[str]:
        return [e["task_type"] for e in self.ranges]

    def count(self, task_type: str) -> int:
        if task_type not in self._by_type_count:
            raise ValueError(
                f"Unknown task_type {task_type!r}. Available: {self.task_types}"
            )
        return self._by_type_count[task_type]

    def resolve(self, task_id: int) -> Tuple[str, int, Dict[str, Any]]:
        """Map a global ``task_id`` to ``(task_type, local_id, sample)``.

        Negative ids and ids past ``total`` raise ``ValueError`` rather
        than wrapping silently — silent wrap-around is exactly the kind
        of bug you only notice on a leaderboard.
        """
        if task_id < 0:
            raise ValueError(f"task_id must be non-negative, got {task_id}")
        if task_id >= self.total:
            raise ValueError(
                f"task_id {task_id} out of range [0, {self.total - 1}]"
            )
        idx = bisect.bisect_right(self._starts, task_id) - 1
        entry = self.ranges[idx]
        local_id = task_id - entry["start"]
        return entry["task_type"], local_id, self._rows[entry["task_type"]][local_id]

    def get_local(self, task_type: str, local_id: int) -> Dict[str, Any]:
        if task_type not in self._rows:
            raise ValueError(
                f"Unknown task_type {task_type!r}. Available: {self.task_types}"
            )
        rows = self._rows[task_type]
        if local_id < 0 or local_id >= len(rows):
            raise ValueError(
                f"local task_id {local_id} out of range for {task_type} "
                f"(0..{len(rows) - 1})"
            )
        return rows[local_id]

    def rows(self, task_type: str) -> List[Dict[str, Any]]:
        """Return the in-memory rows for a task_type (read-only use)."""
        if task_type not in self._rows:
            raise ValueError(
                f"Unknown task_type {task_type!r}. Available: {self.task_types}"
            )
        return self._rows[task_type]

    def stats(self) -> Dict[str, Any]:
        return {
            "total": self.total,
            "ranges": [
                {
                    "task_type": e["task_type"],
                    "start": e["start"],
                    "end": e["end"],
                    "count": e["count"],
                }
                for e in self.ranges
            ],
        }
