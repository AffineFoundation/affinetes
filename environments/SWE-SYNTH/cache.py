"""
SWE-SYNTH Cache Module (Read-Only)

Two-level cache for reading pre-generated tasks:
1. Local cache (fast, per-machine)
2. R2 public CDN (shared, no auth needed)
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any


class LocalCache:
    """Local file system cache"""

    def __init__(self, cache_dir: str = "/tmp/swe-synth-cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_path(self, task_id: int) -> Path:
        return self.cache_dir / f"task_{task_id}.json"

    def load(self, task_id: int) -> Optional[Dict[str, Any]]:
        path = self._get_path(task_id)
        if path.exists():
            with open(path, 'r') as f:
                return json.load(f)
        return None

    def save(self, task_id: int, data: Dict[str, Any]) -> None:
        path = self._get_path(task_id)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def exists(self, task_id: int) -> bool:
        return self._get_path(task_id).exists()


class R2ReadOnlyCache:
    """
    Read-only R2 cache using public CDN URL.
    No credentials required.
    """

    def __init__(
        self,
        public_read_url: str = "https://pub-4b43a94ed07d4ac38fae3f4cb5070d6c.r2.dev",
        prefix: str = "bugs",
    ):
        self.public_read_url = public_read_url
        self.prefix = prefix

    def _get_key(self, task_id: int) -> str:
        return f"{self.prefix}/task_{task_id:011d}.json"

    def load(self, task_id: int) -> Optional[Dict[str, Any]]:
        try:
            import httpx
            url = f"{self.public_read_url}/{self._get_key(task_id)}"
            response = httpx.get(url, timeout=30)
            if response.status_code == 200:
                if not response.content:
                    print(f"R2 read error: empty response for task {task_id}")
                    return None
                return response.json()
            elif response.status_code == 404:
                return None
            else:
                print(f"R2 read error: HTTP {response.status_code}")
                return None
        except Exception as e:
            print(f"R2 read error for task {task_id}: {e}")
            return None

    def exists(self, task_id: int) -> bool:
        try:
            import httpx
            url = f"{self.public_read_url}/{self._get_key(task_id)}"
            response = httpx.head(url, timeout=10)
            return response.status_code == 200
        except Exception:
            return False


class TwoLevelCache:
    """
    Two-level read cache: Local (L1) + R2 (L2)

    Read path:
    1. Check local cache -> hit: return
    2. Check R2 cache -> hit: save to local, return
    3. Miss: return None
    """

    def __init__(
        self,
        local_cache_dir: str = "/tmp/swe-synth-cache",
        r2_public_read_url: str = "https://pub-4b43a94ed07d4ac38fae3f4cb5070d6c.r2.dev",
        r2_prefix: str = "bugs",
    ):
        self.local = LocalCache(local_cache_dir)
        self.r2 = R2ReadOnlyCache(
            public_read_url=r2_public_read_url,
            prefix=r2_prefix,
        ) if r2_public_read_url else None

    def load(self, task_id: int) -> Optional[Dict[str, Any]]:
        """Load from cache (L1 -> L2)"""
        # L1: Local cache
        data = self.local.load(task_id)
        if data is not None:
            return data

        # L2: R2 cache
        if self.r2:
            data = self.r2.load(task_id)
            if data is not None:
                # Populate L1 cache
                self.local.save(task_id, data)
                return data

        return None

    def exists(self, task_id: int) -> bool:
        """Check if cache exists (L1 or L2)"""
        if self.local.exists(task_id):
            return True
        if self.r2 and self.r2.exists(task_id):
            return True
        return False
