"""
SWE-INFINITE Cache Module (Read-Only)

Three-level cache for reading expansion tasks (produced by the mining pipeline):
  L1: Local filesystem (fast, per-machine)
  L1.5: R2 PRIVATE staging via authenticated S3 — optional, internal-only.
        Enabled when R2 credentials are configured. Lets internal
        evaluators test brand-new tasks before they finish the
        publication delay imposed by the release pipeline.
  L2: R2 public bucket via HTTP — what external evaluators see.
      Only tasks that have served their publication delay live here.

SECURITY: the staging credentials must NEVER be baked into images
distributed to external evaluators. Anyone with them can read tasks
ahead of the public release schedule.
"""

import json
import os
from pathlib import Path
from typing import Optional, Dict, Any
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError


class LocalCache:
    """Local filesystem cache keyed by instance_id."""

    def __init__(self, cache_dir: str = "/tmp/swe-infinite-cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_path(self, instance_id: str) -> Path:
        try:
            num = int(instance_id)
            return self.cache_dir / f"task_{num:011d}.json"
        except (ValueError, TypeError):
            return self.cache_dir / f"{instance_id}.json"

    def load(self, instance_id: str) -> Optional[Dict[str, Any]]:
        path = self._get_path(instance_id)
        if path.exists():
            with open(path, 'r') as f:
                return json.load(f)
        return None

    def save(self, instance_id: str, data: Dict[str, Any]) -> None:
        path = self._get_path(instance_id)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def exists(self, instance_id: str) -> bool:
        return self._get_path(instance_id).exists()


class R2PublicCache:
    """Read-only R2 cache via public HTTP URL."""

    DEFAULT_BASE_URL = "https://pub-7882418a56434a479bf9a7febd660b36.r2.dev"
    DEFAULT_PREFIX = "bugs"

    def __init__(
        self,
        base_url: Optional[str] = None,
        prefix: Optional[str] = None,
    ):
        self.base_url = (base_url or os.getenv("R2_PUBLIC_URL") or self.DEFAULT_BASE_URL).rstrip("/")
        self.prefix = prefix if prefix is not None else (os.getenv("R2_PUBLIC_PREFIX") or self.DEFAULT_PREFIX)
        print(f"[CACHE] R2 public: {self.base_url}/{self.prefix}")

    @property
    def enabled(self) -> bool:
        return True

    @staticmethod
    def _format_key(task_id: str) -> str:
        """Format task_id to R2 filename: task_00000000001.json"""
        try:
            num = int(task_id)
            return f"task_{num:011d}.json"
        except (ValueError, TypeError):
            return f"{task_id}.json"

    def _get_url(self, instance_id: str) -> str:
        filename = self._format_key(instance_id)
        if self.prefix:
            return f"{self.base_url}/{self.prefix}/{filename}"
        return f"{self.base_url}/{filename}"

    def load(self, instance_id: str) -> Optional[Dict[str, Any]]:
        url = self._get_url(instance_id)
        try:
            req = Request(url, headers={"Accept": "application/json", "User-Agent": "swe-infinite/1.0"})
            with urlopen(req, timeout=30) as resp:
                return json.loads(resp.read())
        except HTTPError as e:
            if e.code == 404:
                return None
            print(f"[CACHE] R2 HTTP error for {instance_id}: {e.code} {e.reason}")
            return None
        except (URLError, TimeoutError) as e:
            print(f"[CACHE] R2 fetch error for {instance_id}: {e}")
            return None

    def exists(self, instance_id: str) -> bool:
        url = self._get_url(instance_id)
        try:
            req = Request(url, method="HEAD", headers={"User-Agent": "swe-infinite/1.0"})
            with urlopen(req, timeout=10):
                return True
        except Exception:
            return False


class R2StagingCache:
    """Authenticated S3 reader for the PRIVATE staging area.

    Only useful for internal evaluators that need to test a task before
    it serves its publication delay. External evaluators must run this
    environment WITHOUT R2 credentials — otherwise the delay is
    bypassed.

    Construction is lazy on boto3 so deployments that never use staging
    don't pay the import cost.
    """

    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket: str,
        prefix: str = "staging",
    ):
        import boto3
        from botocore.config import Config

        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.s3 = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name="auto",
            config=Config(signature_version="s3v4", retries={"max_attempts": 3}),
        )
        print(f"[CACHE] R2 staging enabled: s3://{bucket}/{self.prefix}/")

    @property
    def enabled(self) -> bool:
        return True

    def _key(self, instance_id: str) -> str:
        try:
            num = int(instance_id)
            return f"{self.prefix}/task_{num:011d}.json"
        except (ValueError, TypeError):
            return f"{self.prefix}/{instance_id}.json"

    def load(self, instance_id: str) -> Optional[Dict[str, Any]]:
        from botocore.exceptions import ClientError
        key = self._key(instance_id)
        try:
            resp = self.s3.get_object(Bucket=self.bucket, Key=key)
            return json.loads(resp["Body"].read())
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code in ("NoSuchKey", "404"):
                return None
            print(f"[CACHE] R2 staging error for {instance_id}: {e}")
            return None
        except Exception as e:
            print(f"[CACHE] R2 staging unexpected error for {instance_id}: {e}")
            return None

    def exists(self, instance_id: str) -> bool:
        from botocore.exceptions import ClientError
        try:
            self.s3.head_object(Bucket=self.bucket, Key=self._key(instance_id))
            return True
        except ClientError:
            return False


def _resolve_staging_config(
    *,
    endpoint: Optional[str],
    access_key: Optional[str],
    secret_key: Optional[str],
    bucket: Optional[str],
    prefix: Optional[str],
) -> Optional[dict]:
    """Return a staging config dict iff every required field is set
    (either explicitly or via env vars). Returns None otherwise so
    external deployments transparently degrade to public-only reads.
    """
    endpoint = endpoint or os.getenv("R2_STAGING_ENDPOINT") or os.getenv("R2_ENDPOINT")
    access_key = access_key or os.getenv("R2_STAGING_ACCESS_KEY") or os.getenv("R2_ACCESS_KEY")
    secret_key = secret_key or os.getenv("R2_STAGING_SECRET_KEY") or os.getenv("R2_SECRET_KEY")
    bucket = bucket or os.getenv("R2_STAGING_BUCKET") or os.getenv("R2_PRIVATE_BUCKET")
    prefix = prefix if prefix is not None else (os.getenv("R2_STAGING_PREFIX") or "staging")
    if not (endpoint and access_key and secret_key and bucket):
        return None
    return {
        "endpoint": endpoint,
        "access_key": access_key,
        "secret_key": secret_key,
        "bucket": bucket,
        "prefix": prefix,
    }


class TwoLevelCache:
    """Read cache with optional staging tier.

    Read path:
      L1 (local) hit            -> return
      L1.5 (R2 staging) hit     -> save to L1, return  [internal only]
      L2 (R2 public HTTP) hit   -> save to L1, return
      miss                      -> None

    Naming stays "TwoLevelCache" for backward compatibility with
    existing callers; the staging tier is opt-in and inert without
    credentials.
    """

    def __init__(
        self,
        local_cache_dir: str = "/tmp/swe-infinite-cache",
        r2_base_url: Optional[str] = None,
        r2_prefix: Optional[str] = None,
        # Optional staging access (internal use only — see module doc).
        r2_endpoint: Optional[str] = None,
        r2_access_key: Optional[str] = None,
        r2_secret_key: Optional[str] = None,
        r2_staging_bucket: Optional[str] = None,
        r2_staging_prefix: Optional[str] = None,
        # Deprecated alias, kept so historical callers don't break.
        r2_bucket: Optional[str] = None,
    ):
        self.local = LocalCache(local_cache_dir)
        self.r2 = R2PublicCache(
            base_url=r2_base_url,
            prefix=r2_prefix,
        )

        staging_cfg = _resolve_staging_config(
            endpoint=r2_endpoint,
            access_key=r2_access_key,
            secret_key=r2_secret_key,
            bucket=r2_staging_bucket or r2_bucket,
            prefix=r2_staging_prefix,
        )
        if staging_cfg is not None:
            try:
                self.staging: Optional[R2StagingCache] = R2StagingCache(**staging_cfg)
            except Exception as e:
                # Don't let a broken staging config break public reads —
                # external deployments that accidentally set a partial
                # config still need to function on L2 alone.
                print(f"[CACHE] R2 staging init failed: {e}; falling back to public-only")
                self.staging = None
        else:
            self.staging = None

    def load(self, instance_id: str) -> Optional[Dict[str, Any]]:
        """Load task by instance_id (L1 -> L1.5 staging -> L2 public)."""
        data = self.local.load(instance_id)
        if data is not None:
            return data

        if self.staging is not None:
            data = self.staging.load(instance_id)
            if data is not None:
                self.local.save(instance_id, data)
                return data

        if self.r2.enabled:
            data = self.r2.load(instance_id)
            if data is not None:
                self.local.save(instance_id, data)
                return data

        return None

    def exists(self, instance_id: str) -> bool:
        if self.local.exists(instance_id):
            return True
        if self.staging is not None and self.staging.exists(instance_id):
            return True
        return self.r2.exists(instance_id)

