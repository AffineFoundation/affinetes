"""
SWE-SYNTH Cache Module

Two-level cache with distributed locking and conflict resolution:
1. Local cache (fast, per-machine)
2. R2 cache (distributed, shared across machines)

Lock mechanism:
- Try to acquire lock before generating
- If lock held by another, wait and retry
- After generation, check for conflicts before saving
- If conflict found, discard own result and use existing
"""

import os
import json
import time
import socket
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError


class CacheLockError(Exception):
    """Raised when unable to acquire distributed lock after retries"""
    pass


class CacheBackend(ABC):
    """Abstract cache backend interface"""

    @abstractmethod
    def load(self, task_id: int) -> Optional[Dict[str, Any]]:
        """Load cached data for task_id, return None if not found"""
        pass

    @abstractmethod
    def save(self, task_id: int, data: Dict[str, Any]) -> None:
        """Save data to cache"""
        pass

    @abstractmethod
    def exists(self, task_id: int) -> bool:
        """Check if cache exists for task_id"""
        pass


class LocalCache(CacheBackend):
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


class R2Cache(CacheBackend):
    """
    Cloudflare R2 cache with distributed locking.

    Supports read/write separation:
    - Write: private endpoint (requires auth)
    - Read: public CDN (faster, no auth needed)

    Lock mechanism uses a separate lock file with timestamp-based expiry.
    """

    def __init__(
        self,
        endpoint_url: str,
        bucket: str,
        access_key_id: str,
        secret_access_key: str,
        prefix: str = "bugs",
        lock_timeout: int = 300,  # Lock expires after 5 minutes
        public_read_url: str = "https://pub-4b43a94ed07d4ac38fae3f4cb5070d6c.r2.dev",
    ):
        self.bucket = bucket
        self.prefix = prefix
        self.lock_timeout = lock_timeout
        self.public_read_url = public_read_url  # e.g., https://pub-xxx.r2.dev

        # Machine identifier for lock ownership
        self.machine_id = f"{socket.gethostname()}_{os.getpid()}_{int(time.time())}"

        # Initialize S3 client for R2 (for writes and lock operations)
        self.s3 = boto3.client(
            's3',
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            config=Config(
                signature_version='s3v4',
                retries={'max_attempts': 3}
            )
        )

    def _get_key(self, task_id: int) -> str:
        return f"{self.prefix}/task_{task_id}.json"

    def _get_lock_key(self, task_id: int) -> str:
        return f"{self.prefix}/locks/task_{task_id}.lock"

    def load(self, task_id: int) -> Optional[Dict[str, Any]]:
        # Try public CDN first (faster, no auth needed)
        if self.public_read_url:
            try:
                import httpx
                url = f"{self.public_read_url}/{self._get_key(task_id)}"
                response = httpx.get(url, timeout=30)
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 404:
                    return None
                # Fall through to S3 API on other errors
            except Exception:
                pass  # Fall back to S3 API

        # Fall back to S3 API
        try:
            response = self.s3.get_object(
                Bucket=self.bucket,
                Key=self._get_key(task_id)
            )
            return json.loads(response['Body'].read().decode('utf-8'))
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                return None
            raise

    def save(self, task_id: int, data: Dict[str, Any]) -> None:
        self.s3.put_object(
            Bucket=self.bucket,
            Key=self._get_key(task_id),
            Body=json.dumps(data, indent=2, default=str).encode('utf-8'),
            ContentType='application/json'
        )

    def exists(self, task_id: int) -> bool:
        try:
            self.s3.head_object(Bucket=self.bucket, Key=self._get_key(task_id))
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return False
            raise

    def _get_lock_info(self, task_id: int) -> Optional[Dict[str, Any]]:
        """Get current lock info, return None if no lock"""
        try:
            response = self.s3.get_object(
                Bucket=self.bucket,
                Key=self._get_lock_key(task_id)
            )
            return json.loads(response['Body'].read().decode('utf-8'))
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                return None
            raise

    def _write_lock(self, task_id: int) -> Dict[str, Any]:
        """Write lock file and return lock data"""
        lock_data = {
            "machine_id": self.machine_id,
            "timestamp": time.time(),
            "expires_at": time.time() + self.lock_timeout
        }
        self.s3.put_object(
            Bucket=self.bucket,
            Key=self._get_lock_key(task_id),
            Body=json.dumps(lock_data).encode('utf-8'),
            ContentType='application/json'
        )
        return lock_data

    def acquire_lock(
        self,
        task_id: int,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ) -> bool:
        """
        Try to acquire distributed lock for task_id.

        Returns True if lock acquired.
        Raises CacheLockError if unable to acquire after retries.
        """
        for attempt in range(max_retries):
            # Check if data already exists (no need to lock)
            if self.exists(task_id):
                return True  # Data exists, no lock needed

            # Check current lock status
            lock_info = self._get_lock_info(task_id)

            if lock_info:
                # Lock exists, check if expired
                if lock_info.get('expires_at', 0) > time.time():
                    # Lock is valid and held by another machine
                    if lock_info.get('machine_id') != self.machine_id:
                        print(f"Lock held by {lock_info.get('machine_id')}, waiting...")
                        time.sleep(retry_delay)
                        continue
                    else:
                        # We already hold the lock
                        return True
                # Lock expired, can take over

            # Try to acquire lock
            self._write_lock(task_id)

            # Small delay for consistency
            time.sleep(0.2)

            # Verify we got the lock
            current_lock = self._get_lock_info(task_id)
            if current_lock and current_lock.get('machine_id') == self.machine_id:
                return True

            # Someone else got the lock, retry
            print(f"Lock contention, retrying... (attempt {attempt + 1}/{max_retries})")
            time.sleep(retry_delay)

        # After all retries, check if data exists (another machine may have completed)
        if self.exists(task_id):
            return True

        raise CacheLockError(
            f"Failed to acquire lock for task {task_id} after {max_retries} retries"
        )

    def release_lock(self, task_id: int) -> None:
        """Release distributed lock for task_id"""
        try:
            # Only delete if we own the lock
            lock_info = self._get_lock_info(task_id)
            if lock_info and lock_info.get('machine_id') == self.machine_id:
                self.s3.delete_object(
                    Bucket=self.bucket,
                    Key=self._get_lock_key(task_id)
                )
        except ClientError:
            pass  # Lock already released or doesn't exist


class TwoLevelCache:
    """
    Two-level cache: Local (L1) + R2 (L2)

    Read path:
    1. Check local cache → hit: return
    2. Check R2 cache → hit: save to local, return
    3. Miss: return None

    Write path (with locking and conflict resolution):
    1. Acquire lock
    2. Double-check if data exists (another machine may have generated)
    3. If exists: use existing, release lock
    4. If not: save to R2, save to local, release lock
    """

    def __init__(
        self,
        local_cache_dir: str = "/tmp/swe-synth-cache",
        r2_endpoint_url: str = None,
        r2_bucket: str = None,
        r2_access_key_id: str = None,
        r2_secret_access_key: str = None,
        r2_prefix: str = "bugs",
        r2_public_read_url: str = "https://pub-4b43a94ed07d4ac38fae3f4cb5070d6c.r2.dev",
    ):
        self.local = LocalCache(local_cache_dir)

        if r2_endpoint_url and r2_bucket and r2_access_key_id and r2_secret_access_key:
            self.r2 = R2Cache(
                endpoint_url=r2_endpoint_url,
                bucket=r2_bucket,
                access_key_id=r2_access_key_id,
                secret_access_key=r2_secret_access_key,
                prefix=r2_prefix,
                public_read_url=r2_public_read_url,
            )
        else:
            self.r2 = None

    def load(self, task_id: int) -> Optional[Dict[str, Any]]:
        """Load from cache (L1 → L2)"""
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

    def save(self, task_id: int, data: Dict[str, Any]) -> None:
        """Save to cache (L2 first, then L1)"""
        # Save to L2 (R2) first
        if self.r2:
            self.r2.save(task_id, data)

        # Save to L1 (local)
        self.local.save(task_id, data)

    def save_if_not_exists(self, task_id: int, data: Dict[str, Any]) -> tuple[bool, Optional[Dict[str, Any]]]:
        """
        Save to cache only if not already exists (conflict resolution).

        Returns:
            (saved, existing_data)
            - (True, None) if saved successfully
            - (False, existing_data) if data already exists (conflict)
        """
        # Check R2 first for existing data
        if self.r2:
            existing = self.r2.load(task_id)
            if existing:
                # Conflict: data already exists, use existing
                self.local.save(task_id, existing)
                return False, existing

        # No conflict, save
        self.save(task_id, data)
        return True, None

    def exists(self, task_id: int) -> bool:
        """Check if cache exists (L1 or L2)"""
        if self.local.exists(task_id):
            return True
        if self.r2 and self.r2.exists(task_id):
            return True
        return False

    def acquire_lock(self, task_id: int) -> bool:
        """Acquire distributed lock before generating"""
        if self.r2:
            return self.r2.acquire_lock(task_id)
        return True  # No lock needed for local-only mode

    def release_lock(self, task_id: int) -> None:
        """Release distributed lock after generating"""
        if self.r2:
            self.r2.release_lock(task_id)
