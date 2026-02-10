#!/usr/bin/env python3
"""
Release Tasks Script

Copies tasks from private R2 to public R2 for controlled release.

Usage:
    # Daemon mode: release 2 tasks every hour (default)
    python release_tasks.py --daemon

    # Daemon mode with custom settings
    python release_tasks.py --daemon --batch-size 5 --interval 1800

    # One-time release: tasks 0-99 (inclusive)
    python release_tasks.py --up-to 99

    # One-time release: specific range
    python release_tasks.py --from 100 --up-to 199

    # Dry run (show what would be copied)
    python release_tasks.py --up-to 99 --dry-run

Environment variables:
    R2_ENDPOINT_URL, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY
    R2_BUCKET (private, e.g. affine-swe-synth-private)
    R2_PUBLIC_BUCKET (public, e.g. affine-swe-synth)
"""

import os
import sys
import json
import time
import argparse
from typing import Optional

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError


def log(msg: str) -> None:
    """Print log message with timestamp."""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


class TaskReleaser:
    """Copies tasks from private R2 to public R2."""

    def __init__(
        self,
        # Private R2 (source)
        private_endpoint_url: str,
        private_bucket: str,
        private_access_key_id: str,
        private_secret_access_key: str,
        # Public R2 (destination)
        public_endpoint_url: str,
        public_bucket: str,
        public_access_key_id: str,
        public_secret_access_key: str,
        # Common
        prefix: str = "bugs",
    ):
        self.prefix = prefix

        # Private R2 client (source)
        self.private_s3 = boto3.client(
            's3',
            endpoint_url=private_endpoint_url,
            aws_access_key_id=private_access_key_id,
            aws_secret_access_key=private_secret_access_key,
            config=Config(signature_version='s3v4', retries={'max_attempts': 3})
        )
        self.private_bucket = private_bucket

        # Public R2 client (destination)
        self.public_s3 = boto3.client(
            's3',
            endpoint_url=public_endpoint_url,
            aws_access_key_id=public_access_key_id,
            aws_secret_access_key=public_secret_access_key,
            config=Config(signature_version='s3v4', retries={'max_attempts': 3})
        )
        self.public_bucket = public_bucket

    def _get_task_key(self, task_id: int) -> str:
        return f"{self.prefix}/task_{task_id:011d}.json"

    def _get_metadata_key(self) -> str:
        return f"{self.prefix}/metadata.json"

    def task_exists_in_private(self, task_id: int) -> bool:
        """Check if task exists in private R2."""
        try:
            self.private_s3.head_object(
                Bucket=self.private_bucket,
                Key=self._get_task_key(task_id)
            )
            return True
        except ClientError:
            return False

    def task_exists_in_public(self, task_id: int) -> bool:
        """Check if task already exists in public R2."""
        try:
            self.public_s3.head_object(
                Bucket=self.public_bucket,
                Key=self._get_task_key(task_id)
            )
            return True
        except ClientError:
            return False

    def copy_task(self, task_id: int) -> bool:
        """Copy a single task from private to public R2."""
        key = self._get_task_key(task_id)
        try:
            # Read from private
            response = self.private_s3.get_object(
                Bucket=self.private_bucket,
                Key=key
            )
            data = response['Body'].read()

            # Write to public
            self.public_s3.put_object(
                Bucket=self.public_bucket,
                Key=key,
                Body=data,
                ContentType='application/json'
            )
            return True
        except ClientError as e:
            log(f"Error copying task {task_id}: {e}")
            return False

    def load_private_metadata(self) -> dict:
        """Load metadata from private R2."""
        try:
            response = self.private_s3.get_object(
                Bucket=self.private_bucket,
                Key=self._get_metadata_key()
            )
            return json.loads(response['Body'].read().decode('utf-8'))
        except ClientError:
            return {"tasks": {"completed_up_to": -1}}

    def load_public_metadata(self) -> dict:
        """Load metadata from public R2."""
        try:
            response = self.public_s3.get_object(
                Bucket=self.public_bucket,
                Key=self._get_metadata_key()
            )
            return json.loads(response['Body'].read().decode('utf-8'))
        except ClientError:
            return {
                "version": 3,
                "last_updated": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "tasks": {"completed_up_to": -1}
            }

    def save_public_metadata(self, metadata: dict) -> None:
        """Save metadata to public R2."""
        metadata["last_updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ")
        self.public_s3.put_object(
            Bucket=self.public_bucket,
            Key=self._get_metadata_key(),
            Body=json.dumps(metadata, indent=2).encode('utf-8'),
            ContentType='application/json'
        )

    def get_release_status(self) -> tuple[int, int]:
        """
        Get current release status.

        Returns:
            (released_up_to, available_up_to):
            - released_up_to: completed_up_to in public R2
            - available_up_to: completed_up_to in private R2
        """
        public_meta = self.load_public_metadata()
        private_meta = self.load_private_metadata()

        released_up_to = public_meta["tasks"].get("completed_up_to", -1)
        available_up_to = private_meta["tasks"].get("completed_up_to", -1)

        return released_up_to, available_up_to

    def release(
        self,
        up_to: int,
        start_from: int = 0,
        dry_run: bool = False,
    ) -> dict:
        """
        Release tasks from private to public R2.

        Args:
            up_to: Release tasks up to this task_id (inclusive)
            start_from: Start from this task_id (default 0)
            dry_run: If True, only show what would be done

        Returns:
            Stats dict with copied/skipped/missing counts
        """
        stats = {"copied": 0, "skipped": 0, "missing": 0, "errors": 0}

        log(f"Releasing tasks {start_from} to {up_to}")
        if dry_run:
            log("(DRY RUN - no changes will be made)")

        for task_id in range(start_from, up_to + 1):
            # Check if exists in private
            if not self.task_exists_in_private(task_id):
                log(f"Task {task_id}: MISSING in private R2")
                stats["missing"] += 1
                continue

            # Check if already in public
            if self.task_exists_in_public(task_id):
                stats["skipped"] += 1
                continue

            # Copy
            if dry_run:
                log(f"Task {task_id}: would copy")
                stats["copied"] += 1
            else:
                if self.copy_task(task_id):
                    log(f"Task {task_id}: copied")
                    stats["copied"] += 1
                else:
                    stats["errors"] += 1

        # Update public metadata
        if not dry_run and stats["copied"] > 0:
            metadata = self.load_public_metadata()
            current_released = metadata["tasks"].get("completed_up_to", -1)
            if up_to > current_released:
                metadata["tasks"]["completed_up_to"] = up_to
                self.save_public_metadata(metadata)
                log(f"Updated completed_up_to: {current_released} -> {up_to}")

        log(f"Done: {stats['copied']} copied, {stats['skipped']} skipped, "
            f"{stats['missing']} missing, {stats['errors']} errors")

        return stats

    def release_batch(self, batch_size: int, dry_run: bool = False) -> dict:
        """
        Release next batch of tasks.

        Args:
            batch_size: Number of tasks to release
            dry_run: If True, only show what would be done

        Returns:
            Stats dict
        """
        released_up_to, available_up_to = self.get_release_status()

        # Calculate how many tasks can be released
        pending = available_up_to - released_up_to
        if pending <= 0:
            log(f"No new tasks to release (released={released_up_to}, available={available_up_to})")
            return {"copied": 0, "skipped": 0, "missing": 0, "errors": 0}

        # Release up to batch_size tasks
        to_release = min(batch_size, pending)
        new_up_to = released_up_to + to_release

        log(f"Releasing {to_release} tasks ({released_up_to + 1} to {new_up_to}), "
            f"{pending - to_release} remaining after this batch")

        return self.release(
            up_to=new_up_to,
            start_from=released_up_to + 1,
            dry_run=dry_run,
        )

    def run_daemon(
        self,
        batch_size: int = 2,
        interval: int = 3600,
        dry_run: bool = False,
    ) -> None:
        """
        Run as daemon, releasing tasks periodically.

        Args:
            batch_size: Number of tasks to release per interval
            interval: Seconds between releases (default: 3600 = 1 hour)
            dry_run: If True, only show what would be done
        """
        log("=" * 50)
        log(f"Starting release daemon")
        log(f"Batch size: {batch_size} tasks")
        log(f"Interval: {interval} seconds ({interval/3600:.1f} hours)")
        if dry_run:
            log("DRY RUN MODE - no changes will be made")
        log("=" * 50)

        while True:
            try:
                self.release_batch(batch_size, dry_run=dry_run)
            except Exception as e:
                log(f"Error during release: {e}")

            log(f"Sleeping {interval} seconds until next release...")
            time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(
        description="Release tasks from private R2 to public R2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Daemon mode
    parser.add_argument("--daemon", action="store_true",
                        help="Run as daemon, releasing tasks periodically")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Tasks to release per interval in daemon mode (default: 2)")
    parser.add_argument("--interval", type=int, default=3600,
                        help="Seconds between releases in daemon mode (default: 3600 = 1 hour)")

    # One-time release mode
    parser.add_argument("--up-to", type=int,
                        help="Release tasks up to this task_id (inclusive)")
    parser.add_argument("--from", dest="start_from", type=int, default=0,
                        help="Start from this task_id (default: 0)")

    # Common
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without making changes")
    parser.add_argument("--prefix", type=str, default="bugs",
                        help="R2 key prefix (default: bugs)")
    parser.add_argument("--status", action="store_true",
                        help="Show current release status and exit")

    args = parser.parse_args()

    # Check required environment variables
    required = ["R2_ENDPOINT_URL", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY",
                "R2_BUCKET", "R2_PUBLIC_BUCKET"]

    missing = [e for e in required if not os.getenv(e)]
    if missing:
        print(f"Missing environment variables: {missing}")
        print("\nRequired: R2_ENDPOINT_URL, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET, R2_PUBLIC_BUCKET")
        sys.exit(1)

    # Use same credentials for both buckets
    endpoint_url = os.getenv("R2_ENDPOINT_URL")
    access_key_id = os.getenv("R2_ACCESS_KEY_ID")
    secret_access_key = os.getenv("R2_SECRET_ACCESS_KEY")

    releaser = TaskReleaser(
        # Private R2 (source)
        private_endpoint_url=endpoint_url,
        private_bucket=os.getenv("R2_BUCKET"),
        private_access_key_id=access_key_id,
        private_secret_access_key=secret_access_key,
        # Public R2 (destination)
        public_endpoint_url=endpoint_url,
        public_bucket=os.getenv("R2_PUBLIC_BUCKET"),
        public_access_key_id=access_key_id,
        public_secret_access_key=secret_access_key,
        prefix=args.prefix,
    )

    # Status mode
    if args.status:
        released, available = releaser.get_release_status()
        pending = available - released
        print(f"Released up to: {released}")
        print(f"Available up to: {available}")
        print(f"Pending: {pending} tasks")
        return

    # Daemon mode
    if args.daemon:
        releaser.run_daemon(
            batch_size=args.batch_size,
            interval=args.interval,
            dry_run=args.dry_run,
        )
        return

    # One-time release mode
    if args.up_to is None:
        parser.error("Either --daemon or --up-to is required")

    releaser.release(
        up_to=args.up_to,
        start_from=args.start_from,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
