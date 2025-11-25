import logging
import os
import json
import random
import asyncio
import aiohttp
import fcntl
from pathlib import Path
from botocore.config import Config
from aiobotocore.session import get_session
from typing import Any, Optional, Dict, List

# R2 Storage Configuration
FOLDER = os.getenv("R2_FOLDER", "affine")
BUCKET = os.getenv("R2_BUCKET_ID", "00523074f51300584834607253cae0fa")
ACCESS = os.getenv("R2_WRITE_ACCESS_KEY_ID", "")
SECRET = os.getenv("R2_WRITE_SECRET_ACCESS_KEY", "")
ENDPOINT = f"https://{BUCKET}.r2.cloudflarestorage.com"

# Public read configuration
PUBLIC_READ = os.getenv("R2_PUBLIC_READ", "true").lower() == "true"
R2_PUBLIC_BASE = os.getenv("R2_PUBLIC_BASE", "https://pub-bf429ea7a5694b99adaf3d444cbbe64d.r2.dev")

# Cache configuration
CACHE_DIR = os.getenv("DATASET_CACHE_DIR", "/tmp/dataset_cache")
DOWNLOAD_CONCURRENCY = int(os.getenv("DOWNLOAD_CONCURRENCY", "1"))

# Logger
logger = logging.getLogger("affine")

# Shared HTTP client session
_http_client: Optional[aiohttp.ClientSession] = None


async def _get_http_client() -> aiohttp.ClientSession:
    """Get or create shared HTTP client session"""
    global _http_client
    if _http_client is None or _http_client.closed:
        _http_client = aiohttp.ClientSession()
    return _http_client


class R2Dataset:
    """
    R2 dataset with background download and local-only sampling.
    
    Features:
    - Background download of all files to local cache
    - Sampling only from local cache (no real-time downloads)
    - Automatic retry on download failures
    """
    
    def __init__(
        self,
        dataset_name: str,
        seed: Optional[int] = None,
        cache_dir: Optional[str] = None,
        download_concurrency: int = DOWNLOAD_CONCURRENCY,
    ):
        """
        Initialize R2 dataset with background downloading.
        
        Args:
            dataset_name: Name of the dataset (e.g., "satpalsr/rl-python")
            seed: Random seed for reproducibility (optional)
            cache_dir: Directory for local cache (default: /tmp/dataset_cache)
            download_concurrency: Number of concurrent downloads (default: 10)
        """
        self.dataset_name = dataset_name
        self._rng = random.Random(seed)
        self._download_concurrency = download_concurrency
        
        # Cache configuration
        self._cache_dir = Path(cache_dir or CACHE_DIR) / dataset_name.replace("/", "_")
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Build dataset paths
        self._dataset_folder = f"affine/datasets/{dataset_name}/"
        self._index_key = self._dataset_folder + "index.json"
        
        # R2 credentials
        self._endpoint_url = ENDPOINT
        self._access_key = ACCESS
        self._secret_key = SECRET
        self._public_read = PUBLIC_READ
        self._public_base = R2_PUBLIC_BASE
        
        # Dataset metadata
        self._index: Optional[Dict[str, Any]] = None
        self._files: List[Dict[str, Any]] = []
        self.total_size: int = 0
        
        # Cache for file sample counts (used by get_by_id)
        self._file_sample_counts: Optional[List[int]] = None
        self._cumulative_counts: Optional[List[int]] = None
        
        # Start background initialization
        self._init_task = asyncio.create_task(self._async_init())
        logger.info(f"Started background initialization for dataset '{self.dataset_name}'")
    
    def _get_s3_client(self):
        """Create S3 client context for private R2 access"""
        if not self._endpoint_url:
            raise RuntimeError("R2 endpoint is not configured (missing R2_BUCKET_ID)")
        
        session = get_session()
        return session.create_client(
            "s3",
            endpoint_url=self._endpoint_url,
            aws_access_key_id=self._access_key,
            aws_secret_access_key=self._secret_key,
            config=Config(max_pool_connections=256),
        )
    
    async def _async_init(self) -> None:
        """Background initialization: load index and download all files"""
        try:
            await self._load_index()
            await self._download_all_files()
            logger.info(f"Dataset '{self.dataset_name}' initialization complete")
        except Exception as e:
            logger.error(f"Background initialization failed: {e}")
    
    async def _load_index(self) -> None:
        """Load dataset index from R2"""
        if self._index is not None:
            return
        
        try:
            if self._public_read:
                url = f"{self._public_base}/{self._index_key}"
                logger.debug(f"Loading public R2 index: {url}")
                
                client = await _get_http_client()
                async with client.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    resp.raise_for_status()
                    self._index = await resp.json()
            else:
                logger.debug(f"Loading R2 index: s3://{FOLDER}/{self._index_key}")
                
                async with self._get_s3_client() as s3:
                    resp = await s3.get_object(Bucket=FOLDER, Key=self._index_key)
                    body = await resp["Body"].read()
                    self._index = json.loads(body.decode())
            
            # Extract file list and metadata
            self._files = list(self._index.get("files", []))
            self.total_size = int(self._index.get("total_rows", 0))
            
            if not self._files:
                raise RuntimeError(f"Dataset '{self.dataset_name}' contains no files")
            
            logger.info(f"Loaded dataset '{self.dataset_name}': {len(self._files)} files, {self.total_size} total samples")
            
        except Exception as e:
            logger.error(f"Failed to load dataset index: {e}")
            raise RuntimeError(f"Failed to load dataset '{self.dataset_name}': {e}") from e
    
    async def _download_all_files(self) -> None:
        """Background task to download all files to local cache in random order"""
        logger.info(f"Starting download of all {len(self._files)} files for dataset '{self.dataset_name}'")
        
        # Shuffle files for random download order
        shuffled_files = self._files.copy()
        self._rng.shuffle(shuffled_files)
        
        # Download files in batches with concurrency control
        semaphore = asyncio.Semaphore(self._download_concurrency)
        
        async def download_with_semaphore(file_info):
            async with semaphore:
                return await self._download_and_cache_file(file_info)
        
        tasks = [download_with_semaphore(file_info) for file_info in shuffled_files]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Count successful downloads
        success_count = sum(1 for r in results if r and not isinstance(r, Exception))
        
        logger.info(
            f"Download completed: {success_count}/{len(self._files)} files cached for dataset '{self.dataset_name}'"
        )
    
    async def _download_and_cache_file(self, file_info: Dict[str, Any]) -> bool:
        """Download a file and save it to local cache with file locking"""
        key = file_info.get("key") or (self._dataset_folder + file_info.get("filename", ""))
        if not key:
            return False
        
        cache_path = self._get_cache_path(key)
        lock_path = cache_path.with_suffix(".lock")
        
        # Skip if already cached
        if cache_path.exists():
            logger.debug(f"File already cached: {key}")
            return True
        
        # Use file lock to prevent concurrent downloads across processes
        try:
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            lock_file = open(lock_path, 'w')
            
            try:
                # Try to acquire exclusive lock (non-blocking)
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                
                # Double-check file doesn't exist (race condition)
                if cache_path.exists():
                    logger.debug(f"File already cached (after lock): {key}")
                    return True
                
                # Download file
                if self._public_read:
                    url = f"{self._public_base}/{key}"
                    client = await _get_http_client()
                    async with client.get(url, timeout=aiohttp.ClientTimeout(total=120)) as resp:
                        resp.raise_for_status()
                        body = await resp.read()
                else:
                    async with self._get_s3_client() as s3:
                        resp = await s3.get_object(Bucket=FOLDER, Key=key)
                        body = await resp["Body"].read()
                
                # Validate JSON
                data = json.loads(body.decode())
                if not isinstance(data, list) or not data:
                    logger.warning(f"Skipping invalid/empty file: {key}")
                    return False
                
                # Write to cache atomically
                temp_path = cache_path.with_suffix(".tmp")
                temp_path.write_bytes(body)
                temp_path.rename(cache_path)
                
                logger.debug(f"Cached file: {key} ({len(data)} samples)")
                return True
                
            except BlockingIOError:
                # Another process is downloading this file, wait for it
                logger.debug(f"File {key} is being downloaded by another process, waiting...")
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_SH)
                
                # Check if download succeeded
                if cache_path.exists():
                    logger.debug(f"File {key} downloaded by another process")
                    return True
                else:
                    logger.warning(f"File {key} download failed in another process")
                    return False
                    
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                lock_file.close()
                
                # Clean up lock file
                try:
                    lock_path.unlink()
                except:
                    pass
                    
        except Exception as e:
            logger.warning(f"Failed to download file {key}: {type(e).__name__}: {str(e)}")
            return False
    
    def _get_cache_path(self, key: str) -> Path:
        """Get local cache path for a file key"""
        filename = key.split("/")[-1]
        return self._cache_dir / filename
    
    def _get_cached_files(self) -> List[str]:
        """Get list of cached file paths"""
        if not self._cache_dir.exists():
            return []
        return [str(f) for f in self._cache_dir.glob("*.json")]
    
    async def _read_from_cache(self, cache_path: Path) -> Optional[List[Any]]:
        """Read file from local cache"""
        if not cache_path.exists():
            return None
        
        try:
            body = cache_path.read_bytes()
            data = json.loads(body.decode())
            
            if not isinstance(data, list):
                return None
            
            return data
            
        except Exception as e:
            logger.warning(f"Failed to read from cache {cache_path}: {e}")
            return None
    
    async def get(self) -> Any:
        """
        Get a random sample from the local cache.
        
        Returns:
            A randomly selected sample from cached files
            
        Raises:
            RuntimeError: If no cached files available after retries
        """
        # Retry mechanism: check for cached files, sleep 5s and retry up to 3 times
        max_retries = 3
        retry_delay = 5
        
        for attempt in range(max_retries):
            cached_files = self._get_cached_files()
            
            if cached_files:
                break
            
            if attempt < max_retries - 1:
                logger.warning(
                    f"No cached files found for dataset '{self.dataset_name}' "
                    f"(attempt {attempt + 1}/{max_retries}), waiting {retry_delay}s..."
                )
                await asyncio.sleep(retry_delay)
            else:
                raise RuntimeError(
                    f"No cached files available for dataset '{self.dataset_name}' after {max_retries} retries"
                )
        
        # Randomly select a cached file
        selected_file = Path(self._rng.choice(cached_files))
        
        # Load file contents
        samples = await self._read_from_cache(selected_file)
        
        if not samples:
            raise RuntimeError(f"Failed to read cached file: {selected_file}")
        
        # Randomly select a sample from the file
        sample = self._rng.choice(samples)
        
        logger.debug(f"Retrieved random sample from dataset '{self.dataset_name}' (file: {selected_file.name})")
        return sample
    
    def __aiter__(self):
        """Support async iteration"""
        return self
    
    async def __anext__(self) -> Any:
        """Get next random sample"""
        return await self.get()
    
    async def _build_file_index(self) -> None:
        """Build index mapping for task_id to file/sample locations"""
        if self._file_sample_counts is not None:
            return  # Already built
        
        # Wait for index to be loaded
        if self._index is None:
            await self._load_index()
        
        # Get sorted list of files (for deterministic ordering)
        file_keys = sorted([
            file_info.get("key") or (self._dataset_folder + file_info.get("filename", ""))
            for file_info in self._files
        ])
        
        # Load sample counts for each file
        sample_counts = []
        for key in file_keys:
            cache_path = self._get_cache_path(key)
            
            # Check if file is cached
            if cache_path.exists():
                samples = await self._read_from_cache(cache_path)
                count = len(samples) if samples else 0
            else:
                # File not yet downloaded - get count from index metadata if available
                file_info = next(
                    (f for f in self._files if (f.get("key") or (self._dataset_folder + f.get("filename", ""))) == key),
                    None
                )
                count = file_info.get("row_count", 0) if file_info else 0
            
            sample_counts.append(count)
        
        # Build cumulative counts for fast lookup
        cumulative = []
        total = 0
        for count in sample_counts:
            cumulative.append(total)
            total += count
        
        self._file_sample_counts = sample_counts
        self._cumulative_counts = cumulative
        
        logger.debug(f"Built file index: {len(file_keys)} files, {total} total samples")
    
    async def _ensure_file_downloaded(self, file_info: Dict[str, Any], timeout: int = 120) -> bool:
        """
        Ensure a specific file is downloaded and cached.
        
        Args:
            file_info: File metadata from index
            timeout: Maximum wait time in seconds
            
        Returns:
            True if file is available, False otherwise
        """
        key = file_info.get("key") or (self._dataset_folder + file_info.get("filename", ""))
        cache_path = self._get_cache_path(key)
        
        # Check if already cached
        if cache_path.exists():
            return True
        
        logger.info(f"File {key} not cached, triggering download...")
        
        # Trigger download
        start_time = asyncio.get_event_loop().time()
        success = await self._download_and_cache_file(file_info)
        
        if success:
            return True
        
        # If download failed, wait for background download with timeout
        while asyncio.get_event_loop().time() - start_time < timeout:
            if cache_path.exists():
                logger.info(f"File {key} downloaded successfully")
                return True
            await asyncio.sleep(1)
        
        logger.error(f"Timeout waiting for file {key} download")
        return False
    
    async def get_by_id(self, task_id: int, download_timeout: int = 120) -> Any:
        """
        Get a specific sample by task ID (deterministic).
        
        Args:
            task_id: Task identifier (0-based global index across entire dataset)
            download_timeout: Maximum seconds to wait for file download if not cached
            
        Returns:
            Sample at the specified index
            
        Raises:
            ValueError: If task_id is out of range or invalid
            RuntimeError: If required file cannot be downloaded
        """
        # Wait for index to be loaded
        if self._index is None:
            logger.debug("Waiting for dataset index to load...")
            await self._load_index()

        # Wait for background initialization (file downloads) to complete
        if self._init_task is not None:
            logger.debug("Waiting for background dataset initialization to complete...")
            try:
                await asyncio.wait_for(self._init_task, timeout=download_timeout)
                logger.debug("Background initialization completed successfully")
            except asyncio.TimeoutError:
                logger.warning(
                    f"Background initialization timeout after {download_timeout}s. "
                    "Will attempt on-demand download for required file."
                )
            except Exception as e:
                logger.warning(f"Background initialization failed: {e}. Will attempt on-demand download.")

        # Validate task_id range
        if task_id < 0:
            raise ValueError(f"task_id must be non-negative, got {task_id}")
        
        if self.total_size > 0 and task_id >= self.total_size:
            raise ValueError(
                f"task_id {task_id} out of range [0, {self.total_size-1}]. "
                f"Dataset '{self.dataset_name}' contains {self.total_size} samples."
            )
        
        # Build file index if not already done
        await self._build_file_index()
        
        # Get sorted file list (must match _build_file_index ordering)
        file_keys = sorted([
            file_info.get("key") or (self._dataset_folder + file_info.get("filename", ""))
            for file_info in self._files
        ])
        
        # Find target file using cumulative counts
        file_idx = None
        local_idx = task_id
        
        for i, cumulative in enumerate(self._cumulative_counts):
            if i == len(self._cumulative_counts) - 1:
                # Last file
                file_idx = i
                local_idx = task_id - cumulative
                break
            elif task_id < self._cumulative_counts[i + 1]:
                # Found target file
                file_idx = i
                local_idx = task_id - cumulative
                break
        
        if file_idx is None:
            raise RuntimeError(f"Failed to locate task_id {task_id} in file index")
        
        # Validate local index
        if local_idx >= self._file_sample_counts[file_idx]:
            raise ValueError(
                f"task_id {task_id} maps to invalid local index {local_idx} "
                f"in file {file_idx} (size: {self._file_sample_counts[file_idx]})"
            )
        
        # Get file info and ensure it's downloaded
        target_key = file_keys[file_idx]
        file_info = next(
            (f for f in self._files if (f.get("key") or (self._dataset_folder + f.get("filename", ""))) == target_key),
            None
        )
        
        if not file_info:
            raise RuntimeError(f"File metadata not found for key: {target_key}")
        
        # Ensure file is downloaded
        cache_path = self._get_cache_path(target_key)
        if not cache_path.exists():
            logger.info(f"task_id {task_id} requires file {target_key} (not yet cached)")
            download_success = await self._ensure_file_downloaded(file_info, timeout=download_timeout)
            
            if not download_success:
                raise RuntimeError(
                    f"Failed to download required file for task_id {task_id}: {target_key}. "
                    f"File may be unavailable or download timed out after {download_timeout}s."
                )
        
        # Load file and return specific sample
        samples = await self._read_from_cache(cache_path)
        
        if not samples:
            raise RuntimeError(f"Failed to read cached file: {cache_path}")
        
        # Update file index if actual sample count differs (file was just downloaded)
        actual_count = len(samples)
        if actual_count != self._file_sample_counts[file_idx]:
            logger.warning(
                f"File {file_idx} sample count mismatch: "
                f"index={self._file_sample_counts[file_idx]}, actual={actual_count}. "
                f"Rebuilding file index..."
            )
            # Invalidate index to force rebuild
            self._file_sample_counts = None
            self._cumulative_counts = None
            await self._build_file_index()
            
            # Recalculate file_idx and local_idx with updated index
            file_idx = None
            local_idx = task_id
            
            for i, cumulative in enumerate(self._cumulative_counts):
                if i == len(self._cumulative_counts) - 1:
                    file_idx = i
                    local_idx = task_id - cumulative
                    break
                elif task_id < self._cumulative_counts[i + 1]:
                    file_idx = i
                    local_idx = task_id - cumulative
                    break
            
            if file_idx is None:
                raise RuntimeError(f"Failed to locate task_id {task_id} after index rebuild")
            
            # Re-validate with updated index
            target_key = file_keys[file_idx]
            cache_path = self._get_cache_path(target_key)
            samples = await self._read_from_cache(cache_path)
            
            if not samples:
                raise RuntimeError(f"Failed to read cached file after rebuild: {cache_path}")
        
        if local_idx >= len(samples):
            raise RuntimeError(
                f"Sample index {local_idx} out of range for file {target_key} "
                f"(expected {self._file_sample_counts[file_idx]} samples, found {len(samples)})"
            )
        
        logger.debug(
            f"Retrieved task_id {task_id} from file {file_idx} "
            f"(local_idx={local_idx}, file={cache_path.name})"
        )
        
        return samples[local_idx]