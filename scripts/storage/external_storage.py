"""
External Problem Storage System
===============================

Handles downloading, caching, and management of optimization problems from external sources.
Supports GitHub releases, direct URLs, and S3-compatible storage with integrity verification.
"""

import os
import hashlib
import urllib.request
import urllib.error
import gzip
import zipfile
import tarfile
import tempfile
import shutil
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import sys

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.utils.logger import get_logger

logger = get_logger("external_storage")


@dataclass
class ExternalProblem:
    """External problem metadata and location."""
    name: str
    url: str
    size: int = 0
    checksum: Optional[str] = None
    checksum_type: str = "sha256"
    compressed: bool = False
    compression_type: Optional[str] = None
    cache_ttl_hours: int = 24 * 7  # 1 week default
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class CacheEntry:
    """Cache entry metadata."""
    problem_name: str
    local_path: str
    url: str
    downloaded_at: datetime
    checksum: str
    size: int
    ttl_hours: int
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        expiry_time = self.downloaded_at + timedelta(hours=self.ttl_hours)
        return datetime.now() > expiry_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "problem_name": self.problem_name,
            "local_path": self.local_path,
            "url": self.url,
            "downloaded_at": self.downloaded_at.isoformat(),
            "checksum": self.checksum,
            "size": self.size,
            "ttl_hours": self.ttl_hours
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CacheEntry':
        """Create from dictionary."""
        return cls(
            problem_name=data["problem_name"],
            local_path=data["local_path"],
            url=data["url"],
            downloaded_at=datetime.fromisoformat(data["downloaded_at"]),
            checksum=data["checksum"],
            size=data["size"],
            ttl_hours=data["ttl_hours"]
        )


class ExternalProblemStorage:
    """Manages external problem storage and caching."""
    
    def __init__(self, cache_dir: Optional[str] = None, max_cache_size_gb: float = 5.0):
        """
        Initialize external problem storage.
        
        Args:
            cache_dir: Directory for local cache (default: ~/.optimization_benchmark_cache)
            max_cache_size_gb: Maximum cache size in GB
        """
        self.logger = get_logger("external_storage")
        
        # Set up cache directory
        if cache_dir is None:
            cache_dir = Path.home() / ".optimization_benchmark_cache"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache settings
        self.max_cache_size_bytes = int(max_cache_size_gb * 1024 * 1024 * 1024)
        self.cache_metadata_file = self.cache_dir / "cache_metadata.json"
        
        # Load existing cache metadata
        self.cache_entries: Dict[str, CacheEntry] = {}
        self._load_cache_metadata()
        
        self.logger.info(f"Initialized external storage with cache dir: {self.cache_dir}")
    
    def _load_cache_metadata(self):
        """Load cache metadata from disk."""
        if self.cache_metadata_file.exists():
            try:
                with open(self.cache_metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                for entry_data in metadata.get("entries", []):
                    entry = CacheEntry.from_dict(entry_data)
                    self.cache_entries[entry.problem_name] = entry
                    
                self.logger.info(f"Loaded {len(self.cache_entries)} cache entries")
                
            except Exception as e:
                self.logger.warning(f"Failed to load cache metadata: {e}")
    
    def _save_cache_metadata(self):
        """Save cache metadata to disk."""
        try:
            metadata = {
                "version": "1.0",
                "updated_at": datetime.now().isoformat(),
                "entries": [entry.to_dict() for entry in self.cache_entries.values()]
            }
            
            with open(self.cache_metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save cache metadata: {e}")
    
    def _calculate_checksum(self, file_path: str, algorithm: str = "sha256") -> str:
        """Calculate file checksum."""
        hash_func = hashlib.new(algorithm)
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_func.update(chunk)
        
        return hash_func.hexdigest()
    
    def _get_cache_size(self) -> int:
        """Get total cache size in bytes."""
        total_size = 0
        for entry in self.cache_entries.values():
            if Path(entry.local_path).exists():
                total_size += entry.size
        return total_size
    
    def _cleanup_cache(self):
        """Clean up expired and oversized cache entries."""
        # Remove expired entries
        expired_entries = [
            name for name, entry in self.cache_entries.items()
            if entry.is_expired() or not Path(entry.local_path).exists()
        ]
        
        for name in expired_entries:
            entry = self.cache_entries[name]
            if Path(entry.local_path).exists():
                try:
                    os.remove(entry.local_path)
                    self.logger.info(f"Removed expired cache entry: {name}")
                except OSError as e:
                    self.logger.warning(f"Failed to remove expired cache file {entry.local_path}: {e}")
            
            del self.cache_entries[name]
        
        # Check cache size and remove oldest entries if needed
        current_size = self._get_cache_size()
        if current_size > self.max_cache_size_bytes:
            self.logger.info(f"Cache size ({current_size / 1024 / 1024:.1f} MB) exceeds limit, cleaning up...")
            
            # Sort by download time (oldest first)
            sorted_entries = sorted(
                self.cache_entries.items(),
                key=lambda x: x[1].downloaded_at
            )
            
            # Remove oldest entries until under limit
            for name, entry in sorted_entries:
                if current_size <= self.max_cache_size_bytes:
                    break
                
                if Path(entry.local_path).exists():
                    try:
                        os.remove(entry.local_path)
                        current_size -= entry.size
                        self.logger.info(f"Removed cache entry to free space: {name}")
                    except OSError as e:
                        self.logger.warning(f"Failed to remove cache file {entry.local_path}: {e}")
                
                del self.cache_entries[name]
        
        # Save updated metadata
        self._save_cache_metadata()
    
    def _download_file(self, url: str, local_path: str, expected_checksum: Optional[str] = None) -> bool:
        """
        Download file from URL with progress and integrity verification.
        
        Args:
            url: URL to download from
            local_path: Local path to save file
            expected_checksum: Expected SHA256 checksum for verification
            
        Returns:
            True if download succeeded and verification passed, False otherwise
        """
        try:
            self.logger.info(f"Downloading {url} to {local_path}")
            
            # Create directory if needed
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Download with progress (simplified for now)
            with urllib.request.urlopen(url) as response:
                total_size = int(response.headers.get('Content-Length', 0))
                downloaded = 0
                
                with open(local_path, 'wb') as f:
                    while True:
                        chunk = response.read(8192)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # Simple progress logging
                        if total_size > 0 and downloaded % (1024 * 1024) == 0:  # Every MB
                            progress = (downloaded / total_size) * 100
                            self.logger.debug(f"Download progress: {progress:.1f}%")
            
            # Verify checksum if provided
            if expected_checksum:
                actual_checksum = self._calculate_checksum(local_path)
                if actual_checksum != expected_checksum:
                    self.logger.error(f"Checksum mismatch for {url}: expected {expected_checksum}, got {actual_checksum}")
                    os.remove(local_path)
                    return False
                
                self.logger.info(f"Checksum verification passed for {url}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to download {url}: {e}")
            if os.path.exists(local_path):
                os.remove(local_path)
            return False
    
    def _extract_compressed_file(self, compressed_path: str, extract_dir: str, compression_type: str) -> bool:
        """Extract compressed file."""
        try:
            self.logger.info(f"Extracting {compressed_path} ({compression_type}) to {extract_dir}")
            
            Path(extract_dir).mkdir(parents=True, exist_ok=True)
            
            if compression_type.lower() in ['gzip', 'gz']:
                with gzip.open(compressed_path, 'rb') as f_in:
                    output_file = Path(extract_dir) / Path(compressed_path).stem
                    with open(output_file, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                        
            elif compression_type.lower() == 'zip':
                with zipfile.ZipFile(compressed_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
                    
            elif compression_type.lower() in ['tar', 'tar.gz', 'tgz']:
                with tarfile.open(compressed_path, 'r:*') as tar_ref:
                    tar_ref.extractall(extract_dir)
                    
            else:
                self.logger.error(f"Unsupported compression type: {compression_type}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to extract {compressed_path}: {e}")
            return False
    
    def get_problem(self, external_problem: ExternalProblem, force_download: bool = False) -> Optional[str]:
        """
        Get problem file, downloading and caching if necessary.
        
        Args:
            external_problem: External problem definition
            force_download: Force re-download even if cached
            
        Returns:
            Local file path if successful, None otherwise
        """
        problem_name = external_problem.name
        
        # Check cache first (unless forced)
        if not force_download and problem_name in self.cache_entries:
            entry = self.cache_entries[problem_name]
            
            if not entry.is_expired() and Path(entry.local_path).exists():
                self.logger.info(f"Using cached problem: {problem_name}")
                return entry.local_path
            else:
                # Remove expired entry
                if Path(entry.local_path).exists():
                    os.remove(entry.local_path)
                del self.cache_entries[problem_name]
        
        # Clean up cache before downloading
        self._cleanup_cache()
        
        # Determine local path
        url_hash = hashlib.sha256(external_problem.url.encode()).hexdigest()[:8]
        filename = f"{problem_name}_{url_hash}"
        
        if external_problem.compressed:
            # For compressed files, download to temp location first
            temp_path = self.cache_dir / f"{filename}_compressed"
            final_path = self.cache_dir / filename
        else:
            temp_path = final_path = self.cache_dir / filename
        
        # Download file
        if not self._download_file(external_problem.url, str(temp_path), external_problem.checksum):
            return None
        
        # Extract if compressed
        if external_problem.compressed:
            extract_dir = self.cache_dir / f"{filename}_extracted"
            if not self._extract_compressed_file(str(temp_path), str(extract_dir), 
                                               external_problem.compression_type or "auto"):
                os.remove(temp_path)
                return None
            
            # Find the main problem file in extracted directory
            extracted_files = list(Path(extract_dir).glob("**/*"))
            problem_files = [f for f in extracted_files if f.is_file() and f.suffix in ['.mps', '.qps', '.py']]
            
            if not problem_files:
                self.logger.error(f"No problem files found in extracted archive for {problem_name}")
                shutil.rmtree(extract_dir)
                os.remove(temp_path)
                return None
            
            # Use the first problem file (or implement better selection logic)
            shutil.copy2(problem_files[0], final_path)
            
            # Clean up
            shutil.rmtree(extract_dir)
            os.remove(temp_path)
        
        # Calculate final checksum and size
        final_checksum = self._calculate_checksum(str(final_path))
        final_size = os.path.getsize(final_path)
        
        # Create cache entry
        cache_entry = CacheEntry(
            problem_name=problem_name,
            local_path=str(final_path),
            url=external_problem.url,
            downloaded_at=datetime.now(),
            checksum=final_checksum,
            size=final_size,
            ttl_hours=external_problem.cache_ttl_hours
        )
        
        self.cache_entries[problem_name] = cache_entry
        self._save_cache_metadata()
        
        self.logger.info(f"Successfully cached problem: {problem_name} ({final_size / 1024:.1f} KB)")
        return str(final_path)
    
    def list_cached_problems(self) -> List[CacheEntry]:
        """List all cached problems."""
        return list(self.cache_entries.values())
    
    def clear_cache(self, problem_name: Optional[str] = None):
        """Clear cache entries."""
        if problem_name:
            # Clear specific problem
            if problem_name in self.cache_entries:
                entry = self.cache_entries[problem_name]
                if Path(entry.local_path).exists():
                    os.remove(entry.local_path)
                del self.cache_entries[problem_name]
                self.logger.info(f"Cleared cache for: {problem_name}")
        else:
            # Clear all cache
            for entry in self.cache_entries.values():
                if Path(entry.local_path).exists():
                    os.remove(entry.local_path)
            self.cache_entries.clear()
            self.logger.info("Cleared all cache entries")
        
        self._save_cache_metadata()
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache information."""
        total_size = self._get_cache_size()
        
        return {
            "cache_dir": str(self.cache_dir),
            "total_entries": len(self.cache_entries),
            "total_size_mb": total_size / 1024 / 1024,
            "max_size_mb": self.max_cache_size_bytes / 1024 / 1024,
            "utilization_percent": (total_size / self.max_cache_size_bytes) * 100,
            "entries": [
                {
                    "name": entry.problem_name,
                    "size_kb": entry.size / 1024,
                    "downloaded_at": entry.downloaded_at.isoformat(),
                    "expires_at": (entry.downloaded_at + timedelta(hours=entry.ttl_hours)).isoformat(),
                    "is_expired": entry.is_expired()
                }
                for entry in self.cache_entries.values()
            ]
        }


def load_external_problem_registry(registry_file: str) -> Dict[str, ExternalProblem]:
    """
    Load external problem registry from YAML file.
    
    Args:
        registry_file: Path to YAML registry file
        
    Returns:
        Dictionary mapping problem names to ExternalProblem objects
    """
    try:
        with open(registry_file, 'r') as f:
            registry_data = yaml.safe_load(f)
        
        problems = {}
        for problem_data in registry_data.get("problems", []):
            problem = ExternalProblem(
                name=problem_data["name"],
                url=problem_data["url"],
                size=problem_data.get("size", 0),
                checksum=problem_data.get("checksum"),
                checksum_type=problem_data.get("checksum_type", "sha256"),
                compressed=problem_data.get("compressed", False),
                compression_type=problem_data.get("compression_type"),
                cache_ttl_hours=problem_data.get("cache_ttl_hours", 24 * 7),
                metadata=problem_data.get("metadata", {})
            )
            problems[problem.name] = problem
        
        logger.info(f"Loaded {len(problems)} external problems from {registry_file}")
        return problems
        
    except Exception as e:
        logger.error(f"Failed to load external problem registry {registry_file}: {e}")
        return {}


if __name__ == "__main__":
    # Test the external storage system
    print("Testing External Problem Storage System...")
    
    # Create storage instance
    storage = ExternalProblemStorage()
    
    # Test cache info
    cache_info = storage.get_cache_info()
    print(f"Cache directory: {cache_info['cache_dir']}")
    print(f"Cache entries: {cache_info['total_entries']}")
    print(f"Cache size: {cache_info['total_size_mb']:.2f} MB")
    
    # Test with a sample external problem (mock)
    sample_problem = ExternalProblem(
        name="test_problem",
        url="https://httpbin.org/json",  # Test URL that returns JSON
        checksum=None,  # Skip checksum for test
        cache_ttl_hours=1
    )
    
    print(f"\nTesting download with sample problem...")
    try:
        local_path = storage.get_problem(sample_problem)
        if local_path:
            print(f"✓ Successfully downloaded and cached: {local_path}")
            print(f"File size: {os.path.getsize(local_path)} bytes")
        else:
            print("✗ Download failed")
    except Exception as e:
        print(f"✗ Download error: {e}")
    
    # Show updated cache info
    cache_info = storage.get_cache_info()
    print(f"\nUpdated cache info:")
    print(f"Cache entries: {cache_info['total_entries']}")
    print(f"Cache size: {cache_info['total_size_mb']:.2f} MB")
    
    print("\n✓ External storage system test completed!")