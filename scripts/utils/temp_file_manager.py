"""
Temporary File Management System for MATLAB Integration.

This module provides robust temporary file management for Python-MATLAB data exchange,
ensuring unique file names, automatic cleanup, and handling of concurrent execution scenarios.
"""

import os
import sys
import time
import uuid
import tempfile
import logging
import glob
from pathlib import Path
from typing import Optional, List, ContextManager
from contextlib import contextmanager

# Add project root for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.utils.logger import get_logger

logger = get_logger("temp_file_manager")


class TempFileManager:
    """
    Manages temporary files for MATLAB integration with robust cleanup and error handling.
    """
    
    def __init__(self, base_prefix: str = "matlab_result", cleanup_age_hours: int = 1):
        """
        Initialize temporary file manager.
        
        Args:
            base_prefix: Base prefix for temporary file names
            cleanup_age_hours: Age in hours after which orphaned files are cleaned up
        """
        self.base_prefix = base_prefix
        self.cleanup_age_hours = cleanup_age_hours
        self.temp_dir = self._get_temp_directory()
        
        logger.debug(f"TempFileManager initialized with prefix '{base_prefix}', "
                    f"temp dir: {self.temp_dir}")
    
    def _get_temp_directory(self) -> str:
        """
        Get appropriate temporary directory with fallback options.
        
        Returns:
            Path to temporary directory
        """
        # Try multiple temp directory options
        temp_options = [
            tempfile.gettempdir(),
            os.path.join(str(project_root), "temp"),
            "/tmp",
            "."
        ]
        
        for temp_dir in temp_options:
            try:
                # Create directory if it doesn't exist
                os.makedirs(temp_dir, exist_ok=True)
                
                # Test write access
                test_file = os.path.join(temp_dir, f"test_{uuid.uuid4().hex[:8]}.tmp")
                with open(test_file, 'w') as f:
                    f.write("test")
                os.remove(test_file)
                
                logger.debug(f"Using temp directory: {temp_dir}")
                return temp_dir
                
            except (OSError, PermissionError) as e:
                logger.warning(f"Cannot use temp directory {temp_dir}: {e}")
                continue
        
        raise RuntimeError("No writable temporary directory found")
    
    def generate_unique_filename(self, extension: str = ".json") -> str:
        """
        Generate a unique temporary file name.
        
        Args:
            extension: File extension (including dot)
            
        Returns:
            Full path to unique temporary file
        """
        # Generate unique components
        process_id = os.getpid()
        timestamp = int(time.time())
        unique_id = uuid.uuid4().hex[:8]
        
        # Construct filename: prefix_processid_timestamp_uuid.extension
        filename = f"{self.base_prefix}_{process_id}_{timestamp}_{unique_id}{extension}"
        
        return os.path.join(self.temp_dir, filename)
    
    def create_temp_file(self, extension: str = ".json") -> str:
        """
        Create a unique temporary file atomically.
        
        Args:
            extension: File extension (including dot)
            
        Returns:
            Path to created temporary file
        """
        max_attempts = 5
        
        for attempt in range(max_attempts):
            try:
                # Generate unique filename
                temp_file = self.generate_unique_filename(extension)
                
                # Create file atomically using tempfile.mkstemp approach
                fd = os.open(temp_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                os.close(fd)
                
                logger.debug(f"Created temp file: {temp_file}")
                return temp_file
                
            except FileExistsError:
                # Very unlikely with UUID, but retry with new name
                if attempt == max_attempts - 1:
                    raise RuntimeError(f"Failed to create unique temp file after {max_attempts} attempts")
                continue
                
            except OSError as e:
                # Handle disk full, permission errors, etc.
                raise RuntimeError(f"Failed to create temp file: {e}")
    
    def cleanup_file(self, file_path: str) -> bool:
        """
        Safely clean up a temporary file.
        
        Args:
            file_path: Path to file to clean up
            
        Returns:
            True if file was cleaned up successfully, False otherwise
        """
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.debug(f"Cleaned up temp file: {file_path}")
                return True
            else:
                logger.debug(f"Temp file already gone: {file_path}")
                return True
                
        except OSError as e:
            logger.warning(f"Failed to clean up temp file {file_path}: {e}")
            return False
    
    def cleanup_orphaned_files(self) -> int:
        """
        Clean up orphaned temporary files older than cleanup_age_hours.
        
        Returns:
            Number of files cleaned up
        """
        pattern = os.path.join(self.temp_dir, f"{self.base_prefix}_*")
        current_time = time.time()
        cutoff_time = current_time - (self.cleanup_age_hours * 3600)
        
        cleaned_count = 0
        
        try:
            for file_path in glob.glob(pattern):
                try:
                    # Check file age
                    file_mtime = os.path.getmtime(file_path)
                    
                    if file_mtime < cutoff_time:
                        # File is old enough to clean up
                        if self.cleanup_file(file_path):
                            cleaned_count += 1
                            
                except OSError as e:
                    logger.warning(f"Error checking temp file {file_path}: {e}")
                    continue
            
            if cleaned_count > 0:
                logger.info(f"Cleaned up {cleaned_count} orphaned temporary files")
            else:
                logger.debug("No orphaned temporary files found")
                
        except Exception as e:
            logger.error(f"Error during orphaned file cleanup: {e}")
        
        return cleaned_count
    
    @contextmanager
    def temp_file_context(self, extension: str = ".json") -> ContextManager[str]:
        """
        Context manager for temporary file with automatic cleanup.
        
        Args:
            extension: File extension (including dot)
            
        Yields:
            Path to temporary file
        """
        temp_file = None
        try:
            temp_file = self.create_temp_file(extension)
            yield temp_file
        finally:
            if temp_file:
                self.cleanup_file(temp_file)
    
    def get_temp_file_stats(self) -> dict:
        """
        Get statistics about temporary files in temp directory.
        
        Returns:
            Dictionary with temp file statistics
        """
        pattern = os.path.join(self.temp_dir, f"{self.base_prefix}_*")
        
        files = glob.glob(pattern)
        total_count = len(files)
        total_size = 0
        oldest_file = None
        oldest_time = time.time()
        
        for file_path in files:
            try:
                stat = os.stat(file_path)
                total_size += stat.st_size
                
                if stat.st_mtime < oldest_time:
                    oldest_time = stat.st_mtime
                    oldest_file = file_path
                    
            except OSError:
                continue
        
        stats = {
            'total_files': total_count,
            'total_size_bytes': total_size,
            'temp_directory': self.temp_dir,
            'oldest_file': oldest_file,
            'oldest_age_hours': (time.time() - oldest_time) / 3600 if oldest_file else 0
        }
        
        return stats


# Global instance for convenience
default_temp_manager = TempFileManager()


def generate_temp_file(extension: str = ".json") -> str:
    """
    Convenience function to generate unique temp file using default manager.
    
    Args:
        extension: File extension (including dot)
        
    Returns:
        Path to unique temporary file
    """
    return default_temp_manager.generate_unique_filename(extension)


def create_temp_file(extension: str = ".json") -> str:
    """
    Convenience function to create temp file using default manager.
    
    Args:
        extension: File extension (including dot)
        
    Returns:
        Path to created temporary file
    """
    return default_temp_manager.create_temp_file(extension)


def cleanup_temp_file(file_path: str) -> bool:
    """
    Convenience function to cleanup temp file using default manager.
    
    Args:
        file_path: Path to file to clean up
        
    Returns:
        True if cleaned up successfully
    """
    return default_temp_manager.cleanup_file(file_path)


def cleanup_orphaned_files() -> int:
    """
    Convenience function to cleanup orphaned files using default manager.
    
    Returns:
        Number of files cleaned up
    """
    return default_temp_manager.cleanup_orphaned_files()


@contextmanager
def temp_file_context(extension: str = ".json") -> ContextManager[str]:
    """
    Convenience context manager for temporary file with automatic cleanup.
    
    Args:
        extension: File extension (including dot)
        
    Yields:
        Path to temporary file
    """
    with default_temp_manager.temp_file_context(extension) as temp_file:
        yield temp_file