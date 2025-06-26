"""
Git utilities for tracking commit hashes and repository information.

This module provides functions to detect Git commit hashes and repository
state for reproducibility tracking.
"""

import subprocess
import sys
from pathlib import Path
from typing import Optional, Dict, Any

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.utils.logger import get_logger

logger = get_logger("git_utils")

# Global cache to avoid repeated git operations
_git_info_cache = None


def get_git_commit_hash() -> Optional[str]:
    """
    Get the current Git commit hash with caching.
    
    Returns:
        Git commit hash or None if not available
    """
    global _git_info_cache
    
    if _git_info_cache is not None:
        return _git_info_cache.get('commit_hash')
    
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=project_root
        )
        if result.returncode == 0:
            commit_hash = result.stdout.strip()
            logger.debug(f"Detected Git commit hash: {commit_hash}")
            
            # Initialize cache if not exists
            if _git_info_cache is None:
                _git_info_cache = {}
            _git_info_cache['commit_hash'] = commit_hash
            
            return commit_hash
        else:
            logger.warning(f"Git command failed: {result.stderr.strip()}")
            return None
    except subprocess.TimeoutExpired:
        logger.warning("Git command timed out")
        return None
    except FileNotFoundError:
        logger.debug("Git not available")
        return None
    except Exception as e:
        logger.error(f"Error getting Git commit hash: {e}")
        return None


def get_git_branch() -> Optional[str]:
    """
    Get the current Git branch name.
    
    Returns:
        Git branch name or None if not available
    """
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=project_root
        )
        if result.returncode == 0:
            branch = result.stdout.strip()
            logger.debug(f"Detected Git branch: {branch}")
            return branch
        else:
            logger.warning(f"Git branch command failed: {result.stderr.strip()}")
            return None
    except subprocess.TimeoutExpired:
        logger.warning("Git branch command timed out")
        return None
    except FileNotFoundError:
        logger.debug("Git not available")
        return None
    except Exception as e:
        logger.error(f"Error getting Git branch: {e}")
        return None


def is_git_repo_dirty() -> Optional[bool]:
    """
    Check if the Git repository has uncommitted changes.
    
    Returns:
        True if repo is dirty, False if clean, None if Git not available
    """
    try:
        result = subprocess.run(
            ['git', 'status', '--porcelain'],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=project_root
        )
        if result.returncode == 0:
            is_dirty = bool(result.stdout.strip())
            logger.debug(f"Git repository dirty: {is_dirty}")
            return is_dirty
        else:
            logger.warning(f"Git status command failed: {result.stderr.strip()}")
            return None
    except subprocess.TimeoutExpired:
        logger.warning("Git status command timed out")
        return None
    except FileNotFoundError:
        logger.debug("Git not available")
        return None
    except Exception as e:
        logger.error(f"Error checking Git repository status: {e}")
        return None


def get_git_info() -> Dict[str, Any]:
    """
    Get comprehensive Git repository information.
    
    Returns:
        Dictionary with Git information
    """
    info = {
        'commit_hash': get_git_commit_hash(),
        'branch': get_git_branch(),
        'is_dirty': is_git_repo_dirty(),
        'available': None
    }
    
    # Determine if Git is available based on any successful command
    info['available'] = any(value is not None for value in info.values() if value != info['available'])
    
    return info


def format_git_commit_for_display(commit_hash: Optional[str]) -> str:
    """
    Format Git commit hash for display purposes.
    
    Args:
        commit_hash: Full commit hash or None
        
    Returns:
        Formatted commit hash (shortened) or 'unknown'
    """
    if commit_hash:
        # Return first 8 characters for display
        return commit_hash[:8]
    else:
        return 'unknown'


def validate_git_commit_hash(commit_hash: str) -> bool:
    """
    Validate that a string looks like a Git commit hash.
    
    Args:
        commit_hash: String to validate
        
    Returns:
        True if it looks like a valid commit hash
    """
    if not commit_hash:
        return False
    
    # Git commit hashes are 40 character hex strings
    if len(commit_hash) == 40:
        try:
            int(commit_hash, 16)
            return True
        except ValueError:
            return False
    
    return False


if __name__ == "__main__":
    # Test Git utilities
    print("=== Git Information Test ===")
    
    git_info = get_git_info()
    print(f"Git Available: {git_info['available']}")
    print(f"Commit Hash: {git_info['commit_hash']}")
    print(f"Branch: {git_info['branch']}")
    print(f"Repository Dirty: {git_info['is_dirty']}")
    
    if git_info['commit_hash']:
        print(f"Formatted Hash: {format_git_commit_for_display(git_info['commit_hash'])}")
        print(f"Hash Valid: {validate_git_commit_hash(git_info['commit_hash'])}")
    
    print(f"\nComplete Git Info: {git_info}")