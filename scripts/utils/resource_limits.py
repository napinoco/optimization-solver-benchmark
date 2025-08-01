"""
Resource limitation utilities for subprocess execution.

This module provides unified resource limitation capabilities for both Python
and MATLAB solver subprocesses, ensuring consistent memory and CPU constraints
across all solver executions.
"""

import platform
import shlex
from typing import List, Optional


def build_resource_limited_command(cmd: List[str], 
                                 memory_limit_gb: Optional[float] = None) -> List[str]:
    """
    Build a command with resource limitations using ulimit.
    
    On Unix-like systems, this wraps the command with ulimit to enforce
    memory constraints. On Windows, returns the command unchanged as ulimit
    is not available.
    
    Args:
        cmd: Original command as a list of arguments
        memory_limit_gb: Memory limit in gigabytes (None for no limit)
        
    Returns:
        Modified command list with resource limitations
    """
    # Windows doesn't support ulimit
    if platform.system() == 'Windows':
        return cmd
    
    # No limit requested
    if memory_limit_gb is None:
        return cmd
    
    # Convert GB to KB for ulimit -v
    memory_limit_kb = int(memory_limit_gb * 1024 * 1024)
    
    # Properly quote the original command for shell execution
    quoted_cmd = ' '.join(shlex.quote(arg) for arg in cmd)
    
    # Build ulimit command
    # -v: virtual memory limit in KB
    ulimit_cmd = f'ulimit -v {memory_limit_kb}; {quoted_cmd}'
    
    # Return as bash -c command
    return ['bash', '-c', ulimit_cmd]


def format_memory_limit_display(memory_limit_gb: Optional[float]) -> str:
    """
    Format memory limit for display in logs.
    
    Args:
        memory_limit_gb: Memory limit in gigabytes
        
    Returns:
        Formatted string for display
    """
    if memory_limit_gb is None:
        return "unlimited"
    elif memory_limit_gb < 1:
        return f"{int(memory_limit_gb * 1024)}MB"
    else:
        return f"{memory_limit_gb}GB"