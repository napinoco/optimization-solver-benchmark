import platform
import psutil
import sys
import subprocess
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any
import json

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.utils.logger import get_logger
from scripts.utils.git_utils import get_git_info

logger = get_logger("environment_info")

# Global cache to avoid repeated expensive environment collection
_environment_cache = None

def get_os_info() -> Dict[str, str]:
    """Get operating system information with enhanced Ubuntu detection."""
    os_info = {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "architecture": platform.architecture()[0],
        "platform": platform.platform()
    }
    
    # Enhanced Ubuntu version detection
    if platform.system() == "Linux":
        try:
            # Try to get Ubuntu version from /etc/os-release
            if os.path.exists("/etc/os-release"):
                with open("/etc/os-release", "r") as f:
                    os_release = f.read()
                    for line in os_release.split('\n'):
                        if line.startswith('PRETTY_NAME='):
                            os_info["ubuntu_version"] = line.split('=')[1].strip('"')
                            break
                        elif line.startswith('VERSION='):
                            os_info["version_number"] = line.split('=')[1].strip('"')
                        elif line.startswith('VERSION_ID='):
                            os_info["version_id"] = line.split('=')[1].strip('"')
            
            # Try lsb_release as fallback
            try:
                result = subprocess.run(['lsb_release', '-d'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    os_info["lsb_description"] = result.stdout.strip().split('\t')[1]
            except (subprocess.TimeoutExpired, FileNotFoundError, IndexError):
                pass
                
        except Exception as e:
            logger.debug(f"Could not get detailed Ubuntu version: {e}")
    
    return os_info

def get_cpu_info() -> Dict[str, Any]:
    """Get CPU information."""
    return {
        "processor": platform.processor(),
        "cpu_count": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "cpu_freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
    }

def get_memory_info() -> Dict[str, Any]:
    """Get memory information."""
    memory = psutil.virtual_memory()
    result = {
        "total": memory.total,
        "available": memory.available,
        "percent": memory.percent,
        "used": memory.used,
        "free": memory.free,
        "total_gb": round(memory.total / (1024**3), 2),
        "available_gb": round(memory.available / (1024**3), 2)
    }
    
    # In containers, use cgroup memory limit if available and more accurate
    if os.path.exists('/.dockerenv'):
        container_info = get_container_info()
        if container_info.get('memory_limit'):
            cgroup_total = container_info['memory_limit']
            # Use cgroup limit as total (container memory limit is authoritative)
            if cgroup_total != memory.total:
                result["total"] = cgroup_total
                result["total_gb"] = round(cgroup_total / (1024**3), 2)
                # Recalculate available memory proportionally
                proportion_available = memory.available / memory.total
                result["available"] = int(cgroup_total * proportion_available)
                result["available_gb"] = round(result["available"] / (1024**3), 2)
                # Recalculate other values
                result["used"] = cgroup_total - result["available"]
                result["free"] = result["available"]
                result["percent"] = round((result["used"] / cgroup_total) * 100, 2)
                logger.debug(f"Using container memory limit: {result['total_gb']}GB (from cgroup)")
    
    return result

def get_python_info() -> Dict[str, str]:
    """Get Python environment information."""
    return {
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "executable": sys.executable,
        "version_info": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    }

def get_disk_info() -> Dict[str, Any]:
    """Get disk usage information."""
    disk_usage = psutil.disk_usage('/')
    return {
        "total": disk_usage.total,
        "used": disk_usage.used,
        "free": disk_usage.free,
        "percent": round((disk_usage.used / disk_usage.total) * 100, 2),
        "total_gb": round(disk_usage.total / (1024**3), 2),
        "free_gb": round(disk_usage.free / (1024**3), 2)
    }

def get_container_info() -> Dict[str, Any]:
    """Get container environment information."""
    container_info = {
        'is_container': False,
        'container_type': None,
        'memory_limit': None,
        'memory_limit_gb': None,
        'cpu_limit': None,
        'container_id': None,
        'container_name': None
    }
    
    try:
        # Check for Docker container
        if os.path.exists('/.dockerenv'):
            container_info['is_container'] = True
            container_info['container_type'] = 'docker'
            logger.debug("Docker container detected via /.dockerenv")
            
            # Try to get container ID from cgroup
            try:
                with open('/proc/self/cgroup', 'r') as f:
                    cgroup_content = f.read()
                    # Look for Docker container ID pattern
                    import re
                    docker_id_match = re.search(r'/docker/([a-f0-9]{64})', cgroup_content)
                    if docker_id_match:
                        container_info['container_id'] = docker_id_match.group(1)[:12]  # Short ID
                        logger.debug(f"Container ID: {container_info['container_id']}")
            except (IOError, OSError):
                logger.debug("Could not read container ID from cgroup")
            
            # Get memory limit (cgroup v1 and v2)
            memory_limit_files = [
                '/sys/fs/cgroup/memory/memory.limit_in_bytes',  # cgroup v1
                '/sys/fs/cgroup/memory.max'  # cgroup v2
            ]
            
            for limit_file in memory_limit_files:
                try:
                    if os.path.exists(limit_file):
                        with open(limit_file, 'r') as f:
                            limit = int(f.read().strip())
                            # Check if it's a real limit (not the huge default value)
                            if limit < 9223372036854775807:  # Max int64 value
                                container_info['memory_limit'] = limit
                                container_info['memory_limit_gb'] = round(limit / (1024**3), 2)
                                logger.debug(f"Memory limit detected: {container_info['memory_limit_gb']}GB")
                                break
                except (IOError, OSError, ValueError) as e:
                    logger.debug(f"Could not read memory limit from {limit_file}: {e}")
            
            # Get CPU limit information with comprehensive cgroup v1/v2 support
            def _detect_cpu_limit():
                """Comprehensive CPU limit detection for various cgroup configurations."""
                import glob
                
                # cgroup v1 patterns
                cgroup_v1_patterns = [
                    '/sys/fs/cgroup/cpu/cpu.cfs_quota_us',
                    '/sys/fs/cgroup/cpu/docker/*/cpu.cfs_quota_us',
                    '/sys/fs/cgroup/cpu/system.slice/docker-*.scope/cpu.cfs_quota_us'
                ]
                
                # cgroup v2 patterns  
                cgroup_v2_patterns = [
                    '/sys/fs/cgroup/cpu.max',
                    '/sys/fs/cgroup/system.slice/docker-*.scope/cpu.max',
                    '/sys/fs/cgroup/user.slice/*/docker-*.scope/cpu.max'
                ]
                
                # Try cgroup v1 detection
                for pattern in cgroup_v1_patterns:
                    quota_files = glob.glob(pattern) if '*' in pattern else ([pattern] if os.path.exists(pattern) else [])
                    for quota_file in quota_files:
                        try:
                            with open(quota_file, 'r') as f:
                                quota_content = f.read().strip()
                                if quota_content != '-1' and quota_content.isdigit():
                                    # Get corresponding period file
                                    period_file = quota_file.replace('cpu.cfs_quota_us', 'cpu.cfs_period_us')
                                    if os.path.exists(period_file):
                                        with open(period_file, 'r') as pf:
                                            period_content = pf.read().strip()
                                            if period_content.isdigit():
                                                quota = int(quota_content)
                                                period = int(period_content)
                                                if quota > 0 and period > 0:
                                                    cpu_limit = quota / period
                                                    logger.debug(f"CPU limit detected (cgroup v1): {cpu_limit} cores from {quota_file}")
                                                    return round(cpu_limit, 2)
                        except (IOError, OSError, ValueError) as e:
                            logger.debug(f"Could not read cgroup v1 CPU limit from {quota_file}: {e}")
                
                # Try cgroup v2 detection
                for pattern in cgroup_v2_patterns:
                    cpu_max_files = glob.glob(pattern) if '*' in pattern else ([pattern] if os.path.exists(pattern) else [])
                    for cpu_max_file in cpu_max_files:
                        try:
                            with open(cpu_max_file, 'r') as f:
                                cpu_max_content = f.read().strip()
                                if cpu_max_content != 'max' and ' ' in cpu_max_content:
                                    # Format: "quota period" (e.g., "400000 100000")
                                    parts = cpu_max_content.split()
                                    if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                                        quota = int(parts[0])
                                        period = int(parts[1])
                                        if quota > 0 and period > 0:
                                            cpu_limit = quota / period
                                            logger.debug(f"CPU limit detected (cgroup v2): {cpu_limit} cores from {cpu_max_file}")
                                            return round(cpu_limit, 2)
                        except (IOError, OSError, ValueError) as e:
                            logger.debug(f"Could not read cgroup v2 CPU limit from {cpu_max_file}: {e}")
                
                return None
            
            container_info['cpu_limit'] = _detect_cpu_limit()
            
            # Try to get container hostname (often the container name or ID)
            try:
                container_info['container_name'] = platform.node()
                logger.debug(f"Container hostname: {container_info['container_name']}")
            except Exception:
                pass
        
        # Check for other container types (Kubernetes, LXC, etc.)
        elif os.path.exists('/proc/1/cgroup'):
            try:
                with open('/proc/1/cgroup', 'r') as f:
                    cgroup_content = f.read()
                    if 'kubepods' in cgroup_content:
                        container_info['is_container'] = True
                        container_info['container_type'] = 'kubernetes'
                        logger.debug("Kubernetes pod detected")
                    elif 'lxc' in cgroup_content:
                        container_info['is_container'] = True
                        container_info['container_type'] = 'lxc'
                        logger.debug("LXC container detected")
            except (IOError, OSError):
                pass
        
        # Additional container detection methods
        if not container_info['is_container']:
            # Check environment variables that indicate containerization
            container_env_vars = [
                'CONTAINER_ENV',
                'DOCKER_CONTAINER',
                'KUBERNETES_SERVICE_HOST'
            ]
            
            for env_var in container_env_vars:
                if os.environ.get(env_var):
                    container_info['is_container'] = True
                    container_info['container_type'] = 'detected_via_env'
                    logger.debug(f"Container detected via environment variable: {env_var}")
                    break
    
    except Exception as e:
        logger.debug(f"Error detecting container information: {e}")
    
    return container_info

def get_timezone_info() -> Dict[str, Any]:
    """Get timezone and time information."""
    
    timezone_info = {
        "current_time_utc": datetime.now(timezone.utc).isoformat(),
        "current_time_local": datetime.now().isoformat(),
        "utc_offset_seconds": time.timezone if time.daylight == 0 else time.altzone,
        "utc_offset_hours": -(time.timezone if time.daylight == 0 else time.altzone) / 3600,
        "timezone_name": time.tzname[0] if time.daylight == 0 else time.tzname[1],
        "daylight_saving": bool(time.daylight and time.localtime().tm_isdst)
    }
    
    # Try to get more detailed timezone info
    try:
        # Try to get timezone from TZ environment variable
        tz_env = os.environ.get('TZ')
        if tz_env:
            timezone_info["tz_environment"] = tz_env
            
        # Try to read /etc/timezone (Linux/Ubuntu)
        if os.path.exists('/etc/timezone'):
            with open('/etc/timezone', 'r') as f:
                system_timezone = f.read().strip()
                timezone_info["system_timezone"] = system_timezone
                
        # Try to read timezone from timedatectl (systemd systems)
        try:
            result = subprocess.run(['timedatectl', 'show', '--property=Timezone', '--value'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                timezone_info["timedatectl_timezone"] = result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
            
        # For macOS, try to get timezone from systemsetup
        if platform.system() == "Darwin":
            try:
                result = subprocess.run(['systemsetup', '-gettimezone'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    tz_line = result.stdout.strip()
                    if "Time Zone:" in tz_line:
                        timezone_info["macos_timezone"] = tz_line.split("Time Zone:")[1].strip()
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass
    
    except Exception as e:
        logger.debug(f"Could not get detailed timezone info: {e}")
    
    return timezone_info

def collect_environment_info() -> Dict[str, Any]:
    """Collect sanitized environment information for benchmark reproducibility with caching."""
    global _environment_cache
    
    if _environment_cache is not None:
        logger.debug("Using cached environment information")
        return _environment_cache
    
    logger.info("Collecting environment information...")
    
    # Collect full environment info first
    full_env_info = {
        "timestamp": psutil.boot_time(),  # System boot time as reference
        "os": get_os_info(),
        "cpu": get_cpu_info(), 
        "memory": get_memory_info(),
        "python": get_python_info(),
        "disk": get_disk_info(),
        "timezone": get_timezone_info(),
        "container": get_container_info(),  # Add container information
        "git": get_git_info()  # Add Git repository information
    }
    
    # Apply sanitization to remove sensitive information
    env_info = _sanitize_environment_info(full_env_info)
    
    _environment_cache = env_info
    logger.info("Environment information collected, sanitized, and cached successfully")
    logger.debug(f"Environment details: {json.dumps(env_info, indent=2, default=str)}")
    
    return env_info


def _sanitize_environment_info(env_info: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize environment info to remove sensitive information at collection time."""
    
    # Create sanitized copy with minimal information for privacy protection
    sanitized = {}
    
    # CPU info - keep essential performance info only
    if 'cpu' in env_info:
        cpu = env_info['cpu']
        sanitized['cpu'] = {
            'cpu_count': cpu.get('cpu_count'),
            'cpu_count_physical': cpu.get('cpu_count_physical'),
            'processor': cpu.get('processor'),
            'architecture': cpu.get('architecture') or env_info.get('os', {}).get('architecture')
        }
    
    # Memory info - keep total only (performance relevant)
    if 'memory' in env_info:
        memory = env_info['memory']
        sanitized['memory'] = {
            'total_gb': memory.get('total_gb')
        }
    
    # OS info - keep basic system info only (no version details that could identify specific systems)
    if 'os' in env_info:
        os_info = env_info['os']
        sanitized['os'] = {
            'system': os_info.get('system'),      # Darwin, Linux, Windows
            'machine': os_info.get('machine'),    # arm64, x86_64
            'release': os_info.get('release')     # Keep for compatibility testing
        }
        # Remove: architecture (duplicated), platform (too detailed), version (too specific)
    
    # Python info - keep version only (remove all paths)
    if 'python' in env_info:
        python = env_info['python']
        sanitized['python'] = {
            'implementation': python.get('implementation'),  # CPython, PyPy
            'version': python.get('version'),                # 3.12.2
            'version_info': python.get('version_info')       # 3.12.2
        }
        # Remove: executable (contains user paths)
    
    # Git info - keep commit hash only (remove branch and dirty status)
    if 'git' in env_info:
        git = env_info['git']
        if git.get('available') and git.get('commit_hash'):
            sanitized['git'] = {
                'commit_hash': git.get('commit_hash')
            }
        # Remove: available, branch, is_dirty (privacy/security sensitive)
    
    # Container info - keep execution environment details (important for reproducibility)
    if 'container' in env_info:
        container = env_info['container']
        sanitized['container'] = {
            'is_container': container.get('is_container', False),
            'container_type': container.get('container_type'),
            'memory_limit_gb': container.get('memory_limit_gb'),
            'cpu_limit': container.get('cpu_limit')
        }
        # Remove: container_id, container_name (potentially identifying)
        # Keep: memory/CPU limits for performance analysis
    
    # Timezone - UTC ONLY (remove all location-specific timezone info)
    # Replace all timezone info with UTC standard to prevent location identification
    sanitized['timezone'] = {
        'timezone_name': 'UTC',
        'utc_offset_hours': 0.0
    }
    
    # Timestamp - keep original timestamp (should be in UTC for consistency)
    if 'timestamp' in env_info:
        sanitized['timestamp'] = env_info['timestamp']
    
    return sanitized

# def get_environment_summary() -> str:
#     """Get a human-readable summary of the environment."""
#     env_info = collect_environment_info()
#
#     # Enhanced OS description
#     os_desc = f"{env_info['os']['system']} {env_info['os']['release']}"
#     if 'ubuntu_version' in env_info['os']:
#         os_desc = env_info['os']['ubuntu_version']
#     elif 'lsb_description' in env_info['os']:
#         os_desc = env_info['os']['lsb_description']
#
#     # Enhanced timezone description
#     tz_info = env_info['timezone']
#     timezone_desc = f"UTC{tz_info['utc_offset_hours']:+.1f} ({tz_info['timezone_name']})"
#     if 'system_timezone' in tz_info:
#         timezone_desc = f"{tz_info['system_timezone']} (UTC{tz_info['utc_offset_hours']:+.1f})"
#     elif 'timedatectl_timezone' in tz_info:
#         timezone_desc = f"{tz_info['timedatectl_timezone']} (UTC{tz_info['utc_offset_hours']:+.1f})"
#     elif 'macos_timezone' in tz_info:
#         timezone_desc = f"{tz_info['macos_timezone']} (UTC{tz_info['utc_offset_hours']:+.1f})"
#
#     # Git information
#     git_info = env_info['git']
#     git_desc = "Not available"
#     if git_info['available']:
#         commit_hash = git_info['commit_hash'][:8] if git_info['commit_hash'] else 'unknown'
#         branch = git_info['branch'] or 'unknown'
#         dirty_flag = ' (dirty)' if git_info['is_dirty'] else ''
#         git_desc = f"{commit_hash} on {branch}{dirty_flag}"
#
#     summary = f"""Environment Summary:
# OS: {os_desc} ({env_info['os']['machine']})
# CPU: {env_info['cpu']['processor']} ({env_info['cpu']['cpu_count']} cores)
# Memory: {env_info['memory']['total_gb']} GB total, {env_info['memory']['available_gb']} GB available
# Python: {env_info['python']['version']} ({env_info['python']['implementation']})
# Git: {git_desc}
# Timezone: {timezone_desc}
# Local Time: {tz_info['current_time_local']}
# Disk: {env_info['disk']['free_gb']} GB free of {env_info['disk']['total_gb']} GB total"""
#
#     return summary

