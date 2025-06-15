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
    return {
        "total": memory.total,
        "available": memory.available,
        "percent": memory.percent,
        "used": memory.used,
        "free": memory.free,
        "total_gb": round(memory.total / (1024**3), 2),
        "available_gb": round(memory.available / (1024**3), 2)
    }

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
    """Collect all environment information for benchmark reproducibility."""
    logger.info("Collecting environment information...")
    
    env_info = {
        "timestamp": psutil.boot_time(),  # System boot time as reference
        "os": get_os_info(),
        "cpu": get_cpu_info(), 
        "memory": get_memory_info(),
        "python": get_python_info(),
        "disk": get_disk_info(),
        "timezone": get_timezone_info(),
        "git": get_git_info()  # Add Git repository information
    }
    
    logger.info("Environment information collected successfully")
    logger.debug(f"Environment details: {json.dumps(env_info, indent=2, default=str)}")
    
    return env_info

def get_environment_summary() -> str:
    """Get a human-readable summary of the environment."""
    env_info = collect_environment_info()
    
    # Enhanced OS description
    os_desc = f"{env_info['os']['system']} {env_info['os']['release']}"
    if 'ubuntu_version' in env_info['os']:
        os_desc = env_info['os']['ubuntu_version']
    elif 'lsb_description' in env_info['os']:
        os_desc = env_info['os']['lsb_description']
    
    # Enhanced timezone description
    tz_info = env_info['timezone']
    timezone_desc = f"UTC{tz_info['utc_offset_hours']:+.1f} ({tz_info['timezone_name']})"
    if 'system_timezone' in tz_info:
        timezone_desc = f"{tz_info['system_timezone']} (UTC{tz_info['utc_offset_hours']:+.1f})"
    elif 'timedatectl_timezone' in tz_info:
        timezone_desc = f"{tz_info['timedatectl_timezone']} (UTC{tz_info['utc_offset_hours']:+.1f})"
    elif 'macos_timezone' in tz_info:
        timezone_desc = f"{tz_info['macos_timezone']} (UTC{tz_info['utc_offset_hours']:+.1f})"
    
    # Git information
    git_info = env_info['git']
    git_desc = "Not available"
    if git_info['available']:
        commit_hash = git_info['commit_hash'][:8] if git_info['commit_hash'] else 'unknown'
        branch = git_info['branch'] or 'unknown'
        dirty_flag = ' (dirty)' if git_info['is_dirty'] else ''
        git_desc = f"{commit_hash} on {branch}{dirty_flag}"
    
    summary = f"""Environment Summary:
OS: {os_desc} ({env_info['os']['machine']})
CPU: {env_info['cpu']['processor']} ({env_info['cpu']['cpu_count']} cores)
Memory: {env_info['memory']['total_gb']} GB total, {env_info['memory']['available_gb']} GB available
Python: {env_info['python']['version']} ({env_info['python']['implementation']})
Git: {git_desc}
Timezone: {timezone_desc}
Local Time: {tz_info['current_time_local']}
Disk: {env_info['disk']['free_gb']} GB free of {env_info['disk']['total_gb']} GB total"""
    
    return summary

if __name__ == "__main__":
    # Test script to collect and display environment information
    try:
        print("Collecting environment information...\n")
        
        # Test individual functions
        print("OS Information:")
        os_info = get_os_info()
        for key, value in os_info.items():
            print(f"  {key}: {value}")
        
        print("\nCPU Information:")
        cpu_info = get_cpu_info()
        for key, value in cpu_info.items():
            print(f"  {key}: {value}")
        
        print("\nMemory Information:")
        memory_info = get_memory_info()
        for key, value in memory_info.items():
            print(f"  {key}: {value}")
        
        print("\nPython Information:")
        python_info = get_python_info()
        for key, value in python_info.items():
            print(f"  {key}: {value}")
        
        print("\nDisk Information:")
        disk_info = get_disk_info()
        for key, value in disk_info.items():
            print(f"  {key}: {value}")
        
        print("\n" + "="*50)
        print("COMPLETE ENVIRONMENT INFO:")
        print("="*50)
        
        # Test complete collection
        complete_info = collect_environment_info()
        print(json.dumps(complete_info, indent=2, default=str))
        
        print("\n" + "="*50)
        print("ENVIRONMENT SUMMARY:")
        print("="*50)
        print(get_environment_summary())
        
    except Exception as e:
        logger.error(f"Failed to collect environment information: {e}")
        raise