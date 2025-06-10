import platform
import psutil
import sys
from pathlib import Path
from typing import Dict, Any
import json

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.utils.logger import get_logger

logger = get_logger("environment_info")

def get_os_info() -> Dict[str, str]:
    """Get operating system information."""
    return {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "architecture": platform.architecture()[0],
        "platform": platform.platform()
    }

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

def collect_environment_info() -> Dict[str, Any]:
    """Collect all environment information for benchmark reproducibility."""
    logger.info("Collecting environment information...")
    
    env_info = {
        "timestamp": psutil.boot_time(),  # System boot time as reference
        "os": get_os_info(),
        "cpu": get_cpu_info(), 
        "memory": get_memory_info(),
        "python": get_python_info(),
        "disk": get_disk_info()
    }
    
    logger.info("Environment information collected successfully")
    logger.debug(f"Environment details: {json.dumps(env_info, indent=2, default=str)}")
    
    return env_info

def get_environment_summary() -> str:
    """Get a human-readable summary of the environment."""
    env_info = collect_environment_info()
    
    summary = f"""Environment Summary:
OS: {env_info['os']['system']} {env_info['os']['release']} ({env_info['os']['machine']})
CPU: {env_info['cpu']['processor']} ({env_info['cpu']['cpu_count']} cores)
Memory: {env_info['memory']['total_gb']} GB total, {env_info['memory']['available_gb']} GB available
Python: {env_info['python']['version']} ({env_info['python']['implementation']})
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