"""
Version detection utilities for solver backends.

This module provides functions to detect versions of various optimization
solver backends for reproducibility tracking.
"""

import sys
import importlib
from typing import Optional, Dict, Any
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.utils.logger import get_logger

logger = get_logger("version_utils")


def detect_cvxpy_version() -> str:
    """Detect CVXPY version."""
    try:
        import cvxpy as cp
        return cp.__version__
    except ImportError:
        logger.warning("CVXPY not installed")
        return "not-installed"
    except Exception as e:
        logger.error(f"Error detecting CVXPY version: {e}")
        return "unknown"


def detect_scipy_version() -> str:
    """Detect SciPy version."""
    try:
        import scipy
        return scipy.__version__
    except ImportError:
        logger.warning("SciPy not installed")
        return "not-installed"
    except Exception as e:
        logger.error(f"Error detecting SciPy version: {e}")
        return "unknown"


def detect_clarabel_version() -> str:
    """Detect CLARABEL solver version."""
    try:
        import clarabel
        return clarabel.__version__
    except ImportError:
        logger.debug("CLARABEL not installed")
        return "not-installed"
    except Exception as e:
        logger.error(f"Error detecting CLARABEL version: {e}")
        return "unknown"


def detect_scs_version() -> str:
    """Detect SCS solver version."""
    try:
        import scs
        return scs.__version__
    except ImportError:
        logger.debug("SCS not installed")
        return "not-installed"
    except Exception as e:
        logger.error(f"Error detecting SCS version: {e}")
        return "unknown"


def detect_ecos_version() -> str:
    """Detect ECOS solver version."""
    try:
        import ecos
        return ecos.__version__
    except ImportError:
        logger.debug("ECOS not installed")
        return "not-installed"
    except Exception as e:
        logger.error(f"Error detecting ECOS version: {e}")
        return "unknown"


def detect_osqp_version() -> str:
    """Detect OSQP solver version."""
    try:
        import osqp
        return osqp.__version__
    except ImportError:
        logger.debug("OSQP not installed")
        return "not-installed"
    except Exception as e:
        logger.error(f"Error detecting OSQP version: {e}")
        return "unknown"


def detect_cvxopt_version() -> str:
    """Detect CVXOPT solver version."""
    try:
        import cvxopt
        return cvxopt.__version__
    except ImportError:
        logger.debug("CVXOPT not installed")
        return "not-installed"
    except Exception as e:
        logger.error(f"Error detecting CVXOPT version: {e}")
        return "unknown"


def detect_backend_version(backend_name: str) -> str:
    """
    Detect version for a specific backend solver.
    
    Args:
        backend_name: Name of the backend (CLARABEL, SCS, ECOS, OSQP, etc.)
        
    Returns:
        Version string or 'not-installed'/'unknown'
    """
    backend_detectors = {
        'CLARABEL': detect_clarabel_version,
        'SCS': detect_scs_version,
        'ECOS': detect_ecos_version,
        'ECOS_BB': detect_ecos_version,  # ECOS_BB uses same version as ECOS
        'OSQP': detect_osqp_version,
        'CVXOPT': detect_cvxopt_version,
        'SCIPY': detect_scipy_version,  # SCIPY backend uses SciPy version
    }
    
    detector = backend_detectors.get(backend_name.upper())
    if detector:
        return detector()
    else:
        logger.warning(f"No version detector for backend: {backend_name}")
        return "unknown"


def detect_cvxpy_backend_version(backend_name: str) -> str:
    """
    Detect CVXPY + backend version combination.
    
    Args:
        backend_name: Name of the CVXPY backend
        
    Returns:
        Combined version string like 'cvxpy-1.4.0+CLARABEL-0.5.0'
    """
    cvxpy_version = detect_cvxpy_version()
    backend_version = detect_backend_version(backend_name)
    
    return f"cvxpy-{cvxpy_version}+{backend_name}-{backend_version}"


def detect_all_backend_versions() -> Dict[str, str]:
    """
    Detect versions for all available backend solvers.
    
    Returns:
        Dictionary mapping backend names to their versions
    """
    backends = ['CLARABEL', 'SCS', 'ECOS', 'OSQP', 'CVXOPT']
    versions = {}
    
    for backend in backends:
        versions[backend] = detect_backend_version(backend)
    
    return versions


def get_solver_version_info() -> Dict[str, Any]:
    """
    Get comprehensive version information for all solvers.
    
    Returns:
        Dictionary with version information for all components
    """
    info = {
        'cvxpy': detect_cvxpy_version(),
        'scipy': detect_scipy_version(),
        'backends': detect_all_backend_versions(),
        'python': sys.version.split()[0],
        'platform': sys.platform
    }
    
    return info


def format_solver_version(solver_name: str, backend: Optional[str] = None) -> str:
    """
    Format a standardized solver version string.
    
    Args:
        solver_name: Base solver name (scipy, cvxpy)
        backend: Optional backend name for CVXPY solvers
        
    Returns:
        Formatted version string
    """
    if solver_name.lower() == 'scipy':
        version = detect_scipy_version()
        return f"scipy-{version}"
    elif solver_name.lower() == 'cvxpy' and backend:
        return detect_cvxpy_backend_version(backend)
    elif solver_name.lower() == 'cvxpy':
        version = detect_cvxpy_version()
        return f"cvxpy-{version}"
    else:
        logger.warning(f"Unknown solver name: {solver_name}")
        return f"{solver_name}-unknown"


if __name__ == "__main__":
    # Test version detection
    print("=== Solver Version Detection Test ===")
    
    print(f"CVXPY: {detect_cvxpy_version()}")
    print(f"SciPy: {detect_scipy_version()}")
    
    print("\nBackend versions:")
    backend_versions = detect_all_backend_versions()
    for backend, version in backend_versions.items():
        print(f"  {backend}: {version}")
    
    print("\nFormatted solver versions:")
    print(f"SciPy: {format_solver_version('scipy')}")
    print(f"CVXPY+CLARABEL: {format_solver_version('cvxpy', 'CLARABEL')}")
    
    print("\nComplete version info:")
    import json
    print(json.dumps(get_solver_version_info(), indent=2))