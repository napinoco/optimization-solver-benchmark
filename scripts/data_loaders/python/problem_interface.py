"""
Problem Interface Module for Benchmark System.

This module provides a unified interface for managing problem loading and registry operations,
creating symmetry with the solver interface modules. It centralizes problem management
responsibilities and eliminates duplication across the system.

Key Features:
- Centralized problem registry management
- Unified problem loading with format detection
- Problem validation and compatibility checking
- Statistics and monitoring for problem libraries
- Consistent interface with solver management modules

Architecture:
This module extracts problem loading logic from the benchmark runner and other components
to create clean separation between orchestration and problem-specific logic.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.data_loaders.problem_loader import ProblemData
from scripts.data_loaders.python.dat_loader import DATLoader
from scripts.data_loaders.python.mat_loader import MATLoader
from scripts.utils.logger import get_logger

logger = get_logger("problem_interface")


class ProblemInterface:
    """
    Interface for managing problem loading and registry operations.
    
    This class provides centralized management of optimization problems,
    handling registry loading, format detection, and problem loading.
    Creates architectural symmetry with solver interface modules.
    """

    # Supported file format loaders
    FORMAT_LOADERS = {
        "mat": MATLoader,
        "dat-s": DATLoader,
        # Future loaders can be added here
        # "mps": MPSLoader,
        # "qps": QPSLoader,
        # "python": PythonLoader,
    }

    def __init__(self, registry_path: Optional[str] = None, **kwargs):
        """
        Initialize problem interface.
        
        Args:
            registry_path: Optional path to problem registry YAML file
            **kwargs: Additional configuration parameters
        """
        self.config = kwargs

        # Set registry path
        if registry_path:
            self.registry_path = Path(registry_path)
        else:
            self.registry_path = project_root / "config" / "problem_registry.yaml"

        # Load problem registry on initialization
        self.problem_registry = self._load_problem_registry()

        # Cache for problem statistics
        self._statistics_cache = None

        logger.info(f"Initialized problem interface with {len(self.problem_registry.get('problem_libraries', {}))} problems")
        logger.debug(f"Registry path: {self.registry_path}")

    def load_problem(self, problem_name: str, problem_config: Optional[Dict[str, Any]] = None) -> ProblemData:
        """
        Load problem using appropriate loader based on file type.
        
        Args:
            problem_name: Name of the problem to load
            problem_config: Optional problem configuration (if None, loads from registry)
            
        Returns:
            Loaded problem data
            
        Raises:
            ValueError: If problem not found or unsupported file type
        """
        logger.debug(f"Loading problem: {problem_name}")

        # Get problem configuration
        if problem_config is None:
            problem_config = self.get_problem_config(problem_name)

        # Extract file information
        file_type = problem_config['file_type']
        file_path = self._resolve_file_path(problem_config['file_path'])

        logger.debug(f"Loading problem {problem_name} from {file_path} (type: {file_type})")

        # Select appropriate loader
        if file_type not in self.FORMAT_LOADERS:
            supported_types = list(self.FORMAT_LOADERS.keys())
            raise ValueError(f"Unsupported file type: {file_type}. Supported: {supported_types}")

        # Create and use loader
        loader_class = self.FORMAT_LOADERS[file_type]
        loader = loader_class()

        try:
            problem_data = loader.load(str(file_path))
            logger.debug(f"Successfully loaded {problem_name}: {problem_data}")
            return problem_data
        except Exception as e:
            raise ValueError(f"Failed to load problem {problem_name}: {e}") from e

    def get_problem_config(self, problem_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific problem.
        
        Args:
            problem_name: Name of the problem
            
        Returns:
            Problem configuration dictionary
            
        Raises:
            ValueError: If problem not found in registry
        """
        problem_libraries = self.problem_registry.get('problem_libraries', {})

        if problem_name not in problem_libraries:
            available_problems = list(problem_libraries.keys())
            raise ValueError(f"Problem '{problem_name}' not found in registry. "
                           f"Available problems: {available_problems}")

        return problem_libraries[problem_name]

    def get_available_problems(self, library_filter: Optional[List[str]] = None,
                             test_only: bool = False) -> List[str]:
        """
        Get list of available problems.
        
        Args:
            library_filter: Optional list of library names to filter by
            test_only: If True, only return problems marked for testing
            
        Returns:
            List of problem names
        """
        problems = []
        problem_libraries = self.problem_registry.get('problem_libraries', {})

        for problem_name, config in problem_libraries.items():
            # Filter by library if specified
            if library_filter:
                library_name = config.get('library_name', 'unknown')
                if library_name not in library_filter:
                    continue

            # Filter by test flag if specified
            if test_only:
                if not config.get('for_test_flag', False):
                    continue

            problems.append(problem_name)

        return problems

    def get_problems_by_type(self, problem_types: List[str]) -> List[str]:
        """
        Get problems filtered by problem type.
        
        Args:
            problem_types: List of problem types (LP, QP, SOCP, SDP)
            
        Returns:
            List of problem names matching the types
        """
        problems = []
        problem_libraries = self.problem_registry.get('problem_libraries', {})

        for problem_name, config in problem_libraries.items():
            problem_type = config.get('problem_type', 'UNKNOWN')
            if problem_type in problem_types:
                problems.append(problem_name)

        return problems

    def get_problems_by_library(self, library_names: List[str]) -> List[str]:
        """
        Get problems filtered by library name.
        
        Args:
            library_names: List of library names (DIMACS, SDPLIB, etc.)
            
        Returns:
            List of problem names from specified libraries
        """
        return self.get_available_problems(library_filter=library_names)

    def validate_problem_compatibility(self, problem_name: str, file_types: List[str]) -> bool:
        """
        Check if problem is compatible with specified file types.
        
        Args:
            problem_name: Name of the problem
            file_types: List of supported file types
            
        Returns:
            True if problem file type is supported
        """
        try:
            problem_config = self.get_problem_config(problem_name)
            file_type = problem_config.get('file_type', 'unknown')
            return file_type in file_types
        except ValueError:
            return False

    def get_problem_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about available problems.
        
        Returns:
            Dictionary with problem statistics
        """
        if self._statistics_cache is None:
            self._statistics_cache = self._compute_statistics()

        return self._statistics_cache.copy()

    def refresh_registry(self) -> Dict[str, Any]:
        """
        Refresh the problem registry from file.
        
        Returns:
            Updated problem registry
        """
        logger.info("Refreshing problem registry...")
        self.problem_registry = self._load_problem_registry()

        # Clear statistics cache
        self._statistics_cache = None

        return self.problem_registry

    def validate_environment(self) -> Dict[str, Any]:
        """
        Validate problem loading environment and return detailed report.
        
        Returns:
            Dictionary with validation results
        """
        logger.info("Validating problem loading environment...")

        validation_report = {
            "status": "success",
            "problems": {},
            "summary": {
                "total_problems": 0,
                "loadable_problems": 0,
                "failed_problems": 0
            },
            "statistics": self.get_problem_statistics(),
            "registry_path": str(self.registry_path),
            "registry_exists": self.registry_path.exists()
        }

        # Test loading a few problems from each library
        test_problems = self.get_available_problems(test_only=True)
        if not test_problems:
            # If no test problems, take first problem from each library
            stats = self.get_problem_statistics()
            test_problems = []
            for library in stats.get("libraries", {}):
                library_problems = self.get_problems_by_library([library])
                if library_problems:
                    test_problems.append(library_problems[0])

        # Limit test problems for validation performance
        test_problems = test_problems[:5]

        for problem_name in test_problems:
            validation_report["summary"]["total_problems"] += 1
            try:
                problem_data = self.load_problem(problem_name)

                validation_report["problems"][problem_name] = {
                    "status": "loadable",
                    "type": problem_data.problem_class,
                    "variables": problem_data._num_variables,
                    "constraints": problem_data._num_constraints
                }
                validation_report["summary"]["loadable_problems"] += 1

            except Exception as e:
                validation_report["problems"][problem_name] = {
                    "status": "error",
                    "error": str(e)
                }
                validation_report["summary"]["failed_problems"] += 1
                validation_report["status"] = "partial" if validation_report["summary"]["loadable_problems"] > 0 else "failed"

        # Log summary
        loadable = validation_report["summary"]["loadable_problems"]
        total = validation_report["summary"]["total_problems"]
        logger.info(f"Problem validation complete: {loadable}/{total} problems loadable")

        return validation_report

    def _load_problem_registry(self) -> Dict[str, Any]:
        """
        Load problem registry from YAML file.
        
        Returns:
            Problem registry dictionary
            
        Raises:
            FileNotFoundError: If registry file not found
            yaml.YAMLError: If registry file has invalid YAML
        """
        logger.debug(f"Loading problem registry: {self.registry_path}")

        try:
            with open(self.registry_path, 'r') as f:
                registry = yaml.safe_load(f)

            # Validate registry structure
            if not isinstance(registry, dict):
                raise ValueError("Registry must be a dictionary")

            if 'problem_libraries' not in registry:
                logger.warning("Registry missing 'problem_libraries' key, using empty structure")
                registry['problem_libraries'] = {}

            logger.info(f"Loaded {len(registry['problem_libraries'])} problems from registry")
            return registry

        except FileNotFoundError:
            logger.error(f"Problem registry not found: {self.registry_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Invalid YAML in problem registry: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load problem registry: {e}")
            raise

    def _resolve_file_path(self, relative_path: str) -> Path:
        """
        Resolve relative file path to absolute path.
        
        Args:
            relative_path: Relative path from project root
            
        Returns:
            Absolute path to problem file
        """
        return project_root / relative_path

    def _compute_statistics(self) -> Dict[str, Any]:
        """
        Compute statistics about available problems.
        
        Returns:
            Dictionary with computed statistics
        """
        problem_libraries = self.problem_registry.get('problem_libraries', {})

        # Count by library
        library_counts = {}
        type_counts = {}
        format_counts = {}
        test_problems = 0

        for config in problem_libraries.values():
            # Count by library
            library = config.get('library_name', 'unknown')
            library_counts[library] = library_counts.get(library, 0) + 1

            # Count by type
            problem_type = config.get('problem_type', 'UNKNOWN')
            type_counts[problem_type] = type_counts.get(problem_type, 0) + 1

            # Count by format
            file_type = config.get('file_type', 'unknown')
            format_counts[file_type] = format_counts.get(file_type, 0) + 1

            # Count test problems
            if config.get('for_test_flag', False):
                test_problems += 1

        return {
            "total_problems": len(problem_libraries),
            "libraries": library_counts,
            "problem_types": type_counts,
            "file_formats": format_counts,
            "test_problems": test_problems,
            "supported_formats": list(self.FORMAT_LOADERS.keys())
        }


def get_problem_interface(registry_path: Optional[str] = None, **kwargs) -> ProblemInterface:
    """
    Factory function to get problem interface instance.
    
    Args:
        registry_path: Optional path to registry file
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured ProblemInterface instance
    """
    return ProblemInterface(registry_path=registry_path, **kwargs)


# Module-level convenience functions for backward compatibility
def load_problem_registry(registry_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function to load problem registry.
    
    Args:
        registry_path: Optional path to registry file
        
    Returns:
        Problem registry dictionary
    """
    interface = get_problem_interface(registry_path=registry_path)
    return interface.problem_registry


def load_problem(problem_name: str, problem_config: Optional[Dict[str, Any]] = None) -> ProblemData:
    """
    Convenience function to load a problem.
    
    Args:
        problem_name: Name of problem to load
        problem_config: Optional problem configuration
        
    Returns:
        Loaded problem data
    """
    interface = get_problem_interface()
    return interface.load_problem(problem_name, problem_config)


def get_available_problems(library_filter: Optional[List[str]] = None, test_only: bool = False) -> List[str]:
    """
    Convenience function to get available problems.
    
    Args:
        library_filter: Optional library name filter
        test_only: Only return test problems
        
    Returns:
        List of available problem names
    """
    interface = get_problem_interface()
    return interface.get_available_problems(library_filter=library_filter, test_only=test_only)


if __name__ == "__main__":
    # Test script for problem interface
    print("Testing Problem Interface...")
    print("=" * 50)

    # Initialize interface
    print("\n1. Initializing problem interface...")
    interface = get_problem_interface()
    print("   ✓ Interface initialized")

    # Show statistics
    stats = interface.get_problem_statistics()
    print("\n2. Problem statistics:")
    print(f"   Total problems: {stats['total_problems']}")
    print(f"   Test problems: {stats['test_problems']}")
    print(f"   Libraries: {list(stats['libraries'].keys())}")
    print(f"   Problem types: {list(stats['problem_types'].keys())}")
    print(f"   File formats: {list(stats['file_formats'].keys())}")
    print(f"   Supported formats: {stats['supported_formats']}")

    # Show available problems
    available = interface.get_available_problems()
    print(f"\n3. Available problems: {len(available)}")
    for problem_name in available[:5]:  # Show first 5
        try:
            config = interface.get_problem_config(problem_name)
            library = config.get('library_name', 'unknown')
            problem_type = config.get('problem_type', 'unknown')
            print(f"   • {problem_name} ({library}, {problem_type})")
        except Exception as e:
            print(f"   • {problem_name} (error: {e})")

    if len(available) > 5:
        print(f"   ... and {len(available) - 5} more")

    # Test problem loading
    test_problems = interface.get_available_problems(test_only=True)
    if test_problems:
        print("\n4. Testing problem loading...")
        test_problem = test_problems[0]
        try:
            problem_data = interface.load_problem(test_problem)
            print(f"   ✓ Loaded {test_problem}: {problem_data}")
        except Exception as e:
            print(f"   ✗ Failed to load {test_problem}: {e}")

    # Environment validation
    print("\n5. Environment validation...")
    validation = interface.validate_environment()
    print(f"   Status: {validation['status']}")
    print(f"   Registry exists: {validation['registry_exists']}")
    print(f"   Loadable: {validation['summary']['loadable_problems']}")
    print(f"   Failed: {validation['summary']['failed_problems']}")

    print("\n" + "=" * 50)
    print("✓ Problem interface testing complete!")
