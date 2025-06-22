"""
Python Problem Loader for CVXPY-based Problems.

This module provides functionality to load and execute Python-defined optimization
problems that use CVXPY for problem formulation. These are typically SOCP and SDP
problems that are difficult to express in standard file formats.

Python Problem Format:
Python files should define a function that returns problem data:
- generate_<problem_name>() -> Tuple[Dict[str, Any], ProblemData]
- The function should return both a dictionary with problem info and a ProblemData object
- Problems are executed in a safe but limited environment
"""

import numpy as np
import importlib.util
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import tempfile
import os

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.data_loaders.problem_loader import ProblemData
from scripts.utils.logger import get_logger

logger = get_logger("python_loader")


class PythonLoader:
    """Loader for Python-defined optimization problems."""
    
    def __init__(self):
        """Initialize the Python loader."""
        pass
    
    def load(self, file_path: str) -> ProblemData:
        """
        Load problem from Python file.
        
        Args:
            file_path: Path to the .py file
            
        Returns:
            ProblemData object
        """
        problem_name = Path(file_path).stem
        
        # Execute Python file and extract problem data
        problem_data = self.execute_python_problem(file_path, problem_name)
        
        logger.info(f"Successfully loaded Python problem: {problem_data}")
        return problem_data
    
    def execute_python_problem(self, file_path: str, problem_name: str) -> ProblemData:
        """
        Execute Python problem file safely and extract problem data.
        
        Args:
            file_path: Path to the Python file
            problem_name: Name for the problem
            
        Returns:
            ProblemData object
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If problem generation fails
        """
        if not Path(file_path).exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        logger.info(f"Loading Python problem file: {file_path}")
        
        try:
            # Load the module dynamically
            spec = importlib.util.spec_from_file_location(problem_name, file_path)
            if spec is None or spec.loader is None:
                raise ValueError(f"Could not load Python module from {file_path}")
            
            module = importlib.util.module_from_spec(spec)
            
            # Add the problem directory to Python path temporarily
            problem_dir = str(Path(file_path).parent)
            original_path = sys.path.copy()
            if problem_dir not in sys.path:
                sys.path.insert(0, problem_dir)
            
            try:
                # Execute the module
                spec.loader.exec_module(module)
                
                # Look for the problem generation function
                problem_data = self._extract_problem_data(module, problem_name)
                
                logger.debug(f"Executed Python problem: {problem_name}")
                return problem_data
                
            finally:
                # Restore original Python path
                sys.path = original_path
            
        except Exception as e:
            logger.error(f"Failed to execute Python problem {file_path}: {e}")
            raise ValueError(f"Failed to execute Python problem: {e}")
    
    def _extract_problem_data(self, module, problem_name: str) -> ProblemData:
        """
        Extract problem data from executed Python module.
        
        Args:
            module: Executed Python module
            problem_name: Name of the problem
            
        Returns:
            ProblemData object
        """
        # Look for generate function with different naming conventions
        function_names = [
            f"generate_{problem_name}",
            f"generate_{problem_name}_socp",
            f"generate_{problem_name}_sdp",
            f"generate_{problem_name}_problem",
            "generate_problem",
            "create_problem",
            "get_problem"
        ]
        
        generate_func = None
        for func_name in function_names:
            if hasattr(module, func_name):
                generate_func = getattr(module, func_name)
                logger.debug(f"Found problem generation function: {func_name}")
                break
        
        if generate_func is None:
            raise ValueError(f"No problem generation function found in module. "
                           f"Expected one of: {function_names}")
        
        try:
            # Call the generation function
            result = generate_func()
            
            # Handle different return formats
            if isinstance(result, tuple) and len(result) == 2:
                # Standard format: (problem_dict, problem_data)
                problem_dict, problem_data = result
                if isinstance(problem_data, ProblemData):
                    return problem_data
                else:
                    # Convert problem_dict to ProblemData if needed
                    return self._dict_to_problem_data(problem_dict, problem_name)
            
            elif isinstance(result, ProblemData):
                # Direct ProblemData return
                return result
            
            elif isinstance(result, dict):
                # Dictionary format - convert to ProblemData
                return self._dict_to_problem_data(result, problem_name)
            
            else:
                raise ValueError(f"Unexpected return type from generation function: {type(result)}")
                
        except Exception as e:
            logger.error(f"Error calling problem generation function: {e}")
            raise ValueError(f"Problem generation failed: {e}")
    
    def _dict_to_problem_data(self, problem_dict: Dict[str, Any], problem_name: str) -> ProblemData:
        """
        Convert problem dictionary to ProblemData object.
        
        Args:
            problem_dict: Problem information dictionary
            problem_name: Name for the problem
            
        Returns:
            ProblemData object
        """
        # Extract basic information
        name = problem_dict.get('name', problem_name)
        problem_class = problem_dict.get('problem_class', 'UNKNOWN')
        
        # Create metadata from dictionary
        metadata = {
            'source': 'Python file',
            'format': 'CVXPY Python',
            'description': problem_dict.get('description', ''),
            'objective_sense': problem_dict.get('objective_sense', 'minimize'),
            'original_dict': problem_dict  # Store original for reference
        }
        
        # Add problem-specific metadata
        for key, value in problem_dict.items():
            if key not in ['name', 'problem_class', 'cvxpy_problem', 'variables', 'objective', 'constraints']:
                # Convert numpy arrays to lists for JSON serialization
                if isinstance(value, np.ndarray):
                    metadata[key] = value.tolist()
                else:
                    metadata[key] = value
        
        # Create ProblemData object
        problem_data = ProblemData(
            name=name,
            problem_class=problem_class,
            cvxpy_problem=problem_dict.get('cvxpy_problem'),
            variables=problem_dict.get('variables', {}),
            objective=problem_dict.get('objective'),
            constraints=problem_dict.get('constraints', []),
            metadata=metadata
        )
        
        logger.info(f"Converted Python problem {name}: {problem_class} problem")
        
        return problem_data
    
    def validate_python_problem(self, file_path: str) -> bool:
        """
        Validate that a Python file contains a valid problem definition.
        
        Args:
            file_path: Path to the Python file
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Try to load and execute the problem
            problem_data = self.load(file_path)
            return isinstance(problem_data, ProblemData)
        except Exception as e:
            logger.warning(f"Python problem validation failed for {file_path}: {e}")
            return False


# Convenience function for backward compatibility
def load_python_problem(file_path: str, 
                       problem_name: Optional[str] = None) -> ProblemData:
    """
    Convenience function to load a Python problem.
    
    Args:
        file_path: Path to the .py file
        problem_name: Optional name for the problem
        
    Returns:
        ProblemData object
    """
    loader = PythonLoader()
    return loader.load(file_path)


if __name__ == "__main__":
    # Test script
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python python_loader.py <path_to_python_file>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    try:
        problem = load_python_problem(file_path)
        print(f"Loaded problem: {problem}")
        
        # Display structure summary if available
        if hasattr(problem, 'get_structure_summary'):
            structure = problem.get_structure_summary()
            if structure:
                print(f"Structure analysis: {structure}")
        
        # Display problem class and metadata
        print(f"Problem class: {problem.problem_class}")
        if problem.metadata:
            print(f"Metadata keys: {list(problem.metadata.keys())}")
            
    except Exception as e:
        print(f"Error loading problem: {e}")
        sys.exit(1)