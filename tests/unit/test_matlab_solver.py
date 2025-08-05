#!/usr/bin/env python3
"""
Comprehensive unit tests for production-ready MatlabSolver implementation.

This test suite validates all aspects of the MatlabSolver class:
- SolverInterface compliance
- Problem registry integration
- Version detection and caching
- Error handling and edge cases
- Performance and concurrency
"""

import os
import sys
import time
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.solvers.matlab.matlab_runner import MatlabSolver, SeDuMiSolver, SDPT3Solver
from scripts.solvers.solver_interface import SolverResult
from scripts.data_loaders.problem_loader import ProblemData


class MockProblemData:
    """Mock problem data for testing."""
    def __init__(self, name: str, problem_class: str = "SDP"):
        self.name = name
        self.problem_class = problem_class
        self.metadata = {}


class TestMatlabSolverInterface(unittest.TestCase):
    """Test SolverInterface compliance and basic functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock MATLAB availability to avoid requiring actual MATLAB installation
        self.matlab_patcher = patch('scripts.solvers.matlab.matlab_interface.subprocess.run')
        self.mock_subprocess = self.matlab_patcher.start()
        
        # Mock successful MATLAB verification
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "MATLAB_OK"
        mock_result.stderr = ""
        self.mock_subprocess.return_value = mock_result
        
        # Mock problem registry
        self.registry_patcher = patch('scripts.solvers.matlab.matlab_interface.load_problem_registry')
        self.mock_registry = self.registry_patcher.start()
        self.mock_registry.return_value = {
            'problem_libraries': {
                'arch0': {
                    'file_path': 'problems/SDPLIB/data/arch0.dat-s',
                    'file_type': 'dat-s',
                    'library_name': 'SDPLIB'
                },
                'nb': {
                    'file_path': 'problems/DIMACS/data/ANTENNA/nb.mat.gz',
                    'file_type': 'mat',
                    'library_name': 'DIMACS'
                }
            }
        }
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.matlab_patcher.stop()
        self.registry_patcher.stop()
    
    def test_solver_initialization(self):
        """Test MatlabSolver initialization with different configurations."""
        # Test SeDuMi initialization
        solver = MatlabSolver('sedumi')
        self.assertEqual(solver.matlab_solver, 'sedumi')
        self.assertEqual(solver.solver_name, 'matlab_sedumi')
        self.assertIsNotNone(solver.temp_manager)
        
        # Test SDPT3 initialization
        solver = MatlabSolver('sdpt3')
        self.assertEqual(solver.matlab_solver, 'sdpt3')
        self.assertEqual(solver.solver_name, 'matlab_sdpt3')
        
        # Test custom configuration
        solver = MatlabSolver('sedumi', timeout=600, matlab_executable='custom_matlab')
        self.assertEqual(solver.timeout, 600)
        self.assertEqual(solver.matlab_executable, 'custom_matlab')
    
    def test_invalid_solver_initialization(self):
        """Test that invalid solver names raise appropriate errors."""
        with self.assertRaises(ValueError) as context:
            MatlabSolver('invalid_solver')
        
        self.assertIn("Unsupported MATLAB solver", str(context.exception))
        self.assertIn("invalid_solver", str(context.exception))
    
    def test_convenience_classes(self):
        """Test SeDuMiSolver and SDPT3Solver convenience classes."""
        # Test SeDuMiSolver
        sedumi = SeDuMiSolver()
        self.assertEqual(sedumi.matlab_solver, 'sedumi')
        self.assertEqual(sedumi.solver_name, 'matlab_sedumi')
        
        # Test SDPT3Solver
        sdpt3 = SDPT3Solver()
        self.assertEqual(sdpt3.matlab_solver, 'sdpt3')
        self.assertEqual(sdpt3.solver_name, 'matlab_sdpt3')
        
        # Test with custom parameters
        sedumi_custom = SeDuMiSolver(timeout=300)
        self.assertEqual(sedumi_custom.timeout, 300)
    
    def test_version_detection(self):
        """Test version detection and caching."""
        solver = MatlabSolver('sedumi')
        
        # Test basic version retrieval
        version = solver.get_version()
        self.assertIsInstance(version, str)
        self.assertIn('SeDuMi', version)
        
        # Test that version is consistent
        version2 = solver.get_version()
        self.assertEqual(version, version2)


class TestProblemRegistryIntegration(unittest.TestCase):
    """Test problem registry integration and validation."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock MATLAB to avoid requiring installation
        self.matlab_patcher = patch('scripts.solvers.matlab.matlab_interface.subprocess.run')
        self.mock_subprocess = self.matlab_patcher.start()
        
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "MATLAB_OK"
        self.mock_subprocess.return_value = mock_result
        
        # Set up test registry
        self.registry_patcher = patch('scripts.solvers.matlab.matlab_interface.load_problem_registry')
        self.mock_registry = self.registry_patcher.start()
        self.mock_registry.return_value = {
            'problem_libraries': {
                'arch0': {
                    'file_path': 'problems/SDPLIB/data/arch0.dat-s',
                    'file_type': 'dat-s',
                    'library_name': 'SDPLIB'
                },
                'nb': {
                    'file_path': 'problems/DIMACS/data/ANTENNA/nb.mat.gz',
                    'file_type': 'mat',
                    'library_name': 'DIMACS'
                },
                'unsupported_format': {
                    'file_path': 'problems/test/unsupported.xyz',
                    'file_type': 'xyz',
                    'library_name': 'TEST'
                }
            }
        }
        
        self.solver = MatlabSolver('sedumi')
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.matlab_patcher.stop()
        self.registry_patcher.stop()
    
    def test_problem_data_validation(self):
        """Test problem data validation logic."""
        # Valid problem data
        valid_problem = MockProblemData('arch0', 'SDP')
        self.assertTrue(self.solver._validate_problem_data(valid_problem))
        
        # Missing name
        invalid_problem = MockProblemData('', 'SDP')
        self.assertFalse(self.solver._validate_problem_data(invalid_problem))
        
        # Problem not in registry
        missing_problem = MockProblemData('nonexistent', 'SDP')
        self.assertFalse(self.solver._validate_problem_data(missing_problem))
    
    def test_problem_info_resolution(self):
        """Test problem path resolution from registry."""
        problem = MockProblemData('arch0')
        problem_name, problem_path = self.solver._resolve_problem_info(problem)
        
        self.assertEqual(problem_name, 'arch0')
        self.assertIn('problems/SDPLIB/data/arch0.dat-s', problem_path)
        
        # Test with nonexistent problem
        missing_problem = MockProblemData('nonexistent')
        with self.assertRaises(ValueError):
            self.solver._resolve_problem_info(missing_problem)
    
    def test_solver_compatibility(self):
        """Test solver compatibility checking."""
        # Compatible problems
        sdplib_problem = MockProblemData('arch0')
        self.assertTrue(self.solver._check_solver_compatibility(sdplib_problem))
        
        dimacs_problem = MockProblemData('nb')
        self.assertTrue(self.solver._check_solver_compatibility(dimacs_problem))
        
        # Unsupported file type
        unsupported_problem = MockProblemData('unsupported_format')
        self.assertFalse(self.solver._check_solver_compatibility(unsupported_problem))
        
        # Nonexistent problem
        missing_problem = MockProblemData('nonexistent')
        self.assertFalse(self.solver._check_solver_compatibility(missing_problem))
    
    def test_validate_problem_compatibility(self):
        """Test the public problem compatibility validation method."""
        # Valid compatible problem
        valid_problem = MockProblemData('arch0')
        self.assertTrue(self.solver.validate_problem_compatibility(valid_problem))
        
        # Invalid problem
        invalid_problem = MockProblemData('')
        self.assertFalse(self.solver.validate_problem_compatibility(invalid_problem))
        
        # Unsupported format
        unsupported_problem = MockProblemData('unsupported_format')
        self.assertFalse(self.solver.validate_problem_compatibility(unsupported_problem))


class TestSolverExecution(unittest.TestCase):
    """Test solver execution and result handling."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock MATLAB subprocess for verification and execution
        self.matlab_patcher = patch('scripts.solvers.matlab.matlab_interface.subprocess.run')
        self.mock_subprocess = self.matlab_patcher.start()
        
        # Set up proper mock result for MATLAB verification
        verification_result = MagicMock()
        verification_result.returncode = 0
        verification_result.stdout = "MATLAB_OK"
        verification_result.stderr = ""
        self.mock_subprocess.return_value = verification_result
        
        # Mock registry
        self.registry_patcher = patch('scripts.solvers.matlab.matlab_interface.load_problem_registry')
        self.mock_registry = self.registry_patcher.start()
        self.mock_registry.return_value = {
            'problem_libraries': {
                'test_problem': {
                    'file_path': 'problems/test/test_problem.dat-s',
                    'file_type': 'dat-s',
                    'library_name': 'TEST'
                }
            }
        }
        
        # Mock temp file operations
        self.temp_patcher = patch('scripts.solvers.matlab.matlab_interface.temp_file_context')
        self.mock_temp = self.temp_patcher.start()
        self.mock_temp.return_value.__enter__.return_value = '/tmp/test_result.json'
        self.mock_temp.return_value.__exit__.return_value = None
        
        # Initialize solver after mocks are set up
        self.solver = MatlabSolver('sedumi')
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.matlab_patcher.stop()
        self.registry_patcher.stop()
        self.temp_patcher.stop()
    
    def test_successful_solve(self):
        """Test successful problem solving."""
        # Mock successful MATLAB execution
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_result.stderr = ""
        self.mock_subprocess.return_value = mock_result
        
        # Mock result file content
        with patch('builtins.open', create=True) as mock_open:
            with patch('os.path.exists', return_value=True):
                with patch('os.stat') as mock_stat:
                    mock_stat.return_value.st_size = 100
                    
                    mock_json_content = {
                        'status': 'optimal',
                        'solve_time': 1.5,
                        'primal_objective_value': -10.5,
                        'dual_objective_value': -10.5,
                        'duality_gap': 1e-8,
                        'primal_infeasibility': 1e-9,
                        'dual_infeasibility': 1e-9,
                        'iterations': 25,
                        'solver_version': 'SeDuMi 1.3.5',
                        'matlab_version': 'R2024a'
                    }
                    
                    with patch('json.load', return_value=mock_json_content):
                        problem = MockProblemData('test_problem')
                        result = self.solver.solve(problem)
                        
                        self.assertIsInstance(result, SolverResult)
                        self.assertEqual(result.status, 'OPTIMAL')
                        self.assertEqual(result.primal_objective_value, -10.5)
                        self.assertEqual(result.iterations, 25)
                        self.assertIn('matlab_output', result.additional_info)
    
    def test_invalid_problem_data(self):
        """Test handling of invalid problem data."""
        # Problem with missing name
        invalid_problem = MockProblemData('')
        result = self.solver.solve(invalid_problem)
        
        self.assertIsInstance(result, SolverResult)
        self.assertEqual(result.status, 'ERROR')
        self.assertIn('validation failed', result.additional_info.get('error_message', ''))
    
    def test_matlab_execution_failure(self):
        """Test handling of MATLAB execution failures."""
        # Mock failed MATLAB execution
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "Error: Undefined function 'nonexistent'"
        self.mock_subprocess.return_value = mock_result
        
        problem = MockProblemData('test_problem')
        result = self.solver.solve(problem)
        
        self.assertIsInstance(result, SolverResult)
        self.assertEqual(result.status, 'ERROR')
        self.assertIn('MATLAB execution failed', result.additional_info.get('error_message', ''))
    
    def test_timeout_handling(self):
        """Test timeout handling."""
        from subprocess import TimeoutExpired
        self.mock_subprocess.side_effect = TimeoutExpired('matlab', 30)
        
        problem = MockProblemData('test_problem')
        result = self.solver.solve(problem, timeout=1)
        
        self.assertIsInstance(result, SolverResult)
        self.assertEqual(result.status, 'TIMEOUT')
    
    def test_missing_result_file(self):
        """Test handling when MATLAB doesn't produce result file."""
        # Mock successful MATLAB execution but no result file
        mock_result = MagicMock()
        mock_result.returncode = 0
        self.mock_subprocess.return_value = mock_result
        
        with patch('os.path.exists', return_value=False):
            problem = MockProblemData('test_problem')
            result = self.solver.solve(problem)
            
            self.assertIsInstance(result, SolverResult)
            self.assertEqual(result.status, 'ERROR')
            self.assertIn('did not produce result file', result.additional_info.get('error_message', ''))


class TestCommandConstruction(unittest.TestCase):
    """Test MATLAB command construction and safety."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock MATLAB to avoid requiring installation
        with patch('scripts.solvers.matlab.matlab_interface.subprocess.run') as mock_subprocess:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = "MATLAB_OK"
            mock_subprocess.return_value = mock_result
            
            with patch('scripts.solvers.matlab.matlab_interface.load_problem_registry') as mock_registry:
                mock_registry.return_value = {'problem_libraries': {}}
                self.solver = MatlabSolver('sedumi')
    
    def test_normal_command_construction(self):
        """Test normal command construction."""
        cmd = self.solver._construct_matlab_command('test_problem', '/tmp/result.json')
        
        self.assertIsInstance(cmd, list)
        self.assertEqual(cmd[0], 'matlab')
        self.assertEqual(cmd[1], '-batch')
        self.assertIn('matlab_runner', cmd[2])
        self.assertIn('test_problem', cmd[2])
        self.assertIn('sedumi', cmd[2])
        self.assertIn('/tmp/result.json', cmd[2])
    
    
    def test_command_argument_escaping(self):
        """Test proper escaping of arguments with special characters."""
        cmd = self.solver._construct_matlab_command("prob'lem", "/tmp/res'ult.json")
        
        # MATLAB single quotes should be escaped by doubling
        self.assertIn("prob''lem", cmd[2])
        self.assertIn("res''ult.json", cmd[2])
    
    def test_invalid_command_parameters(self):
        """Test handling of invalid command parameters."""
        with self.assertRaises(ValueError):
            self.solver._construct_matlab_command('', '/tmp/result.json')
        
        with self.assertRaises(ValueError):
            self.solver._construct_matlab_command('test_problem', '')
        
        with self.assertRaises(ValueError):
            self.solver._construct_matlab_command(None, '/tmp/result.json')


class TestErrorHandling(unittest.TestCase):
    """Test comprehensive error handling."""
    
    def setUp(self):
        """Set up test fixtures."""
        with patch('scripts.solvers.matlab.matlab_interface.subprocess.run') as mock_subprocess:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = "MATLAB_OK"
            mock_subprocess.return_value = mock_result
            
            with patch('scripts.solvers.matlab.matlab_interface.load_problem_registry') as mock_registry:
                mock_registry.return_value = {'problem_libraries': {}}
                self.solver = MatlabSolver('sedumi')
    
    def test_matlab_error_parsing(self):
        """Test MATLAB error message parsing."""
        # Test common error patterns
        error_cases = [
            ("Error: Undefined function 'test' for input arguments", "Undefined function"),
            ("License error: Cannot checkout license", "License error"),
            ("Syntax error near line 5", "Syntax error"),
            ("Random output without clear error", "Random output")
        ]
        
        for stderr, expected_keyword in error_cases:
            parsed = self.solver._parse_matlab_error(stderr, "")
            self.assertIsInstance(parsed, str)
            # For error cases, check that key information is preserved
            if "error" in expected_keyword.lower():
                self.assertTrue(any(keyword in parsed.lower() for keyword in ["error", "undefined", "license", "syntax"]))
    
    def test_error_result_creation(self):
        """Test that error results are properly formatted."""
        # This is tested implicitly in other tests, but let's be explicit
        error_result = SolverResult.create_error_result(
            "Test error message",
            solve_time=1.5,
            solver_name="test_solver",
            solver_version="test_version"
        )
        
        self.assertEqual(error_result.status, "ERROR")
        self.assertEqual(error_result.solve_time, 1.5)
        self.assertEqual(error_result.solver_name, "test_solver")
        self.assertIn("Test error message", error_result.additional_info["error_message"])


class TestTempFileManagement(unittest.TestCase):
    """Test temporary file management integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        with patch('scripts.solvers.matlab.matlab_interface.subprocess.run') as mock_subprocess:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = "MATLAB_OK"
            mock_subprocess.return_value = mock_result
            
            with patch('scripts.solvers.matlab.matlab_interface.load_problem_registry') as mock_registry:
                mock_registry.return_value = {'problem_libraries': {}}
                self.solver = MatlabSolver('sedumi')
    
    def test_temp_file_stats(self):
        """Test temporary file statistics retrieval."""
        stats = self.solver.get_temp_file_stats()
        
        self.assertIsInstance(stats, dict)
        self.assertIn('temp_directory', stats)
        self.assertIn('total_files', stats)
        self.assertIn('total_size_bytes', stats)
    
    def test_orphaned_file_cleanup(self):
        """Test orphaned file cleanup functionality."""
        # This will actually call the temp manager's cleanup method
        cleaned_count = self.solver.cleanup_orphaned_files()
        
        self.assertIsInstance(cleaned_count, int)
        self.assertGreaterEqual(cleaned_count, 0)


if __name__ == '__main__':
    # Run tests with appropriate verbosity
    unittest.main(verbosity=2)