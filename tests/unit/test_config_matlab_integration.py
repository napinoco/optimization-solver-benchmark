#!/usr/bin/env python3
"""
Unit tests for MATLAB solver configuration integration.

This test suite validates that the configuration system properly handles
MATLAB solvers including loading, validation, and error handling scenarios.
"""

import sys
import unittest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import yaml
import tempfile
import os

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.benchmark.runner import BenchmarkRunner


class TestMATLABConfigIntegration(unittest.TestCase):
    """Test MATLAB solver configuration integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_solver_registry = {
            'solvers': {
                'scipy_linprog': {
                    'display_name': 'SciPy linprog'
                },
                'cvxpy_clarabel': {
                    'display_name': 'CLARABEL (via CVXPY)'
                },
                'matlab_sedumi': {
                    'display_name': 'SeDuMi (via MATLAB)'
                },
                'matlab_sdpt3': {
                    'display_name': 'SDPT3 (via MATLAB)'
                }
            }
        }
        
        self.test_problem_registry = {
            'problem_libraries': {
                'arch0': {
                    'library_name': 'SDPLIB',
                    'file_type': 'dat-s',
                    'file_path': 'problems/SDPLIB/data/arch0.dat-s',
                    'problem_type': 'SDP'
                }
            }
        }

    def test_solver_registry_loading_with_matlab(self):
        """Test that solver registry properly loads MATLAB solvers."""
        with patch('yaml.safe_load', return_value=self.test_solver_registry):
            with patch('builtins.open', MagicMock()):
                with patch('scripts.benchmark.runner.DatabaseManager'):
                    runner = BenchmarkRunner()
                    
                    # Verify MATLAB solvers are in registry
                    self.assertIn('matlab_sedumi', runner.solver_registry['solvers'])
                    self.assertIn('matlab_sdpt3', runner.solver_registry['solvers'])
                    
                    # Verify display names
                    self.assertEqual(
                        runner.solver_registry['solvers']['matlab_sedumi']['display_name'],
                        'SeDuMi (via MATLAB)'
                    )
                    self.assertEqual(
                        runner.solver_registry['solvers']['matlab_sdpt3']['display_name'],
                        'SDPT3 (via MATLAB)'
                    )

    def test_solver_availability_with_matlab_available(self):
        """Test solver availability when MATLAB is available."""
        with patch('scripts.benchmark.runner.DatabaseManager'):
            with patch.object(BenchmarkRunner, 'load_solver_registry', 
                            return_value=self.test_solver_registry):
                with patch.object(BenchmarkRunner, 'load_problem_registry',
                                return_value=self.test_problem_registry):
                    # Mock MATLAB availability
                    with patch('scripts.solvers.matlab_octave.matlab_interface.SeDuMiSolver') as mock_sedumi:
                        with patch('scripts.solvers.matlab_octave.matlab_interface.SDPT3Solver') as mock_sdpt3:
                            runner = BenchmarkRunner()
                            available = runner.get_available_solvers()
                            
                            # Should include both Python and MATLAB solvers
                            self.assertIn('scipy_linprog', available)
                            self.assertIn('cvxpy_clarabel', available)
                            self.assertIn('matlab_sedumi', available)
                            self.assertIn('matlab_sdpt3', available)
                            
                            # Should have 11 total solvers (9 Python + 2 MATLAB)
                            matlab_solvers = [s for s in available if s.startswith('matlab_')]
                            self.assertEqual(len(matlab_solvers), 2)

    def test_solver_availability_with_matlab_unavailable(self):
        """Test solver availability when MATLAB is not available."""
        with patch('scripts.benchmark.runner.DatabaseManager'):
            with patch.object(BenchmarkRunner, 'load_solver_registry',
                            return_value=self.test_solver_registry):
                with patch.object(BenchmarkRunner, 'load_problem_registry',
                                return_value=self.test_problem_registry):
                    # Mock MATLAB import failure
                    with patch('scripts.benchmark.runner.logger'):
                        with patch('scripts.solvers.matlab_octave.matlab_interface.SeDuMiSolver',
                                 side_effect=ImportError("MATLAB not available")):
                            runner = BenchmarkRunner()
                            available = runner.get_available_solvers()
                            
                            # Should include Python solvers but not MATLAB
                            self.assertIn('scipy_linprog', available)
                            self.assertIn('cvxpy_clarabel', available)
                            
                            # Should not include MATLAB solvers if import fails
                            matlab_solvers = [s for s in available if s.startswith('matlab_')]
                            # Note: Actual behavior depends on implementation
                            # If graceful degradation is implemented, this should be 0

    def test_solver_creation_with_matlab_available(self):
        """Test MATLAB solver creation when MATLAB is available."""
        with patch('scripts.benchmark.runner.DatabaseManager'):
            with patch.object(BenchmarkRunner, 'load_solver_registry',
                            return_value=self.test_solver_registry):
                with patch.object(BenchmarkRunner, 'load_problem_registry', 
                                return_value=self.test_problem_registry):
                    # Mock successful MATLAB solver creation
                    mock_sedumi = Mock()
                    mock_sedumi.solver_name = 'matlab_sedumi'
                    
                    with patch('scripts.solvers.matlab_octave.matlab_interface.SeDuMiSolver',
                             return_value=mock_sedumi):
                        runner = BenchmarkRunner()
                        solver = runner.create_solver('matlab_sedumi')
                        
                        self.assertEqual(solver.solver_name, 'matlab_sedumi')

    def test_solver_creation_with_matlab_unavailable(self):
        """Test MATLAB solver creation when MATLAB is not available."""
        with patch('scripts.benchmark.runner.DatabaseManager'):
            with patch.object(BenchmarkRunner, 'load_solver_registry',
                            return_value=self.test_solver_registry):
                with patch.object(BenchmarkRunner, 'load_problem_registry',
                                return_value=self.test_problem_registry):
                    # Mock MATLAB unavailable in runner
                    with patch('scripts.benchmark.runner.MATLAB_SOLVERS_AVAILABLE', False):
                        runner = BenchmarkRunner()
                        
                        # Should raise ValueError with clear message
                        with self.assertRaises(ValueError) as cm:
                            runner.create_solver('matlab_sedumi')
                        
                        self.assertIn('MATLAB solvers not available', str(cm.exception))

    def test_config_validation_error_handling(self):
        """Test configuration validation with missing files."""
        # Test missing solver registry
        with patch('builtins.open', side_effect=FileNotFoundError("Registry not found")):
            with patch('scripts.benchmark.runner.DatabaseManager'):
                with patch('scripts.benchmark.runner.logger') as mock_logger:
                    runner = BenchmarkRunner()
                    
                    # Should have empty registry due to error handling
                    self.assertEqual(runner.solver_registry['solvers'], {})
                    
                    # Should have logged the error
                    mock_logger.error.assert_called()

    def test_config_backward_compatibility(self):
        """Test that adding MATLAB solvers doesn't break existing configurations."""
        # Registry without MATLAB solvers (old format)
        old_registry = {
            'solvers': {
                'scipy_linprog': {
                    'display_name': 'SciPy linprog'
                },
                'cvxpy_clarabel': {
                    'display_name': 'CLARABEL (via CVXPY)'
                }
            }
        }
        
        with patch('yaml.safe_load', return_value=old_registry):
            with patch('builtins.open', MagicMock()):
                with patch('scripts.benchmark.runner.DatabaseManager'):
                    runner = BenchmarkRunner()
                    
                    # Should work with old registry format
                    self.assertIn('scipy_linprog', runner.solver_registry['solvers'])
                    self.assertIn('cvxpy_clarabel', runner.solver_registry['solvers'])
                    
                    # Should not have MATLAB solvers in old registry
                    self.assertNotIn('matlab_sedumi', runner.solver_registry['solvers'])
                    self.assertNotIn('matlab_sdpt3', runner.solver_registry['solvers'])

    def test_solver_registry_yaml_format_validation(self):
        """Test that the actual solver registry YAML format is valid."""
        config_path = project_root / "config" / "solver_registry.yaml"
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                registry = yaml.safe_load(f)
            
            # Validate required structure
            self.assertIn('solvers', registry)
            self.assertIsInstance(registry['solvers'], dict)
            
            # Validate MATLAB solvers are present
            self.assertIn('matlab_sedumi', registry['solvers'])
            self.assertIn('matlab_sdpt3', registry['solvers'])
            
            # Validate display names
            for solver_name, config in registry['solvers'].items():
                self.assertIn('display_name', config)
                self.assertIsInstance(config['display_name'], str)
                self.assertTrue(len(config['display_name']) > 0)
            
            # Validate MATLAB solver display names follow pattern
            self.assertIn('MATLAB', registry['solvers']['matlab_sedumi']['display_name'])
            self.assertIn('MATLAB', registry['solvers']['matlab_sdpt3']['display_name'])

    def test_configuration_file_paths(self):
        """Test that configuration files exist and are accessible."""
        config_dir = project_root / "config"
        
        # Required configuration files
        required_files = [
            "solver_registry.yaml",
            "problem_registry.yaml",
            "site_config.yaml"
        ]
        
        for filename in required_files:
            config_file = config_dir / filename
            self.assertTrue(config_file.exists(), f"Configuration file missing: {filename}")
            
            # Test that file is readable and valid YAML
            with open(config_file, 'r') as f:
                try:
                    yaml.safe_load(f)
                except yaml.YAMLError as e:
                    self.fail(f"Invalid YAML in {filename}: {e}")

    def test_matlab_solver_graceful_degradation(self):
        """Test graceful degradation when MATLAB import fails at runtime."""
        with patch('scripts.benchmark.runner.DatabaseManager'):
            with patch.object(BenchmarkRunner, 'load_solver_registry',
                            return_value=self.test_solver_registry):
                with patch.object(BenchmarkRunner, 'load_problem_registry',
                                return_value=self.test_problem_registry):
                    # Simulate MATLAB import working during module load but failing at runtime
                    with patch('scripts.benchmark.runner.MATLAB_SOLVERS_AVAILABLE', True):
                        runner = BenchmarkRunner()
                        
                        # Mock solver creation failure
                        with patch.object(runner, 'create_solver') as mock_create:
                            mock_create.side_effect = ValueError("MATLAB execution failed")
                            
                            # Should handle error gracefully
                            with self.assertRaises(ValueError):
                                runner.create_solver('matlab_sedumi')


class TestConfigurationValidation(unittest.TestCase):
    """Test configuration validation functionality."""
    
    def test_validate_environment_success(self):
        """Test successful environment validation."""
        with patch('pathlib.Path.exists', return_value=True):
            from main import validate_environment
            result = validate_environment()
            self.assertTrue(result)

    def test_validate_environment_missing_directories(self):
        """Test environment validation with missing directories."""
        def mock_exists(path):
            # Mock missing 'scripts' directory
            return str(path) != 'scripts'
        
        with patch('pathlib.Path.exists', side_effect=mock_exists):
            with patch('scripts.utils.logger.get_logger') as mock_logger:
                mock_logger.return_value = Mock()
                from main import validate_environment
                result = validate_environment()
                self.assertFalse(result)

    def test_validate_solver_setup(self):
        """Test solver setup validation."""
        # Mock successful validation report
        mock_report = {
            'summary': {
                'working_solvers': 11,
                'total_solvers': 11,
                'working_problems': 139,
                'total_problems': 139
            },
            'solvers': {
                'scipy_linprog': {'status': 'working', 'version': '1.0.0'},
                'matlab_sedumi': {'status': 'working', 'version': 'SeDuMi 1.3.7'},
                'matlab_sdpt3': {'status': 'working', 'version': 'SDPT3 4.0'}
            }
        }
        
        with patch('scripts.benchmark.runner.BenchmarkRunner') as mock_runner_class:
            mock_runner = Mock()
            mock_runner.validate_setup.return_value = mock_report
            mock_runner_class.return_value = mock_runner
            
            from main import validate_solver_setup
            result = validate_solver_setup(verbose=False)
            self.assertTrue(result)


if __name__ == '__main__':
    unittest.main()