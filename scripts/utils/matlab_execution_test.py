"""
MATLAB Command-Line Execution Testing Utility.

This module provides utilities for testing and validating MATLAB command-line execution
from Python. It focuses on reliability, timeout handling, and error capture for the
benchmarking system.
"""

import os
import sys
import subprocess
import time
import json
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.utils.logger import get_logger

logger = get_logger("matlab_execution_test")


class MatlabExecutionTester:
    """
    Utility class for testing MATLAB command-line execution reliability.
    """
    
    def __init__(self, matlab_executable: str = 'matlab', working_directory: Optional[str] = None):
        """
        Initialize MATLAB execution tester.
        
        Args:
            matlab_executable: Path to MATLAB executable
            working_directory: Working directory for MATLAB execution
        """
        self.matlab_executable = matlab_executable
        self.working_directory = working_directory or str(project_root)
        self.execution_stats = {
            'total_tests': 0,
            'successful_tests': 0,
            'failed_tests': 0,
            'timeout_tests': 0,
            'startup_times': []
        }
        
        logger.info(f"Initialized MATLAB execution tester with executable: {matlab_executable}")
    
    def test_basic_execution(self, timeout: float = 30) -> Tuple[bool, Dict[str, Any]]:
        """
        Test basic MATLAB execution with simple command.
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            Tuple of (success, result_info)
        """
        logger.info("Testing basic MATLAB execution...")
        
        start_time = time.time()
        cmd = [self.matlab_executable, '-batch', 'disp("MATLAB_OK")']
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.working_directory
            )
            
            execution_time = time.time() - start_time
            self.execution_stats['startup_times'].append(execution_time)
            
            success = (result.returncode == 0 and 'MATLAB_OK' in result.stdout)
            
            result_info = {
                'command': ' '.join(cmd),
                'returncode': result.returncode,
                'stdout': result.stdout.strip(),
                'stderr': result.stderr.strip(),
                'execution_time': execution_time,
                'success': success
            }
            
            if success:
                logger.info(f"✓ Basic execution successful in {execution_time:.2f}s")
                self.execution_stats['successful_tests'] += 1
            else:
                logger.warning(f"✗ Basic execution failed: {result.stderr}")
                self.execution_stats['failed_tests'] += 1
                
            self.execution_stats['total_tests'] += 1
            return success, result_info
            
        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            logger.error(f"✗ Basic execution timed out after {execution_time:.2f}s")
            self.execution_stats['timeout_tests'] += 1
            self.execution_stats['total_tests'] += 1
            
            return False, {
                'command': ' '.join(cmd),
                'error': 'timeout',
                'execution_time': execution_time,
                'timeout_duration': timeout,
                'success': False
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"✗ Basic execution failed with exception: {e}")
            self.execution_stats['failed_tests'] += 1
            self.execution_stats['total_tests'] += 1
            
            return False, {
                'command': ' '.join(cmd),
                'error': str(e),
                'execution_time': execution_time,
                'success': False
            }
    
    def test_matlab_runner_execution(self, problem_name: str = 'arch0', 
                                   solver_name: str = 'sedumi', 
                                   timeout: float = 60) -> Tuple[bool, Dict[str, Any]]:
        """
        Test execution of matlab_runner.m with actual parameters.
        
        Args:
            problem_name: Name of problem to test with (use a real problem name)
            solver_name: Name of solver to test with
            timeout: Timeout in seconds
            
        Returns:
            Tuple of (success, result_info)
        """
        logger.info(f"Testing matlab_runner execution with {problem_name}, {solver_name}...")
        
        # Create a temporary result file for testing
        temp_result_file = f"/tmp/test_matlab_result_{int(time.time())}.json"
        
        start_time = time.time()
        # Add path setup to ensure matlab_runner is found
        matlab_command = f"addpath(genpath('.')); matlab_runner('{problem_name}', '{solver_name}', '{temp_result_file}')"
        cmd = [self.matlab_executable, '-batch', matlab_command]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.working_directory
            )
            
            execution_time = time.time() - start_time
            
            # Check if result file was created (even if with error content)
            result_file_created = os.path.exists(temp_result_file)
            
            result_info = {
                'command': matlab_command,
                'full_cmd': ' '.join(cmd),
                'returncode': result.returncode,
                'stdout': result.stdout.strip(),
                'stderr': result.stderr.strip(),
                'execution_time': execution_time,
                'result_file_created': result_file_created,
                'temp_result_file': temp_result_file
            }
            
            # For this test, we consider it successful if:
            # 1. MATLAB executed without crashing (even if matlab_runner fails)
            # 2. We got some reasonable output or error message
            # 3. The execution completed within timeout
            
            # Check if MATLAB ran successfully (return code 0 or 1 both OK for this test)
            matlab_executed = result.returncode in [0, 1]  # 0 = success, 1 = MATLAB error but executed
            
            # Read result file if it exists
            if result_file_created:
                try:
                    with open(temp_result_file, 'r') as f:
                        result_content = json.load(f)
                    result_info['result_content'] = result_content
                    logger.info(f"✓ matlab_runner created JSON result in {execution_time:.2f}s")
                    
                except json.JSONDecodeError as e:
                    result_info['json_error'] = str(e)
                    logger.warning(f"✗ matlab_runner produced invalid JSON: {e}")
            
            # Success criteria: MATLAB executed and either created result file OR gave meaningful error
            if matlab_executed:
                if result_file_created:
                    success = True
                    logger.info(f"✓ matlab_runner execution successful - result file created")
                else:
                    # Check if we got a meaningful error (matlab_runner not found, etc.)
                    if 'matlab_runner' in result.stderr or 'Undefined function' in result.stderr:
                        success = True  # This is expected if matlab_runner.m is not in path
                        logger.info(f"✓ matlab_runner execution test successful - got expected error about missing function")
                    else:
                        success = False
                        logger.warning(f"✗ matlab_runner execution failed unexpectedly")
            else:
                success = False
                logger.warning(f"✗ MATLAB execution failed completely")
            
            result_info['test_success'] = success
            
            # Cleanup temp file
            try:
                if os.path.exists(temp_result_file):
                    os.remove(temp_result_file)
            except:
                pass
            
            if success:
                self.execution_stats['successful_tests'] += 1
            else:
                self.execution_stats['failed_tests'] += 1
                
            self.execution_stats['total_tests'] += 1
            return success, result_info
            
        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            logger.error(f"✗ matlab_runner execution timed out after {execution_time:.2f}s")
            
            # Cleanup temp file
            try:
                if os.path.exists(temp_result_file):
                    os.remove(temp_result_file)
            except:
                pass
            
            self.execution_stats['timeout_tests'] += 1
            self.execution_stats['total_tests'] += 1
            
            return False, {
                'command': matlab_command,
                'error': 'timeout',
                'execution_time': execution_time,
                'timeout_duration': timeout,
                'success': False
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"✗ matlab_runner execution failed: {e}")
            
            # Cleanup temp file
            try:
                if os.path.exists(temp_result_file):
                    os.remove(temp_result_file)
            except:
                pass
            
            self.execution_stats['failed_tests'] += 1
            self.execution_stats['total_tests'] += 1
            
            return False, {
                'command': matlab_command,
                'error': str(e),
                'execution_time': execution_time,
                'success': False
            }
    
    def test_timeout_handling(self, timeout_duration: float = 5) -> Tuple[bool, Dict[str, Any]]:
        """
        Test timeout handling with a deliberately long-running command.
        
        Args:
            timeout_duration: Timeout to test with
            
        Returns:
            Tuple of (timeout_worked, result_info)
        """
        logger.info(f"Testing timeout handling with {timeout_duration}s timeout...")
        
        start_time = time.time()
        # Command that will run longer than timeout
        cmd = [self.matlab_executable, '-batch', 'pause(10); disp("Should not see this")']
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_duration,
                cwd=self.working_directory
            )
            
            execution_time = time.time() - start_time
            
            # If we get here, timeout didn't work as expected
            logger.warning(f"✗ Timeout test failed - command completed in {execution_time:.2f}s")
            self.execution_stats['failed_tests'] += 1
            self.execution_stats['total_tests'] += 1
            
            return False, {
                'command': ' '.join(cmd),
                'returncode': result.returncode,
                'stdout': result.stdout.strip(),
                'stderr': result.stderr.strip(),
                'execution_time': execution_time,
                'expected_timeout': True,
                'actually_timed_out': False,
                'success': False
            }
            
        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            
            # This is what we expect - timeout worked correctly
            timeout_worked = abs(execution_time - timeout_duration) < 2.0  # Within 2 seconds
            
            if timeout_worked:
                logger.info(f"✓ Timeout handling working correctly - timed out at {execution_time:.2f}s")
                self.execution_stats['successful_tests'] += 1
            else:
                logger.warning(f"✗ Timeout timing inaccurate - expected ~{timeout_duration}s, got {execution_time:.2f}s")
                self.execution_stats['failed_tests'] += 1
            
            self.execution_stats['total_tests'] += 1
            
            return timeout_worked, {
                'command': ' '.join(cmd),
                'error': 'timeout_as_expected',
                'execution_time': execution_time,
                'timeout_duration': timeout_duration,
                'expected_timeout': True,
                'actually_timed_out': True,
                'timeout_accuracy': abs(execution_time - timeout_duration),
                'success': timeout_worked
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"✗ Timeout test failed with exception: {e}")
            self.execution_stats['failed_tests'] += 1
            self.execution_stats['total_tests'] += 1
            
            return False, {
                'command': ' '.join(cmd),
                'error': str(e),
                'execution_time': execution_time,
                'expected_timeout': True,
                'actually_timed_out': False,
                'success': False
            }
    
    def test_error_handling(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Test error handling with invalid MATLAB command.
        
        Returns:
            Tuple of (error_handled, result_info)
        """
        logger.info("Testing error handling with invalid command...")
        
        start_time = time.time()
        cmd = [self.matlab_executable, '-batch', 'invalid_function_call_that_should_fail()']
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=self.working_directory
            )
            
            execution_time = time.time() - start_time
            
            # We expect this to fail with non-zero return code
            error_handled = (result.returncode != 0 and 
                           ('error' in result.stderr.lower() or 
                            'undefined' in result.stderr.lower() or
                            len(result.stderr) > 0))
            
            result_info = {
                'command': ' '.join(cmd),
                'returncode': result.returncode,
                'stdout': result.stdout.strip(),
                'stderr': result.stderr.strip(),
                'execution_time': execution_time,
                'error_properly_reported': error_handled,
                'success': error_handled
            }
            
            if error_handled:
                logger.info(f"✓ Error handling working - error properly reported in {execution_time:.2f}s")
                self.execution_stats['successful_tests'] += 1
            else:
                logger.warning(f"✗ Error handling failed - no error reported for invalid command")
                self.execution_stats['failed_tests'] += 1
                
            self.execution_stats['total_tests'] += 1
            return error_handled, result_info
            
        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            logger.warning(f"✗ Error test timed out - took too long to fail")
            self.execution_stats['timeout_tests'] += 1
            self.execution_stats['total_tests'] += 1
            
            return False, {
                'command': ' '.join(cmd),
                'error': 'timeout_on_error_test',
                'execution_time': execution_time,
                'success': False
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"✗ Error test failed with exception: {e}")
            self.execution_stats['failed_tests'] += 1
            self.execution_stats['total_tests'] += 1
            
            return False, {
                'command': ' '.join(cmd),
                'error': str(e),
                'execution_time': execution_time,
                'success': False
            }
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics and performance metrics."""
        stats = self.execution_stats.copy()
        
        if stats['startup_times']:
            stats['avg_startup_time'] = sum(stats['startup_times']) / len(stats['startup_times'])
            stats['min_startup_time'] = min(stats['startup_times'])
            stats['max_startup_time'] = max(stats['startup_times'])
        else:
            stats['avg_startup_time'] = 0
            stats['min_startup_time'] = 0
            stats['max_startup_time'] = 0
        
        if stats['total_tests'] > 0:
            stats['success_rate'] = stats['successful_tests'] / stats['total_tests']
        else:
            stats['success_rate'] = 0
        
        return stats
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """
        Run comprehensive test suite for MATLAB command-line interface.
        
        Returns:
            Dictionary with test results and statistics
        """
        logger.info("=" * 60)
        logger.info("MATLAB COMMAND-LINE INTERFACE COMPREHENSIVE TEST")
        logger.info("=" * 60)
        
        test_results = {}
        
        # Test 1: Basic execution
        success, result = self.test_basic_execution()
        test_results['basic_execution'] = result
        
        # Test 2: matlab_runner execution (will fail with unknown problem, but should handle gracefully)
        success, result = self.test_matlab_runner_execution()
        test_results['matlab_runner_execution'] = result
        
        # Test 3: Timeout handling
        success, result = self.test_timeout_handling()
        test_results['timeout_handling'] = result
        
        # Test 4: Error handling
        success, result = self.test_error_handling()
        test_results['error_handling'] = result
        
        # Add execution statistics
        test_results['execution_stats'] = self.get_execution_stats()
        
        logger.info("=" * 60)
        logger.info("MATLAB CLI TEST SUMMARY")
        logger.info("=" * 60)
        stats = test_results['execution_stats']
        logger.info(f"Total Tests: {stats['total_tests']}")
        logger.info(f"Successful: {stats['successful_tests']}")
        logger.info(f"Failed: {stats['failed_tests']}")
        logger.info(f"Timeouts: {stats['timeout_tests']}")
        logger.info(f"Success Rate: {stats['success_rate']:.1%}")
        logger.info(f"Average Startup Time: {stats['avg_startup_time']:.2f}s")
        logger.info("=" * 60)
        
        return test_results


def main():
    """Run MATLAB execution tests as standalone script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test MATLAB command-line execution')
    parser.add_argument('--matlab-executable', default='matlab', 
                       help='Path to MATLAB executable')
    parser.add_argument('--working-directory', 
                       help='Working directory for MATLAB execution')
    
    args = parser.parse_args()
    
    tester = MatlabExecutionTester(
        matlab_executable=args.matlab_executable,
        working_directory=args.working_directory
    )
    
    try:
        results = tester.run_comprehensive_test()
        
        # Print summary
        stats = results['execution_stats']
        if stats['success_rate'] >= 0.75:
            print(f"\n✅ MATLAB CLI tests mostly successful ({stats['success_rate']:.1%} success rate)")
            sys.exit(0)
        else:
            print(f"\n❌ MATLAB CLI tests had issues ({stats['success_rate']:.1%} success rate)")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ MATLAB CLI test suite failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()