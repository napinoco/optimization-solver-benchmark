#!/usr/bin/env python3
"""
Integration tests for MATLAB command-line interface.

This module tests the reliability and robustness of MATLAB command execution
from Python for the benchmarking system.
"""

import os
import sys
import time
import tempfile
import unittest
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from tests.test_utils.matlab_execution_tester import MatlabExecutionTester
from scripts.utils.logger import get_logger

logger = get_logger("test_matlab_cli")


class TestMatlabCLI(unittest.TestCase):
    """Integration tests for MATLAB command-line interface."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test class with MATLAB execution tester."""
        cls.tester = MatlabExecutionTester()
        cls.matlab_available = False
        
        # Test if MATLAB is available
        try:
            success, _ = cls.tester.test_basic_execution()
            cls.matlab_available = success
            if not success:
                logger.warning("MATLAB not available - some tests will be skipped")
        except Exception as e:
            logger.warning(f"MATLAB availability test failed: {e}")
    
    def setUp(self):
        """Set up individual test."""
        if not self.matlab_available:
            self.skipTest("MATLAB not available")
    
    def test_basic_matlab_execution(self):
        """Test basic MATLAB command execution."""
        logger.info("Testing basic MATLAB execution...")
        
        success, result = self.tester.test_basic_execution(timeout=30)
        
        self.assertTrue(success, f"Basic MATLAB execution failed: {result}")
        self.assertEqual(result['returncode'], 0, "MATLAB should exit with code 0")
        self.assertIn('MATLAB_OK', result['stdout'], "Expected output not found")
        self.assertLess(result['execution_time'], 30, "Execution should complete within timeout")
    
    def test_matlab_startup_performance(self):
        """Test MATLAB startup performance and consistency."""
        logger.info("Testing MATLAB startup performance...")
        
        startup_times = []
        num_tests = 3
        
        for i in range(num_tests):
            success, result = self.tester.test_basic_execution(timeout=30)
            self.assertTrue(success, f"Startup test {i+1} failed")
            startup_times.append(result['execution_time'])
        
        avg_startup = sum(startup_times) / len(startup_times)
        max_startup = max(startup_times)
        
        logger.info(f"Startup times: {startup_times}")
        logger.info(f"Average startup: {avg_startup:.2f}s, Max: {max_startup:.2f}s")
        
        # MATLAB startup should be reasonable (under 15 seconds)
        self.assertLess(max_startup, 15.0, f"MATLAB startup too slow: {max_startup:.2f}s")
        
        # Consistency check - no startup should be more than 3x the average
        for startup_time in startup_times:
            self.assertLess(startup_time, avg_startup * 3, 
                          f"Inconsistent startup time: {startup_time:.2f}s vs avg {avg_startup:.2f}s")
    
    def test_timeout_handling(self):
        """Test that timeout handling works correctly."""
        logger.info("Testing timeout handling...")
        
        timeout_worked, result = self.tester.test_timeout_handling(timeout_duration=5)
        
        self.assertTrue(timeout_worked, f"Timeout handling failed: {result}")
        self.assertTrue(result['actually_timed_out'], "Command should have timed out")
        self.assertLess(result['timeout_accuracy'], 2.0, 
                       f"Timeout timing inaccurate: {result['timeout_accuracy']:.2f}s")
    
    def test_error_handling(self):
        """Test that MATLAB errors are properly captured and reported."""
        logger.info("Testing error handling...")
        
        error_handled, result = self.tester.test_error_handling()
        
        self.assertTrue(error_handled, f"Error handling failed: {result}")
        self.assertNotEqual(result['returncode'], 0, "Invalid command should return non-zero code")
        self.assertTrue(len(result['stderr']) > 0, "Error message should be captured in stderr")
    
    def test_working_directory(self):
        """Test that MATLAB executes in the correct working directory."""
        logger.info("Testing working directory handling...")
        
        # Test with project root as working directory
        tester_with_wd = MatlabExecutionTester(working_directory=str(project_root))
        
        start_time = time.time()
        cmd_parts = ['matlab', '-batch', 'disp(pwd)']
        
        try:
            import subprocess
            result = subprocess.run(
                cmd_parts,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(project_root)
            )
            
            execution_time = time.time() - start_time
            
            if result.returncode == 0:
                # Check that pwd output contains our project directory
                current_dir = result.stdout.strip()
                logger.info(f"MATLAB working directory: {current_dir}")
                
                # The working directory should be related to our project
                self.assertTrue(
                    str(project_root) in current_dir or 
                    os.path.basename(str(project_root)) in current_dir,
                    f"Working directory issue: expected {project_root}, got {current_dir}"
                )
            else:
                self.fail(f"Working directory test failed: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            self.fail("Working directory test timed out")
        except Exception as e:
            self.fail(f"Working directory test failed with exception: {e}")
    
    def test_matlab_runner_execution(self):
        """Test execution of matlab_runner.m (will fail gracefully with unknown problem)."""
        logger.info("Testing matlab_runner execution...")
        
        # This should fail gracefully since 'test_problem' doesn't exist
        success, result = self.tester.test_matlab_runner_execution(
            problem_name='test_problem',
            solver_name='sedumi',
            timeout=60
        )
        
        # We expect this to complete execution (even if it results in an error)
        # The important thing is that it doesn't crash or hang
        self.assertIsNotNone(result, "Should get a result from matlab_runner test")
        self.assertIn('execution_time', result, "Should measure execution time")
        self.assertLess(result['execution_time'], 60, "Should complete within timeout")
        
        # Check if result file was created (even with error content)
        if 'result_file_created' in result:
            # If a result file was created, that's good - means matlab_runner ran
            if result['result_file_created'] and 'result_content' in result:
                logger.info("matlab_runner executed and produced JSON result")
                # Check that we got some kind of result structure
                content = result['result_content']
                self.assertIsInstance(content, dict, "Result should be a JSON object")
    
    def test_concurrent_execution(self):
        """Test concurrent MATLAB execution to check for conflicts."""
        logger.info("Testing concurrent MATLAB execution...")
        
        import threading
        import queue
        
        results_queue = queue.Queue()
        num_concurrent = 3
        
        def run_matlab_test(test_id):
            """Run a MATLAB test and put result in queue."""
            try:
                tester = MatlabExecutionTester()
                success, result = tester.test_basic_execution(timeout=30)
                results_queue.put((test_id, success, result))
            except Exception as e:
                results_queue.put((test_id, False, {'error': str(e)}))
        
        # Start concurrent threads
        threads = []
        for i in range(num_concurrent):
            thread = threading.Thread(target=run_matlab_test, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all to complete
        for thread in threads:
            thread.join(timeout=60)  # Give each thread up to 60 seconds
        
        # Collect results
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())
        
        # Verify all tests completed
        self.assertEqual(len(results), num_concurrent, 
                        f"Expected {num_concurrent} results, got {len(results)}")
        
        # Verify all tests succeeded
        successful_tests = [r for r in results if r[1]]  # r[1] is success flag
        success_rate = len(successful_tests) / len(results)
        
        logger.info(f"Concurrent test success rate: {success_rate:.1%}")
        
        # At least 80% should succeed (allowing for some timing issues)
        self.assertGreaterEqual(success_rate, 0.8, 
                               f"Concurrent execution success rate too low: {success_rate:.1%}")
    
    def test_argument_handling(self):
        """Test handling of various argument types and special characters."""
        logger.info("Testing argument handling...")
        
        # Test with different types of arguments
        test_cases = [
            ("Simple string", "disp('Hello World')"),
            ("String with spaces", "disp('Hello MATLAB World')"),
            ("Numbers", "disp(42)"),
            ("Mathematical expression", "disp(2+2)"),
        ]
        
        for test_name, matlab_cmd in test_cases:
            with self.subTest(test_case=test_name):
                start_time = time.time()
                
                try:
                    import subprocess
                    result = subprocess.run(
                        ['matlab', '-batch', matlab_cmd],
                        capture_output=True,
                        text=True,
                        timeout=30,
                        cwd=str(project_root)
                    )
                    
                    execution_time = time.time() - start_time
                    
                    # Should execute without crashing
                    self.assertIsNotNone(result.returncode, 
                                       f"Command should complete: {test_name}")
                    
                    # Should not timeout
                    self.assertLess(execution_time, 30, 
                                  f"Command should not timeout: {test_name}")
                    
                    logger.debug(f"{test_name}: returncode={result.returncode}, "
                               f"time={execution_time:.2f}s")
                    
                except subprocess.TimeoutExpired:
                    self.fail(f"Argument test timed out: {test_name}")
                except Exception as e:
                    self.fail(f"Argument test failed: {test_name}, error: {e}")


class TestMatlabCLIComprehensive(unittest.TestCase):
    """Comprehensive integration test using the MatlabExecutionTester."""
    
    def test_comprehensive_matlab_cli(self):
        """Run comprehensive MATLAB CLI test suite."""
        logger.info("Running comprehensive MATLAB CLI test...")
        
        tester = MatlabExecutionTester()
        
        try:
            results = tester.run_comprehensive_test()
            
            # Check overall statistics
            stats = results['execution_stats']
            
            self.assertGreater(stats['total_tests'], 0, "Should run some tests")
            self.assertGreaterEqual(stats['success_rate'], 0.5, 
                                  f"Success rate too low: {stats['success_rate']:.1%}")
            
            # Check startup performance
            if stats['avg_startup_time'] > 0:
                self.assertLess(stats['avg_startup_time'], 15.0, 
                              f"Average startup time too slow: {stats['avg_startup_time']:.2f}s")
            
            logger.info(f"Comprehensive test completed with {stats['success_rate']:.1%} success rate")
            
        except Exception as e:
            self.fail(f"Comprehensive MATLAB CLI test failed: {e}")


def main():
    """Run MATLAB CLI integration tests."""
    # Configure logging for test output
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run tests
    unittest.main(verbosity=2)


if __name__ == "__main__":
    main()