#!/usr/bin/env python3
"""
Results Validation Module

This module provides validation functions to ensure benchmark results are reasonable
and to catch errors before they corrupt the analysis.
"""

import logging
import math
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

# Configure logger for this module
logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of validation checks"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    severity: str  # 'info', 'warning', 'error', 'critical'


class ResultValidator:
    """
    Validates benchmark results to ensure they are reasonable and catch potential errors.
    """
    
    def __init__(self, 
                 max_solve_time: float = 3600.0,  # 1 hour max
                 min_solve_time: float = 1e-6,    # 1 microsecond min
                 max_objective_magnitude: float = 1e12):
        """
        Initialize validator with configurable thresholds.
        
        Args:
            max_solve_time: Maximum reasonable solve time in seconds
            min_solve_time: Minimum reasonable solve time in seconds
            max_objective_magnitude: Maximum reasonable objective value magnitude
        """
        self.max_solve_time = max_solve_time
        self.min_solve_time = min_solve_time
        self.max_objective_magnitude = max_objective_magnitude
        
        # Valid solver statuses
        self.valid_statuses = {
            'optimal', 'feasible', 'infeasible', 'unbounded', 
            'error', 'timeout', 'unknown', 'interrupted'
        }
        
        logger.info(f"Validator initialized with max_time={max_solve_time}s, "
                   f"min_time={min_solve_time}s, max_obj_mag={max_objective_magnitude}")
    
    def validate_solve_time(self, solve_time: float, timeout: Optional[float] = None) -> ValidationResult:
        """
        Validate solve time is reasonable.
        
        Args:
            solve_time: Time taken to solve the problem
            timeout: Configured timeout for the solver (if available)
            
        Returns:
            ValidationResult with validation outcome
        """
        errors = []
        warnings = []
        
        # Check for invalid time values
        if math.isnan(solve_time) or math.isinf(solve_time):
            errors.append(f"Solve time is invalid: {solve_time}")
        elif solve_time < 0:
            errors.append(f"Solve time is negative: {solve_time}")
        elif solve_time < self.min_solve_time:
            warnings.append(f"Solve time suspiciously small: {solve_time}s (< {self.min_solve_time}s)")
        elif solve_time > self.max_solve_time:
            errors.append(f"Solve time exceeds maximum threshold: {solve_time}s (> {self.max_solve_time}s)")
        
        # Check against timeout if provided
        if timeout is not None and solve_time > timeout:
            errors.append(f"Solve time {solve_time}s exceeds configured timeout {timeout}s")
        
        # Check for suspiciously long times
        if timeout is not None and solve_time > 0.9 * timeout:
            warnings.append(f"Solve time {solve_time}s is close to timeout {timeout}s")
        
        severity = 'error' if errors else ('warning' if warnings else 'info')
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            severity=severity
        )
    
    def validate_solver_status(self, status: str) -> ValidationResult:
        """
        Validate solver status is a known value.
        
        Args:
            status: Status returned by the solver
            
        Returns:
            ValidationResult with validation outcome
        """
        errors = []
        warnings = []
        
        if not isinstance(status, str):
            errors.append(f"Solver status must be string, got {type(status)}: {status}")
        elif status.lower() not in self.valid_statuses:
            errors.append(f"Unknown solver status: '{status}'. Valid statuses: {sorted(self.valid_statuses)}")
        
        severity = 'error' if errors else ('warning' if warnings else 'info')
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            severity=severity
        )
    
    def validate_objective_value(self, objective_value: Optional[float], 
                                status: str) -> ValidationResult:
        """
        Validate objective value is reasonable.
        
        Args:
            objective_value: Objective value returned by solver
            status: Solver status to determine if objective should exist
            
        Returns:
            ValidationResult with validation outcome
        """
        errors = []
        warnings = []
        
        # Check if objective should exist based on status
        if status.lower() in {'optimal', 'feasible'}:
            if objective_value is None:
                errors.append(f"Objective value is None but solver status is '{status}'")
            elif math.isnan(objective_value):
                errors.append(f"Objective value is NaN with status '{status}'")
            elif math.isinf(objective_value):
                if status.lower() == 'optimal':
                    errors.append(f"Objective value is infinite but status is 'optimal'")
                else:
                    warnings.append(f"Objective value is infinite with status '{status}'")
            elif abs(objective_value) > self.max_objective_magnitude:
                warnings.append(f"Objective value magnitude is very large: {objective_value}")
        
        elif status.lower() in {'infeasible', 'error', 'timeout'}:
            if objective_value is not None and not (math.isnan(objective_value) or math.isinf(objective_value)):
                warnings.append(f"Objective value exists ({objective_value}) but status is '{status}'")
        
        severity = 'error' if errors else ('warning' if warnings else 'info')
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            severity=severity
        )
    
    def validate_result_consistency(self, results: List[Dict[str, Any]]) -> ValidationResult:
        """
        Validate consistency across multiple results.
        
        Args:
            results: List of result dictionaries to check for consistency
            
        Returns:
            ValidationResult with validation outcome
        """
        errors = []
        warnings = []
        
        if not results:
            return ValidationResult(True, [], [], 'info')
        
        # Group by (solver, problem) pairs
        solver_problem_results = {}
        for result in results:
            key = (result.get('solver_name', 'unknown'), result.get('problem_name', 'unknown'))
            if key not in solver_problem_results:
                solver_problem_results[key] = []
            solver_problem_results[key].append(result)
        
        # Check for suspicious variations in solve times for same (solver, problem)
        for (solver, problem), problem_results in solver_problem_results.items():
            if len(problem_results) < 2:
                continue
                
            optimal_times = [
                r['solve_time'] for r in problem_results 
                if r.get('status', '').lower() == 'optimal' and 
                isinstance(r.get('solve_time'), (int, float)) and 
                r['solve_time'] > 0
            ]
            
            if len(optimal_times) >= 2:
                min_time = min(optimal_times)
                max_time = max(optimal_times)
                
                # Check for large variations (more than 10x difference)
                if max_time > 10 * min_time:
                    warnings.append(
                        f"Large time variation for {solver} on {problem}: "
                        f"{min_time:.6f}s to {max_time:.6f}s (ratio: {max_time/min_time:.1f}x)"
                    )
        
        # Check for solvers that never succeed
        solver_success_rates = {}
        for result in results:
            solver = result.get('solver_name', 'unknown')
            if solver not in solver_success_rates:
                solver_success_rates[solver] = {'total': 0, 'successful': 0}
            
            solver_success_rates[solver]['total'] += 1
            if result.get('status', '').lower() in {'optimal', 'feasible'}:
                solver_success_rates[solver]['successful'] += 1
        
        for solver, stats in solver_success_rates.items():
            success_rate = stats['successful'] / stats['total'] if stats['total'] > 0 else 0
            if stats['total'] >= 3 and success_rate == 0:
                warnings.append(f"Solver '{solver}' has 0% success rate across {stats['total']} attempts")
            elif stats['total'] >= 5 and success_rate < 0.2:
                warnings.append(f"Solver '{solver}' has low success rate: {success_rate:.1%} across {stats['total']} attempts")
        
        severity = 'error' if errors else ('warning' if warnings else 'info')
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            severity=severity
        )
    
    def validate_single_result(self, result: Dict[str, Any], 
                             timeout: Optional[float] = None) -> ValidationResult:
        """
        Perform comprehensive validation on a single benchmark result.
        
        Args:
            result: Dictionary containing benchmark result data
            timeout: Configured timeout for the solver
            
        Returns:
            ValidationResult with overall validation outcome
        """
        all_errors = []
        all_warnings = []
        
        # Required fields check
        required_fields = ['solver_name', 'problem_name', 'solve_time', 'status']
        for field in required_fields:
            if field not in result:
                all_errors.append(f"Missing required field: {field}")
        
        if all_errors:  # Stop if missing required fields
            return ValidationResult(False, all_errors, all_warnings, 'error')
        
        # Validate individual components
        time_validation = self.validate_solve_time(result['solve_time'], timeout)
        all_errors.extend(time_validation.errors)
        all_warnings.extend(time_validation.warnings)
        
        status_validation = self.validate_solver_status(result['status'])
        all_errors.extend(status_validation.errors)
        all_warnings.extend(status_validation.warnings)
        
        if 'objective_value' in result:
            obj_validation = self.validate_objective_value(result['objective_value'], result['status'])
            all_errors.extend(obj_validation.errors)
            all_warnings.extend(obj_validation.warnings)
        
        # Log validation results
        if all_errors:
            logger.error(f"Validation failed for {result.get('solver_name')} on {result.get('problem_name')}: {all_errors}")
        elif all_warnings:
            logger.warning(f"Validation warnings for {result.get('solver_name')} on {result.get('problem_name')}: {all_warnings}")
        
        severity = 'error' if all_errors else ('warning' if all_warnings else 'info')
        return ValidationResult(
            is_valid=len(all_errors) == 0,
            errors=all_errors,
            warnings=all_warnings,
            severity=severity
        )
    
    def validate_batch_results(self, results: List[Dict[str, Any]], 
                             timeout: Optional[float] = None) -> Tuple[List[ValidationResult], ValidationResult]:
        """
        Validate a batch of results, both individually and for consistency.
        
        Args:
            results: List of result dictionaries
            timeout: Configured timeout for the solver
            
        Returns:
            Tuple of (individual_validations, consistency_validation)
        """
        individual_results = []
        
        # Validate each result individually
        for result in results:
            validation = self.validate_single_result(result, timeout)
            individual_results.append(validation)
        
        # Validate consistency across results
        consistency_validation = self.validate_result_consistency(results)
        
        # Log summary
        total_results = len(results)
        valid_results = sum(1 for v in individual_results if v.is_valid)
        error_results = sum(1 for v in individual_results if v.severity == 'error')
        warning_results = sum(1 for v in individual_results if v.severity == 'warning')
        
        logger.info(f"Batch validation completed: {valid_results}/{total_results} valid results, "
                   f"{error_results} errors, {warning_results} warnings")
        
        return individual_results, consistency_validation


def create_default_validator() -> ResultValidator:
    """
    Create a validator with default settings suitable for most use cases.
    
    Returns:
        ResultValidator instance with default configuration
    """
    return ResultValidator(
        max_solve_time=3600.0,    # 1 hour
        min_solve_time=1e-6,      # 1 microsecond  
        max_objective_magnitude=1e12
    )


# Convenience functions for common validation tasks
def validate_result(result: Dict[str, Any], timeout: Optional[float] = None) -> ValidationResult:
    """
    Validate a single result using default validator.
    
    Args:
        result: Result dictionary to validate
        timeout: Optional timeout value
        
    Returns:
        ValidationResult
    """
    validator = create_default_validator()
    return validator.validate_single_result(result, timeout)


def validate_results_batch(results: List[Dict[str, Any]], 
                          timeout: Optional[float] = None) -> Tuple[List[ValidationResult], ValidationResult]:
    """
    Validate a batch of results using default validator.
    
    Args:
        results: List of result dictionaries
        timeout: Optional timeout value
        
    Returns:
        Tuple of (individual_validations, consistency_validation)
    """
    validator = create_default_validator()
    return validator.validate_batch_results(results, timeout)


if __name__ == "__main__":
    # Test the validation system with some sample data
    logging.basicConfig(level=logging.INFO)
    
    # Test valid result
    valid_result = {
        'solver_name': 'SciPy',
        'problem_name': 'test_lp',
        'solve_time': 0.001,
        'status': 'optimal',
        'objective_value': 1.5
    }
    
    # Test invalid results
    invalid_results = [
        {
            'solver_name': 'BadSolver',
            'problem_name': 'test_lp',
            'solve_time': -1.0,  # Negative time
            'status': 'optimal',
            'objective_value': 1.5
        },
        {
            'solver_name': 'AnotherSolver',
            'problem_name': 'test_lp',
            'solve_time': 0.001,
            'status': 'invalid_status',  # Bad status
            'objective_value': float('nan')  # NaN objective
        }
    ]
    
    validator = create_default_validator()
    
    print("Testing valid result:")
    result = validator.validate_single_result(valid_result)
    print(f"Valid: {result.is_valid}, Errors: {result.errors}, Warnings: {result.warnings}")
    
    print("\nTesting invalid results:")
    for i, invalid_result in enumerate(invalid_results):
        result = validator.validate_single_result(invalid_result)
        print(f"Result {i+1} - Valid: {result.is_valid}, Errors: {result.errors}")
    
    print("\nTesting batch validation:")
    all_results = [valid_result] + invalid_results
    individual, consistency = validator.validate_batch_results(all_results)
    print(f"Consistency validation - Valid: {consistency.is_valid}, Warnings: {consistency.warnings}")