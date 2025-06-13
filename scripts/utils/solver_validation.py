import sys
import time
import importlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.utils.logger import get_logger

logger = get_logger("solver_validation")


class ProblemType(Enum):
    """Supported optimization problem types."""
    LP = "LP"      # Linear Programming
    QP = "QP"      # Quadratic Programming  
    SOCP = "SOCP"  # Second-Order Cone Programming
    SDP = "SDP"    # Semidefinite Programming


@dataclass
class BackendCapability:
    """Capability information for a CVXPY backend."""
    name: str
    supported_types: List[ProblemType]
    installation_status: bool
    version: Optional[str] = None
    performance_tier: str = "medium"  # low, medium, high
    memory_efficiency: str = "medium"  # low, medium, high
    stability: str = "stable"  # experimental, stable, mature
    installation_notes: str = ""


@dataclass
class ValidationResult:
    """Result of backend validation check."""
    backend_name: str
    available: bool
    capabilities: Optional[BackendCapability] = None
    error_message: Optional[str] = None
    validation_time: float = 0.0


class SolverValidator:
    """Validates CVXPY solver backends and their capabilities."""
    
    def __init__(self):
        self.logger = get_logger("solver_validator")
        
        # Define comprehensive backend capability matrix
        self.backend_capabilities = {
            # General purpose solvers
            "CLARABEL": BackendCapability(
                name="CLARABEL",
                supported_types=[ProblemType.LP, ProblemType.QP, ProblemType.SOCP, ProblemType.SDP],
                installation_status=False,
                performance_tier="high",
                memory_efficiency="high", 
                stability="stable",
                installation_notes="Rust-based conic solver, excellent for all problem types"
            ),
            "SCS": BackendCapability(
                name="SCS",
                supported_types=[ProblemType.LP, ProblemType.QP, ProblemType.SOCP, ProblemType.SDP],
                installation_status=False,
                performance_tier="medium",
                memory_efficiency="medium",
                stability="mature",
                installation_notes="Splitting Conic Solver, good general-purpose solver"
            ),
            "ECOS": BackendCapability(
                name="ECOS",
                supported_types=[ProblemType.LP, ProblemType.QP, ProblemType.SOCP],
                installation_status=False,
                performance_tier="medium",
                memory_efficiency="high",
                stability="mature",
                installation_notes="Embedded Conic Solver, lightweight and fast"
            ),
            
            # QP specialists
            "OSQP": BackendCapability(
                name="OSQP",
                supported_types=[ProblemType.QP, ProblemType.SOCP],
                installation_status=False,
                performance_tier="high",
                memory_efficiency="high",
                stability="mature",
                installation_notes="Operator Splitting QP solver, excellent for QP problems"
            ),
            
            # LP specialists  
            "CBC": BackendCapability(
                name="CBC",
                supported_types=[ProblemType.LP],
                installation_status=False,
                performance_tier="high",
                memory_efficiency="medium",
                stability="mature",
                installation_notes="COIN-OR Branch and Cut, excellent for integer LP"
            ),
            "GLOP": BackendCapability(
                name="GLOP",
                supported_types=[ProblemType.LP],
                installation_status=False,
                performance_tier="high",
                memory_efficiency="high",
                stability="stable",
                installation_notes="Google's Linear Optimizer, fast and reliable"
            ),
            "GLOP_MI": BackendCapability(
                name="GLOP_MI",
                supported_types=[ProblemType.LP],
                installation_status=False,
                performance_tier="medium",
                memory_efficiency="medium",
                stability="stable",
                installation_notes="GLOP with mixed-integer support"
            ),
            
            # Other open-source solvers
            "CVXOPT": BackendCapability(
                name="CVXOPT",
                supported_types=[ProblemType.LP, ProblemType.QP, ProblemType.SOCP],
                installation_status=False,
                performance_tier="medium",
                memory_efficiency="medium",
                stability="mature",
                installation_notes="Python optimization package, well-tested"
            ),
            "SCIP": BackendCapability(
                name="SCIP",
                supported_types=[ProblemType.LP, ProblemType.QP],
                installation_status=False,
                performance_tier="high",
                memory_efficiency="medium",
                stability="mature",
                installation_notes="Solving Constraint Integer Programs"
            ),
            
            # Commercial solvers (for reference, typically disabled)
            "GUROBI": BackendCapability(
                name="GUROBI",
                supported_types=[ProblemType.LP, ProblemType.QP, ProblemType.SOCP],
                installation_status=False,
                performance_tier="high",
                memory_efficiency="high",
                stability="mature",
                installation_notes="Commercial solver (requires license)"
            ),
            "MOSEK": BackendCapability(
                name="MOSEK",
                supported_types=[ProblemType.LP, ProblemType.QP, ProblemType.SOCP, ProblemType.SDP],
                installation_status=False,
                performance_tier="high",
                memory_efficiency="high",
                stability="mature",
                installation_notes="Commercial solver (requires license)"
            ),
            "CPLEX": BackendCapability(
                name="CPLEX",
                supported_types=[ProblemType.LP, ProblemType.QP, ProblemType.SOCP],
                installation_status=False,
                performance_tier="high",
                memory_efficiency="high",
                stability="mature",
                installation_notes="IBM commercial solver (requires license)"
            )
        }
    
    def validate_cvxpy_installation(self) -> ValidationResult:
        """Validate CVXPY installation and get available backends."""
        start_time = time.time()
        
        try:
            import cvxpy as cp
            validation_time = time.time() - start_time
            
            # Get list of installed solvers
            available_backends = cp.installed_solvers()
            
            return ValidationResult(
                backend_name="CVXPY",
                available=True,
                validation_time=validation_time,
                capabilities=None  # Will populate with backend-specific info
            )
            
        except ImportError as e:
            validation_time = time.time() - start_time
            return ValidationResult(
                backend_name="CVXPY",
                available=False,
                error_message=f"CVXPY not installed: {str(e)}",
                validation_time=validation_time
            )
    
    def validate_backend(self, backend_name: str) -> ValidationResult:
        """Validate a specific CVXPY backend."""
        start_time = time.time()
        
        # Check if backend is in our capability matrix
        if backend_name not in self.backend_capabilities:
            validation_time = time.time() - start_time
            return ValidationResult(
                backend_name=backend_name,
                available=False,
                error_message=f"Unknown backend: {backend_name}",
                validation_time=validation_time
            )
        
        try:
            import cvxpy as cp
            available_backends = cp.installed_solvers()
            
            if backend_name in available_backends:
                # Backend is available, create capability info
                capability = self.backend_capabilities[backend_name]
                capability.installation_status = True
                
                # Try to get version if possible
                try:
                    solver_module = getattr(cp, backend_name, None)
                    if hasattr(solver_module, 'version'):
                        capability.version = str(solver_module.version())
                except:
                    capability.version = "unknown"
                
                validation_time = time.time() - start_time
                return ValidationResult(
                    backend_name=backend_name,
                    available=True,
                    capabilities=capability,
                    validation_time=validation_time
                )
            else:
                validation_time = time.time() - start_time
                return ValidationResult(
                    backend_name=backend_name,
                    available=False,
                    error_message=f"Backend {backend_name} not installed",
                    validation_time=validation_time
                )
                
        except ImportError:
            validation_time = time.time() - start_time
            return ValidationResult(
                backend_name=backend_name,
                available=False,
                error_message="CVXPY not available",
                validation_time=validation_time
            )
        except Exception as e:
            validation_time = time.time() - start_time
            return ValidationResult(
                backend_name=backend_name,
                available=False,
                error_message=f"Validation error: {str(e)}",
                validation_time=validation_time
            )
    
    def validate_all_backends(self) -> Dict[str, ValidationResult]:
        """Validate all known CVXPY backends."""
        self.logger.info("Validating all CVXPY backends...")
        
        results = {}
        
        # First validate CVXPY itself
        cvxpy_result = self.validate_cvxpy_installation()
        if not cvxpy_result.available:
            self.logger.error("CVXPY not available, skipping backend validation")
            return {"CVXPY": cvxpy_result}
        
        # Validate each backend
        for backend_name in self.backend_capabilities.keys():
            result = self.validate_backend(backend_name)
            results[backend_name] = result
            
            if result.available:
                self.logger.info(f"✓ {backend_name}: Available")
            else:
                self.logger.debug(f"✗ {backend_name}: {result.error_message}")
        
        available_count = sum(1 for r in results.values() if r.available)
        total_count = len(results)
        
        self.logger.info(f"Backend validation complete: {available_count}/{total_count} available")
        
        return results
    
    def get_backends_for_problem_type(self, problem_type: ProblemType) -> List[str]:
        """Get list of backends that support a specific problem type."""
        compatible_backends = []
        
        for backend_name, capability in self.backend_capabilities.items():
            if problem_type in capability.supported_types:
                compatible_backends.append(backend_name)
        
        return compatible_backends
    
    def get_recommended_backend(self, problem_type: ProblemType, 
                               available_backends: List[str]) -> Optional[str]:
        """Recommend the best backend for a problem type from available backends."""
        
        # Get backends that support this problem type
        compatible_backends = self.get_backends_for_problem_type(problem_type)
        
        # Filter to only available backends
        available_compatible = [b for b in compatible_backends if b in available_backends]
        
        if not available_compatible:
            return None
        
        # Define preference order based on problem type and performance
        recommendations = {
            ProblemType.LP: ["CLARABEL", "GLOP", "CBC", "ECOS", "SCS", "CVXOPT", "SCIP"],
            ProblemType.QP: ["CLARABEL", "OSQP", "SCS", "ECOS", "CVXOPT", "SCIP"],
            ProblemType.SOCP: ["CLARABEL", "SCS", "ECOS", "OSQP", "CVXOPT"],
            ProblemType.SDP: ["CLARABEL", "SCS"]
        }
        
        preferred_order = recommendations.get(problem_type, available_compatible)
        
        # Return first available backend in preference order
        for backend in preferred_order:
            if backend in available_compatible:
                return backend
        
        # Fallback to first available
        return available_compatible[0] if available_compatible else None
    
    def generate_validation_report(self, validation_results: Dict[str, ValidationResult]) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        
        available_backends = {k: v for k, v in validation_results.items() if v.available}
        unavailable_backends = {k: v for k, v in validation_results.items() if not v.available}
        
        # Problem type coverage analysis
        problem_type_coverage = {}
        for problem_type in ProblemType:
            compatible_backends = self.get_backends_for_problem_type(problem_type)
            available_compatible = [b for b in compatible_backends if b in available_backends]
            problem_type_coverage[problem_type.value] = {
                "total_backends": len(compatible_backends),
                "available_backends": len(available_compatible),
                "coverage_percentage": len(available_compatible) / len(compatible_backends) * 100 if compatible_backends else 0,
                "available_names": available_compatible,
                "recommended": self.get_recommended_backend(problem_type, list(available_backends.keys()))
            }
        
        # Performance tier analysis
        performance_analysis = {"high": [], "medium": [], "low": []}
        for backend_name, result in available_backends.items():
            if result.capabilities:
                tier = result.capabilities.performance_tier
                performance_analysis[tier].append(backend_name)
        
        return {
            "summary": {
                "total_backends": len(validation_results),
                "available_backends": len(available_backends),
                "unavailable_backends": len(unavailable_backends),
                "availability_percentage": len(available_backends) / len(validation_results) * 100 if validation_results else 0
            },
            "available_backends": {k: {
                "capabilities": v.capabilities.__dict__ if v.capabilities else None,
                "validation_time": v.validation_time
            } for k, v in available_backends.items()},
            "unavailable_backends": {k: {
                "error_message": v.error_message,
                "validation_time": v.validation_time
            } for k, v in unavailable_backends.items()},
            "problem_type_coverage": problem_type_coverage,
            "performance_analysis": performance_analysis,
            "recommendations": {
                problem_type.value: self.get_recommended_backend(problem_type, list(available_backends.keys()))
                for problem_type in ProblemType
            }
        }


if __name__ == "__main__":
    # Test script to verify solver validation
    try:
        print("Testing Solver Validation System...")
        
        validator = SolverValidator()
        
        # Test CVXPY validation
        print("\nTesting CVXPY installation:")
        cvxpy_result = validator.validate_cvxpy_installation()
        if cvxpy_result.available:
            print(f"✓ CVXPY available (validated in {cvxpy_result.validation_time:.3f}s)")
        else:
            print(f"✗ CVXPY not available: {cvxpy_result.error_message}")
            exit(1)
        
        # Test all backend validation
        print("\nValidating all backends:")
        validation_results = validator.validate_all_backends()
        
        available_count = sum(1 for r in validation_results.values() if r.available)
        print(f"\nValidation Summary: {available_count}/{len(validation_results)} backends available")
        
        # Show available backends
        print("\nAvailable backends:")
        for name, result in validation_results.items():
            if result.available and result.capabilities:
                cap = result.capabilities
                supported_types = [pt.value for pt in cap.supported_types]
                print(f"  ✓ {name}: {', '.join(supported_types)} | {cap.performance_tier} performance | {cap.stability}")
        
        # Show unavailable backends
        print("\nUnavailable backends:")
        for name, result in validation_results.items():
            if not result.available:
                print(f"  ✗ {name}: {result.error_message}")
        
        # Test problem type recommendations
        print("\nBackend recommendations by problem type:")
        for problem_type in ProblemType:
            available_backends = [k for k, v in validation_results.items() if v.available]
            recommended = validator.get_recommended_backend(problem_type, available_backends)
            compatible = validator.get_backends_for_problem_type(problem_type)
            available_compatible = [b for b in compatible if b in available_backends]
            
            print(f"  {problem_type.value}: {recommended} (from {len(available_compatible)} available: {available_compatible})")
        
        # Generate full validation report
        print("\nGenerating validation report...")
        report = validator.generate_validation_report(validation_results)
        
        print(f"\nValidation Report Summary:")
        print(f"  Total backends: {report['summary']['total_backends']}")
        print(f"  Available: {report['summary']['available_backends']} ({report['summary']['availability_percentage']:.1f}%)")
        print(f"  Problem type coverage:")
        for pt, coverage in report['problem_type_coverage'].items():
            print(f"    {pt}: {coverage['available_backends']}/{coverage['total_backends']} ({coverage['coverage_percentage']:.1f}%) - Recommended: {coverage['recommended']}")
        
        print("\n✓ Solver validation system test completed!")
        
    except Exception as e:
        logger.error(f"Solver validation test failed: {e}")
        raise