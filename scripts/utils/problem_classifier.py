"""
Problem Classification System
============================

Intelligent problem analysis and classification for optimization problems.
Provides automatic problem type detection, complexity scoring, and difficulty estimation.
"""

import numpy as np
import cvxpy as cp
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.benchmark.problem_loader import ProblemData
from scripts.utils.logger import get_logger

logger = get_logger("problem_classifier")


class ProblemType(Enum):
    """Enumeration of optimization problem types."""
    LP = "LP"           # Linear Programming
    QP = "QP"           # Quadratic Programming
    SOCP = "SOCP"       # Second-Order Cone Programming
    SDP = "SDP"         # Semidefinite Programming
    UNKNOWN = "Unknown"


class DifficultyLevel(Enum):
    """Enumeration of problem difficulty levels."""
    TRIVIAL = "Trivial"     # < 10 variables
    EASY = "Easy"           # 10-100 variables
    MEDIUM = "Medium"       # 100-1000 variables
    HARD = "Hard"           # 1000-10000 variables
    EXTREME = "Extreme"     # > 10000 variables


@dataclass
class ProblemCharacteristics:
    """Container for problem characteristics and metrics."""
    name: str
    problem_type: ProblemType
    n_variables: int
    n_constraints: int
    n_equality_constraints: int = 0
    n_inequality_constraints: int = 0
    n_special_constraints: int = 0  # SOCP/SDP constraints
    
    # Complexity metrics
    complexity_score: float = 0.0
    difficulty_level: DifficultyLevel = DifficultyLevel.EASY
    sparsity_ratio: float = 0.0
    condition_estimate: float = 1.0
    
    # Problem-specific properties
    matrix_properties: Dict[str, Any] = None
    constraint_properties: Dict[str, Any] = None
    
    # Classification confidence
    classification_confidence: float = 1.0
    
    def __post_init__(self):
        if self.matrix_properties is None:
            self.matrix_properties = {}
        if self.constraint_properties is None:
            self.constraint_properties = {}


class ProblemClassifier:
    """Comprehensive problem classification and analysis system."""
    
    def __init__(self):
        """Initialize the problem classifier."""
        self.logger = get_logger("problem_classifier")
        
        # Difficulty thresholds (variables count)
        self.difficulty_thresholds = {
            DifficultyLevel.TRIVIAL: (0, 10),
            DifficultyLevel.EASY: (10, 100),
            DifficultyLevel.MEDIUM: (100, 1000),
            DifficultyLevel.HARD: (1000, 10000),
            DifficultyLevel.EXTREME: (10000, float('inf'))
        }
    
    def classify_problem(self, problem: ProblemData) -> ProblemCharacteristics:
        """
        Classify a problem and extract comprehensive characteristics.
        
        Args:
            problem: Problem data to classify
            
        Returns:
            ProblemCharacteristics containing classification results
        """
        self.logger.debug(f"Classifying problem: {problem.name}")
        
        # Detect problem type
        problem_type, confidence = self._detect_problem_type(problem)
        
        # Extract basic metrics
        n_vars, n_constraints = self._extract_problem_size(problem)
        n_eq, n_ineq, n_special = self._analyze_constraints(problem)
        
        # Calculate complexity metrics
        complexity_score = self._calculate_complexity_score(n_vars, n_constraints, problem_type)
        difficulty_level = self._determine_difficulty_level(n_vars, complexity_score)
        
        # Analyze matrix properties
        matrix_props = self._analyze_matrix_properties(problem)
        constraint_props = self._analyze_constraint_properties(problem)
        
        # Calculate sparsity and conditioning
        sparsity_ratio = self._calculate_sparsity(problem)
        condition_estimate = self._estimate_condition_number(problem)
        
        characteristics = ProblemCharacteristics(
            name=problem.name,
            problem_type=problem_type,
            n_variables=n_vars,
            n_constraints=n_constraints,
            n_equality_constraints=n_eq,
            n_inequality_constraints=n_ineq,
            n_special_constraints=n_special,
            complexity_score=complexity_score,
            difficulty_level=difficulty_level,
            sparsity_ratio=sparsity_ratio,
            condition_estimate=condition_estimate,
            matrix_properties=matrix_props,
            constraint_properties=constraint_props,
            classification_confidence=confidence
        )
        
        self.logger.info(f"Classified {problem.name}: {problem_type.value}, "
                        f"{difficulty_level.value}, {n_vars} vars, {n_constraints} constraints")
        
        return characteristics
    
    def _detect_problem_type(self, problem: ProblemData) -> Tuple[ProblemType, float]:
        """Detect the problem type based on structure and metadata."""
        
        # First check explicit problem class
        if hasattr(problem, 'problem_class') and problem.problem_class:
            type_mapping = {
                "LP": ProblemType.LP,
                "QP": ProblemType.QP,
                "SOCP": ProblemType.SOCP,
                "SDP": ProblemType.SDP
            }
            if problem.problem_class in type_mapping:
                return type_mapping[problem.problem_class], 1.0
        
        # Analyze CVXPY problem structure if available
        if problem.cvxpy_problem is not None:
            return self._analyze_cvxpy_problem_type(problem.cvxpy_problem)
        
        # Analyze traditional problem data structure
        if problem.P is not None:
            # Has quadratic terms
            return ProblemType.QP, 0.9
        elif problem.A_ub is not None or problem.A_eq is not None:
            # Linear constraints only
            return ProblemType.LP, 0.8
        
        return ProblemType.UNKNOWN, 0.1
    
    def _analyze_cvxpy_problem_type(self, cvxpy_problem) -> Tuple[ProblemType, float]:
        """Analyze CVXPY problem to determine type."""
        
        if not hasattr(cvxpy_problem, 'constraints'):
            return ProblemType.UNKNOWN, 0.1
        
        has_socp = False
        has_sdp = False
        has_quadratic = False
        
        # Check constraints for special structures
        for constraint in cvxpy_problem.constraints:
            # Check for SOC constraints
            if hasattr(constraint, 'args') and len(constraint.args) > 0:
                if any(hasattr(arg, 'cone_type') and getattr(arg, 'cone_type', None) == 'SOC' 
                       for arg in constraint.args if hasattr(arg, 'cone_type')):
                    has_socp = True
                
                # Check for PSD constraints
                if any(hasattr(arg, 'cone_type') and getattr(arg, 'cone_type', None) == 'PSD' 
                       for arg in constraint.args if hasattr(arg, 'cone_type')):
                    has_sdp = True
            
            # Alternative SOC detection
            if hasattr(constraint, 'cone_type'):
                if constraint.cone_type == 'SOC':
                    has_socp = True
                elif constraint.cone_type == 'PSD':
                    has_sdp = True
            
            # Check constraint string representation for CVXPY structures
            constraint_str = str(constraint)
            if 'SOC' in constraint_str or 'norm' in constraint_str:
                has_socp = True
            if 'PSD' in constraint_str or '>>' in constraint_str:
                has_sdp = True
        
        # Check objective for quadratic terms
        if hasattr(cvxpy_problem, 'objective') and cvxpy_problem.objective:
            obj_str = str(cvxpy_problem.objective)
            if 'quad_form' in obj_str or '**2' in obj_str or 'square' in obj_str:
                has_quadratic = True
        
        # Determine type based on detected structures
        if has_sdp:
            return ProblemType.SDP, 0.95
        elif has_socp:
            return ProblemType.SOCP, 0.95
        elif has_quadratic:
            return ProblemType.QP, 0.9
        else:
            return ProblemType.LP, 0.8
    
    def _extract_problem_size(self, problem: ProblemData) -> Tuple[int, int]:
        """Extract problem size (variables and constraints)."""
        
        # Try CVXPY problem first
        if problem.cvxpy_problem is not None:
            try:
                if hasattr(problem.cvxpy_problem, 'size_metrics'):
                    n_vars = getattr(problem.cvxpy_problem.size_metrics, 'num_scalar_variables', 0)
                else:
                    # Count variables manually
                    n_vars = sum(var.size for var in problem.cvxpy_problem.variables())
                
                n_constraints = len(problem.cvxpy_problem.constraints) if hasattr(problem.cvxpy_problem, 'constraints') else 0
                
                return n_vars, n_constraints
            except (AttributeError, TypeError):
                pass
        
        # Fallback to traditional problem data
        n_vars = 0
        if problem.c is not None:
            n_vars = len(problem.c)
        
        n_constraints = 0
        if problem.A_ub is not None:
            n_constraints += problem.A_ub.shape[0]
        if problem.A_eq is not None:
            n_constraints += problem.A_eq.shape[0]
        
        return n_vars, n_constraints
    
    def _analyze_constraints(self, problem: ProblemData) -> Tuple[int, int, int]:
        """Analyze constraint types."""
        
        n_eq = 0
        n_ineq = 0
        n_special = 0
        
        # Traditional format
        if problem.A_eq is not None:
            n_eq = problem.A_eq.shape[0]
        if problem.A_ub is not None:
            n_ineq = problem.A_ub.shape[0]
        
        # CVXPY format
        if problem.cvxpy_problem is not None and hasattr(problem.cvxpy_problem, 'constraints'):
            for constraint in problem.cvxpy_problem.constraints:
                constraint_str = str(constraint)
                if '==' in constraint_str:
                    n_eq += 1
                elif '<=' in constraint_str or '>=' in constraint_str:
                    n_ineq += 1
                elif 'SOC' in constraint_str or 'PSD' in constraint_str or '>>' in constraint_str:
                    n_special += 1
                else:
                    # Default to inequality for unknown types
                    n_ineq += 1
        
        return n_eq, n_ineq, n_special
    
    def _calculate_complexity_score(self, n_vars: int, n_constraints: int, problem_type: ProblemType) -> float:
        """Calculate a complexity score for the problem."""
        
        # Base complexity from problem size
        size_complexity = np.log10(max(1, n_vars)) + np.log10(max(1, n_constraints))
        
        # Type-based multiplier
        type_multipliers = {
            ProblemType.LP: 1.0,
            ProblemType.QP: 1.5,
            ProblemType.SOCP: 2.0,
            ProblemType.SDP: 3.0,
            ProblemType.UNKNOWN: 1.0
        }
        
        type_factor = type_multipliers.get(problem_type, 1.0)
        
        # Constraint density factor
        if n_vars > 0:
            density_factor = 1.0 + (n_constraints / n_vars) * 0.1
        else:
            density_factor = 1.0
        
        complexity_score = size_complexity * type_factor * density_factor
        
        return float(complexity_score)
    
    def _determine_difficulty_level(self, n_vars: int, complexity_score: float) -> DifficultyLevel:
        """Determine difficulty level based on problem size and complexity."""
        
        # Primary classification by variable count
        for difficulty, (min_vars, max_vars) in self.difficulty_thresholds.items():
            if min_vars <= n_vars < max_vars:
                base_difficulty = difficulty
                break
        else:
            base_difficulty = DifficultyLevel.EXTREME
        
        # Adjust based on complexity score
        if complexity_score > 10.0:
            # Very high complexity, bump up difficulty
            difficulty_levels = list(DifficultyLevel)
            current_idx = difficulty_levels.index(base_difficulty)
            if current_idx < len(difficulty_levels) - 1:
                return difficulty_levels[current_idx + 1]
        elif complexity_score < 2.0:
            # Low complexity, possibly bump down difficulty
            difficulty_levels = list(DifficultyLevel)
            current_idx = difficulty_levels.index(base_difficulty)
            if current_idx > 0:
                return difficulty_levels[current_idx - 1]
        
        return base_difficulty
    
    def _analyze_matrix_properties(self, problem: ProblemData) -> Dict[str, Any]:
        """Analyze matrix properties of the problem."""
        
        properties = {}
        
        # Analyze constraint matrices
        if problem.A_ub is not None:
            properties['inequality_matrix_shape'] = problem.A_ub.shape
            properties['inequality_matrix_rank'] = np.linalg.matrix_rank(problem.A_ub)
            
        if problem.A_eq is not None:
            properties['equality_matrix_shape'] = problem.A_eq.shape
            properties['equality_matrix_rank'] = np.linalg.matrix_rank(problem.A_eq)
        
        # Analyze quadratic matrix
        if problem.P is not None:
            properties['quadratic_matrix_shape'] = problem.P.shape
            properties['quadratic_matrix_rank'] = np.linalg.matrix_rank(problem.P)
            
            # Check positive definiteness
            try:
                eigenvals = np.linalg.eigvals(problem.P)
                properties['quadratic_min_eigenvalue'] = float(np.min(eigenvals))
                properties['quadratic_max_eigenvalue'] = float(np.max(eigenvals))
                properties['quadratic_is_psd'] = bool(np.all(eigenvals >= -1e-8))
                properties['quadratic_condition_number'] = float(np.max(eigenvals) / np.max([np.min(eigenvals), 1e-12]))
            except np.linalg.LinAlgError:
                pass
        
        return properties
    
    def _analyze_constraint_properties(self, problem: ProblemData) -> Dict[str, Any]:
        """Analyze constraint structure properties."""
        
        properties = {}
        
        # Count different constraint types from bounds
        if problem.bounds is not None:
            n_bounded_vars = sum(1 for bound in problem.bounds if bound is not None)
            n_free_vars = len(problem.bounds) - n_bounded_vars
            
            properties['n_bounded_variables'] = n_bounded_vars
            properties['n_free_variables'] = n_free_vars
            properties['bound_ratio'] = n_bounded_vars / max(1, len(problem.bounds))
        
        # Analyze RHS vectors
        if problem.b_ub is not None:
            properties['inequality_rhs_norm'] = float(np.linalg.norm(problem.b_ub))
            properties['inequality_rhs_mean'] = float(np.mean(problem.b_ub))
            
        if problem.b_eq is not None:
            properties['equality_rhs_norm'] = float(np.linalg.norm(problem.b_eq))
            properties['equality_rhs_mean'] = float(np.mean(problem.b_eq))
        
        return properties
    
    def _calculate_sparsity(self, problem: ProblemData) -> float:
        """Calculate sparsity ratio of problem matrices."""
        
        total_elements = 0
        nonzero_elements = 0
        
        # Count constraint matrix elements
        for matrix in [problem.A_ub, problem.A_eq, problem.P]:
            if matrix is not None:
                total_elements += matrix.size
                nonzero_elements += np.count_nonzero(matrix)
        
        if total_elements == 0:
            return 0.0
        
        sparsity_ratio = 1.0 - (nonzero_elements / total_elements)
        return float(sparsity_ratio)
    
    def _estimate_condition_number(self, problem: ProblemData) -> float:
        """Estimate condition number of the problem."""
        
        condition_estimates = []
        
        # Check constraint matrices
        for matrix in [problem.A_ub, problem.A_eq]:
            if matrix is not None and matrix.shape[0] > 0 and matrix.shape[1] > 0:
                try:
                    # Use SVD for more stable condition number estimation
                    _, s, _ = np.linalg.svd(matrix, compute_uv=False)
                    if len(s) > 0 and s[-1] > 1e-14:
                        cond = s[0] / s[-1]
                        condition_estimates.append(cond)
                except (np.linalg.LinAlgError, ValueError):
                    pass
        
        # Check quadratic matrix
        if problem.P is not None:
            try:
                cond = np.linalg.cond(problem.P)
                if np.isfinite(cond):
                    condition_estimates.append(cond)
            except (np.linalg.LinAlgError, ValueError):
                pass
        
        if condition_estimates:
            return float(np.max(condition_estimates))
        else:
            return 1.0
    
    def recommend_solvers(self, characteristics: ProblemCharacteristics) -> List[Tuple[str, float]]:
        """
        Recommend solvers based on problem characteristics.
        
        Args:
            characteristics: Problem characteristics
            
        Returns:
            List of (solver_name, suitability_score) tuples, sorted by score
        """
        
        recommendations = []
        
        # Define solver capabilities and preferences
        solver_profiles = {
            "CLARABEL": {
                "types": [ProblemType.LP, ProblemType.QP, ProblemType.SOCP, ProblemType.SDP],
                "difficulty_preference": [DifficultyLevel.EASY, DifficultyLevel.MEDIUM, DifficultyLevel.HARD],
                "base_score": 0.9
            },
            "SCS": {
                "types": [ProblemType.LP, ProblemType.QP, ProblemType.SOCP, ProblemType.SDP],
                "difficulty_preference": [DifficultyLevel.MEDIUM, DifficultyLevel.HARD, DifficultyLevel.EXTREME],
                "base_score": 0.85
            },
            "ECOS": {
                "types": [ProblemType.LP, ProblemType.QP, ProblemType.SOCP],
                "difficulty_preference": [DifficultyLevel.TRIVIAL, DifficultyLevel.EASY, DifficultyLevel.MEDIUM],
                "base_score": 0.8
            },
            "OSQP": {
                "types": [ProblemType.QP, ProblemType.SOCP],
                "difficulty_preference": [DifficultyLevel.EASY, DifficultyLevel.MEDIUM, DifficultyLevel.HARD],
                "base_score": 0.85
            },
            "SciPy": {
                "types": [ProblemType.LP, ProblemType.QP],
                "difficulty_preference": [DifficultyLevel.TRIVIAL, DifficultyLevel.EASY],
                "base_score": 0.6
            }
        }
        
        for solver_name, profile in solver_profiles.items():
            score = profile["base_score"]
            
            # Type compatibility
            if characteristics.problem_type in profile["types"]:
                score *= 1.0
            else:
                score *= 0.1  # Heavy penalty for unsupported types
            
            # Difficulty preference
            if characteristics.difficulty_level in profile["difficulty_preference"]:
                score *= 1.2
            elif characteristics.difficulty_level == DifficultyLevel.EXTREME and solver_name == "SCS":
                score *= 1.1  # SCS handles extreme problems better
            
            # Special bonuses
            if characteristics.problem_type == ProblemType.SDP and solver_name in ["CLARABEL", "SCS"]:
                score *= 1.3  # These are the only SDP solvers
            elif characteristics.problem_type == ProblemType.QP and solver_name == "OSQP":
                score *= 1.2  # OSQP specializes in QP
            
            # Sparsity bonus
            if characteristics.sparsity_ratio > 0.8 and solver_name in ["SCS", "OSQP"]:
                score *= 1.1  # These handle sparse problems well
            
            # Condition number penalty
            if characteristics.condition_estimate > 1e6:
                if solver_name == "CLARABEL":
                    score *= 1.1  # CLARABEL is more numerically stable
                else:
                    score *= 0.9
            
            recommendations.append((solver_name, min(1.0, score)))
        
        # Sort by score (descending)
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        return recommendations


def analyze_problem_registry() -> Dict[str, ProblemCharacteristics]:
    """Analyze all problems in the registry and return their characteristics."""
    
    from scripts.benchmark.problem_loader import load_problem_registry
    
    classifier = ProblemClassifier()
    registry = load_problem_registry()
    problem_characteristics = {}
    
    logger.info("Analyzing problem registry...")
    
    for problem_set in registry["problems"]:
        for problem_class in registry["problems"][problem_set]:
            for problem_info in registry["problems"][problem_set][problem_class]:
                problem_name = problem_info["name"]
                
                try:
                    # Load problem (this might fail for some problems)
                    from scripts.benchmark.problem_loader import load_problem
                    problem_data = load_problem(problem_name, problem_set)
                    
                    # Classify problem
                    characteristics = classifier.classify_problem(problem_data)
                    problem_characteristics[problem_name] = characteristics
                    
                    logger.info(f"Analyzed {problem_name}: {characteristics.problem_type.value}, "
                              f"{characteristics.difficulty_level.value}")
                    
                except Exception as e:
                    logger.warning(f"Failed to analyze problem {problem_name}: {e}")
                    
                    # Create basic characteristics from registry info
                    problem_type = ProblemType.UNKNOWN
                    if "problem_class" in problem_info:
                        try:
                            problem_type = ProblemType(problem_info["problem_class"])
                        except ValueError:
                            pass
                    
                    characteristics = ProblemCharacteristics(
                        name=problem_name,
                        problem_type=problem_type,
                        n_variables=problem_info.get("variables", 0),
                        n_constraints=problem_info.get("constraints", 0),
                        classification_confidence=0.5
                    )
                    problem_characteristics[problem_name] = characteristics
    
    logger.info(f"Analyzed {len(problem_characteristics)} problems")
    return problem_characteristics


if __name__ == "__main__":
    # Test the problem classification system
    print("Testing Problem Classification System...")
    
    # Analyze all problems in registry
    problem_characteristics = analyze_problem_registry()
    
    print(f"\n=== Problem Classification Results ===")
    print(f"Total problems analyzed: {len(problem_characteristics)}")
    
    # Group by type
    type_counts = {}
    difficulty_counts = {}
    
    for name, char in problem_characteristics.items():
        problem_type = char.problem_type.value
        difficulty = char.difficulty_level.value
        
        type_counts[problem_type] = type_counts.get(problem_type, 0) + 1
        difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1
        
        print(f"  {name:20} | {problem_type:6} | {difficulty:8} | "
              f"{char.n_variables:4} vars | {char.n_constraints:3} cons | "
              f"Score: {char.complexity_score:.2f}")
    
    print(f"\n=== Type Distribution ===")
    for ptype, count in sorted(type_counts.items()):
        print(f"  {ptype:8}: {count:2} problems")
    
    print(f"\n=== Difficulty Distribution ===")
    for difficulty, count in sorted(difficulty_counts.items()):
        print(f"  {difficulty:8}: {count:2} problems")
    
    # Test solver recommendations
    print(f"\n=== Solver Recommendations ===")
    classifier = ProblemClassifier()
    
    for name, char in list(problem_characteristics.items())[:3]:  # First 3 problems
        recommendations = classifier.recommend_solvers(char)
        print(f"\n{name} ({char.problem_type.value}, {char.difficulty_level.value}):")
        for solver, score in recommendations[:3]:  # Top 3 recommendations
            print(f"  {solver:12}: {score:.3f}")
    
    print("\n✓ Problem classification system test completed!")