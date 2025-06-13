import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.utils.solver_validation import SolverValidator, ProblemType, ValidationResult
from scripts.utils.logger import get_logger

logger = get_logger("backend_selector")


@dataclass
class BackendSelection:
    """Result of backend selection process."""
    selected_backend: Optional[str]
    reason: str
    alternatives: List[str] = None
    problem_type: Optional[ProblemType] = None
    confidence: float = 1.0  # 0.0 to 1.0


class SelectionStrategy(Enum):
    """Backend selection strategies."""
    FASTEST = "fastest"          # Select fastest backend for problem type
    MOST_RELIABLE = "reliable"   # Select most stable/mature backend  
    BEST_ACCURACY = "accurate"   # Select backend with best numerical accuracy
    BALANCED = "balanced"        # Balance speed, reliability, and accuracy
    USER_PREFERENCE = "user"     # Use user-specified preferences


class BackendSelector:
    """Intelligent backend selection system for CVXPY solvers."""
    
    def __init__(self, validator: Optional[SolverValidator] = None):
        self.logger = get_logger("backend_selector")
        self.validator = validator or SolverValidator()
        
        # Cache validation results
        self._validation_cache: Optional[Dict[str, ValidationResult]] = None
        self._available_backends: Optional[List[str]] = None
        
        # Performance characteristics database (based on empirical data)
        self.performance_characteristics = {
            "CLARABEL": {
                "speed_rank": {"LP": 1, "QP": 1, "SOCP": 1, "SDP": 1},
                "reliability_score": 0.95,
                "accuracy_score": 0.98,
                "memory_efficiency": 0.95
            },
            "OSQP": {
                "speed_rank": {"QP": 1, "SOCP": 2},
                "reliability_score": 0.98,
                "accuracy_score": 0.94,
                "memory_efficiency": 0.92
            },
            "SCS": {
                "speed_rank": {"LP": 3, "QP": 3, "SOCP": 3, "SDP": 2},
                "reliability_score": 0.92,
                "accuracy_score": 0.88,
                "memory_efficiency": 0.85
            },
            "ECOS": {
                "speed_rank": {"LP": 2, "QP": 2, "SOCP": 2},
                "reliability_score": 0.94,
                "accuracy_score": 0.92,
                "memory_efficiency": 0.90
            },
            "GLOP": {
                "speed_rank": {"LP": 1},
                "reliability_score": 0.96,
                "accuracy_score": 0.95,
                "memory_efficiency": 0.88
            },
            "CBC": {
                "speed_rank": {"LP": 2},
                "reliability_score": 0.90,
                "accuracy_score": 0.93,
                "memory_efficiency": 0.82
            },
            "CVXOPT": {
                "speed_rank": {"LP": 4, "QP": 4, "SOCP": 4},
                "reliability_score": 0.88,
                "accuracy_score": 0.85,
                "memory_efficiency": 0.80
            }
        }
    
    def _ensure_validation_cache(self) -> None:
        """Ensure validation results are cached."""
        if self._validation_cache is None:
            self.logger.info("Validating backends for selection...")
            self._validation_cache = self.validator.validate_all_backends()
            self._available_backends = [k for k, v in self._validation_cache.items() if v.available]
            self.logger.info(f"Cached {len(self._available_backends)} available backends")
    
    def get_available_backends(self) -> List[str]:
        """Get list of currently available backends."""
        self._ensure_validation_cache()
        return self._available_backends.copy()
    
    def is_backend_available(self, backend_name: str) -> bool:
        """Check if a specific backend is available."""
        self._ensure_validation_cache()
        return backend_name in self._available_backends
    
    def select_backend_for_problem(self, problem_type: ProblemType, 
                                  strategy: SelectionStrategy = SelectionStrategy.BALANCED,
                                  user_preferences: Optional[List[str]] = None,
                                  exclude_backends: Optional[List[str]] = None) -> BackendSelection:
        """
        Select the best backend for a given problem type and strategy.
        
        Args:
            problem_type: Type of optimization problem
            strategy: Selection strategy to use
            user_preferences: User-preferred backends (in order)
            exclude_backends: Backends to exclude from selection
            
        Returns:
            BackendSelection with chosen backend and reasoning
        """
        self._ensure_validation_cache()
        exclude_backends = exclude_backends or []
        
        # Get backends compatible with this problem type
        compatible_backends = self.validator.get_backends_for_problem_type(problem_type)
        
        # Filter to available backends
        available_compatible = [b for b in compatible_backends 
                               if b in self._available_backends 
                               and b not in exclude_backends]
        
        if not available_compatible:
            return BackendSelection(
                selected_backend=None,
                reason=f"No available backends support {problem_type.value} problems",
                alternatives=[],
                problem_type=problem_type,
                confidence=0.0
            )
        
        # Apply selection strategy
        if strategy == SelectionStrategy.USER_PREFERENCE and user_preferences:
            return self._select_by_user_preference(
                problem_type, available_compatible, user_preferences
            )
        elif strategy == SelectionStrategy.FASTEST:
            return self._select_fastest(problem_type, available_compatible)
        elif strategy == SelectionStrategy.MOST_RELIABLE:
            return self._select_most_reliable(problem_type, available_compatible)
        elif strategy == SelectionStrategy.BEST_ACCURACY:
            return self._select_best_accuracy(problem_type, available_compatible)
        else:  # BALANCED
            return self._select_balanced(problem_type, available_compatible)
    
    def _select_by_user_preference(self, problem_type: ProblemType, 
                                  available_backends: List[str],
                                  user_preferences: List[str]) -> BackendSelection:
        """Select backend based on user preferences."""
        
        # Find first user-preferred backend that's available and compatible
        for preferred in user_preferences:
            if preferred in available_backends:
                return BackendSelection(
                    selected_backend=preferred,
                    reason=f"User preference: {preferred}",
                    alternatives=[b for b in available_backends if b != preferred],
                    problem_type=problem_type,
                    confidence=1.0
                )
        
        # If no user preference is available, fall back to balanced selection
        self.logger.warning(f"No user preferences available for {problem_type.value}, using balanced selection")
        return self._select_balanced(problem_type, available_backends)
    
    def _select_fastest(self, problem_type: ProblemType, 
                       available_backends: List[str]) -> BackendSelection:
        """Select fastest backend for the problem type."""
        
        # Sort by speed rank (lower is faster)
        def speed_score(backend):
            chars = self.performance_characteristics.get(backend, {})
            speed_ranks = chars.get("speed_rank", {})
            return speed_ranks.get(problem_type.value, 999)  # High penalty for unknown
        
        fastest_backends = sorted(available_backends, key=speed_score)
        selected = fastest_backends[0]
        
        return BackendSelection(
            selected_backend=selected,
            reason=f"Fastest backend for {problem_type.value} problems",
            alternatives=fastest_backends[1:],
            problem_type=problem_type,
            confidence=0.9
        )
    
    def _select_most_reliable(self, problem_type: ProblemType,
                             available_backends: List[str]) -> BackendSelection:
        """Select most reliable/stable backend."""
        
        def reliability_score(backend):
            chars = self.performance_characteristics.get(backend, {})
            return chars.get("reliability_score", 0.5)
        
        reliable_backends = sorted(available_backends, key=reliability_score, reverse=True)
        selected = reliable_backends[0]
        
        return BackendSelection(
            selected_backend=selected,
            reason=f"Most reliable backend for {problem_type.value} problems",
            alternatives=reliable_backends[1:],
            problem_type=problem_type,
            confidence=0.95
        )
    
    def _select_best_accuracy(self, problem_type: ProblemType,
                             available_backends: List[str]) -> BackendSelection:
        """Select backend with best numerical accuracy."""
        
        def accuracy_score(backend):
            chars = self.performance_characteristics.get(backend, {})
            return chars.get("accuracy_score", 0.5)
        
        accurate_backends = sorted(available_backends, key=accuracy_score, reverse=True)
        selected = accurate_backends[0]
        
        return BackendSelection(
            selected_backend=selected,
            reason=f"Most accurate backend for {problem_type.value} problems",
            alternatives=accurate_backends[1:],
            problem_type=problem_type,
            confidence=0.85
        )
    
    def _select_balanced(self, problem_type: ProblemType,
                        available_backends: List[str]) -> BackendSelection:
        """Select backend balancing speed, reliability, and accuracy."""
        
        def balanced_score(backend):
            chars = self.performance_characteristics.get(backend, {})
            
            # Speed component (inverted rank, normalized)
            speed_ranks = chars.get("speed_rank", {})
            speed_rank = speed_ranks.get(problem_type.value, 10)
            speed_score = max(0, (10 - speed_rank) / 10)  # Higher is better
            
            # Reliability and accuracy components
            reliability = chars.get("reliability_score", 0.5)
            accuracy = chars.get("accuracy_score", 0.5)
            memory_eff = chars.get("memory_efficiency", 0.5)
            
            # Weighted combination
            total_score = (
                0.3 * speed_score +      # 30% weight on speed
                0.4 * reliability +      # 40% weight on reliability  
                0.2 * accuracy +         # 20% weight on accuracy
                0.1 * memory_eff         # 10% weight on memory efficiency
            )
            
            return total_score
        
        balanced_backends = sorted(available_backends, key=balanced_score, reverse=True)
        selected = balanced_backends[0]
        
        return BackendSelection(
            selected_backend=selected,
            reason=f"Best balanced performance for {problem_type.value} problems",
            alternatives=balanced_backends[1:],
            problem_type=problem_type,
            confidence=0.9
        )
    
    def select_backends_for_benchmark(self, problem_types: List[ProblemType],
                                     strategy: SelectionStrategy = SelectionStrategy.BALANCED,
                                     max_backends_per_type: int = 3) -> Dict[ProblemType, List[BackendSelection]]:
        """
        Select multiple backends for comprehensive benchmarking.
        
        Args:
            problem_types: List of problem types to benchmark
            strategy: Selection strategy
            max_backends_per_type: Maximum backends to select per problem type
            
        Returns:
            Dictionary mapping problem types to selected backends
        """
        self.logger.info(f"Selecting backends for {len(problem_types)} problem types")
        
        selections = {}
        
        for problem_type in problem_types:
            # Get all compatible backends
            compatible_backends = self.validator.get_backends_for_problem_type(problem_type)
            available_compatible = [b for b in compatible_backends if b in self._available_backends]
            
            type_selections = []
            exclude_list = []
            
            # Select multiple backends for diversity
            for i in range(min(max_backends_per_type, len(available_compatible))):
                selection = self.select_backend_for_problem(
                    problem_type, strategy, exclude_backends=exclude_list
                )
                
                if selection.selected_backend:
                    type_selections.append(selection)
                    exclude_list.append(selection.selected_backend)
                else:
                    break
            
            selections[problem_type] = type_selections
            self.logger.info(f"Selected {len(type_selections)} backends for {problem_type.value}")
        
        return selections
    
    def generate_selection_report(self, selections: Dict[ProblemType, List[BackendSelection]]) -> Dict[str, Any]:
        """Generate comprehensive backend selection report."""
        
        total_selections = sum(len(backends) for backends in selections.values())
        unique_backends = set()
        
        for backend_list in selections.values():
            for selection in backend_list:
                if selection.selected_backend:
                    unique_backends.add(selection.selected_backend)
        
        # Problem type coverage
        coverage_details = {}
        for problem_type, backend_selections in selections.items():
            selected_backends = [s.selected_backend for s in backend_selections if s.selected_backend]
            coverage_details[problem_type.value] = {
                "selected_backends": selected_backends,
                "selection_count": len(selected_backends),
                "confidence_scores": [s.confidence for s in backend_selections],
                "avg_confidence": sum(s.confidence for s in backend_selections) / len(backend_selections) if backend_selections else 0.0
            }
        
        return {
            "summary": {
                "total_problem_types": len(selections),
                "total_selections": total_selections,
                "unique_backends_selected": len(unique_backends),
                "selected_backends": list(unique_backends)
            },
            "coverage_by_problem_type": coverage_details,
            "selection_details": {
                problem_type.value: [
                    {
                        "backend": s.selected_backend,
                        "reason": s.reason,
                        "confidence": s.confidence,
                        "alternatives": s.alternatives
                    } for s in backend_selections
                ] for problem_type, backend_selections in selections.items()
            }
        }


if __name__ == "__main__":
    # Test script to verify backend selection
    try:
        print("Testing Backend Selection System...")
        
        # Initialize components
        validator = SolverValidator()
        selector = BackendSelector(validator)
        
        # Test basic backend availability
        print("\nTesting backend availability:")
        available_backends = selector.get_available_backends()
        print(f"Available backends: {available_backends}")
        
        # Test selection for each problem type
        print("\nTesting backend selection by problem type:")
        for problem_type in ProblemType:
            selection = selector.select_backend_for_problem(
                problem_type, SelectionStrategy.BALANCED
            )
            print(f"  {problem_type.value}: {selection.selected_backend} ({selection.reason})")
            if selection.alternatives:
                print(f"    Alternatives: {selection.alternatives[:3]}")
        
        # Test different selection strategies
        print("\nTesting selection strategies for LP problems:")
        strategies = [SelectionStrategy.FASTEST, SelectionStrategy.MOST_RELIABLE, 
                     SelectionStrategy.BEST_ACCURACY, SelectionStrategy.BALANCED]
        
        for strategy in strategies:
            selection = selector.select_backend_for_problem(ProblemType.LP, strategy)
            print(f"  {strategy.value}: {selection.selected_backend} (confidence: {selection.confidence:.2f})")
        
        # Test user preferences
        print("\nTesting user preferences:")
        user_prefs = ["CLARABEL", "OSQP", "SCS"]
        selection = selector.select_backend_for_problem(
            ProblemType.QP, SelectionStrategy.USER_PREFERENCE, user_preferences=user_prefs
        )
        print(f"  User preference result: {selection.selected_backend} ({selection.reason})")
        
        # Test comprehensive benchmark selection
        print("\nTesting comprehensive benchmark selection:")
        problem_types = [ProblemType.LP, ProblemType.QP]
        benchmark_selections = selector.select_backends_for_benchmark(
            problem_types, max_backends_per_type=2
        )
        
        for pt, selections_list in benchmark_selections.items():
            print(f"  {pt.value}: {[s.selected_backend for s in selections_list]}")
        
        # Generate selection report
        print("\nGenerating selection report:")
        report = selector.generate_selection_report(benchmark_selections)
        print(f"  Total selections: {report['summary']['total_selections']}")
        print(f"  Unique backends: {report['summary']['unique_backends_selected']}")
        print(f"  Selected backends: {report['summary']['selected_backends']}")
        
        print("\n✓ Backend selection system test completed!")
        
    except Exception as e:
        logger.error(f"Backend selection test failed: {e}")
        raise