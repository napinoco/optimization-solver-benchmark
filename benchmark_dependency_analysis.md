# Benchmark Execution Dependency Graph Analysis

## Call Flow Diagram

```mermaid
graph TD
    A[main.py] --> B[run_benchmark()]
    B --> C[Load YAML Registries 30ms]
    B --> D[Create DatabaseManager]
    B --> E[Create BenchmarkRunner 300ms]
    
    E --> F[collect_environment_info() 200ms]
    E --> G[get_git_commit_hash() 50ms]
    E --> H[Load solver registry again 30ms]
    
    B --> I[Filter Problems]
    B --> J[Filter Solvers + Test Availability 300ms]
    B --> K[Main Benchmark Loop]
    
    K --> L[load_problem()]
    L --> M[MAT/DAT Loaders]
    M --> N[scipy.io.loadmat / Custom parsing]
    
    K --> O[create_solver()]
    O --> P[ScipySolver / CvxpySolver]
    
    K --> Q[validate_compatibility()]
    K --> R[solver.solve()]
    K --> S[store_result()]
    
    S --> T[Database INSERT]
    
    style C fill:#ffcccc
    style E fill:#ffcccc  
    style F fill:#ff9999
    style G fill:#ff9999
    style H fill:#ffcccc
    style J fill:#ff9999
```

## Identified Waste Areas

### 🔴 Critical Waste (High Impact)

1. **Environment Collection (200ms per run)**
   - `collect_environment_info()` runs on every BenchmarkRunner init
   - Performs extensive system inspection: CPU, memory, disk, timezone
   - **Total waste for 695 combinations: ~2.3 minutes**

2. **Solver Capability Detection (300ms per solver)**
   - Tests solver availability with trial problems
   - Runs during solver filtering phase
   - **Total waste: ~25 minutes for initial testing**

3. **Registry Loading Redundancy (30ms per run)**
   - YAML files loaded in main.py AND BenchmarkRunner
   - **Total waste: ~35 seconds**

### 🟡 Medium Waste (Medium Impact)

4. **Git Operations (50ms per session)**
   - Subprocess calls to get commit hash
   - Called on every BenchmarkRunner init

5. **Import Overhead**
   - Heavy modules like `scipy.io`, `cvxpy` loaded eagerly
   - Could be lazy-loaded when needed

### 🟢 Minor Waste (Low Impact)

6. **Path Management Duplication**
   - `sys.path.insert()` repeated in every module

7. **Logging Setup Overhead**
   - Multiple logger creation checks

## Performance Impact

### Current Overhead per Run:
- Environment: 200ms
- Registry loading: 30ms  
- Git ops: 50ms (once)
- Solver testing: 300ms (per solver)
- **Total per benchmark: ~230ms**

### Total Waste for Full Benchmark Suite:
- **695 combinations × 230ms = ~2.7 minutes of unnecessary overhead**
- **Initial solver testing: ~25 minutes**
- **Combined waste: ~28 minutes** (significant!)

## Optimization Priority

### Priority 1: Environment Caching
```python
# Current: Runs every time
def __init__(self, db_manager, dry_run=False):
    self.environment_info = collect_environment_info()  # 200ms!
    
# Optimized: Cache globally
_environment_cache = None
def get_cached_environment():
    global _environment_cache
    if _environment_cache is None:
        _environment_cache = collect_environment_info()
    return _environment_cache
```

### Priority 2: Registry Caching
```python
# Current: Load multiple times
with open(registry_path, 'r') as f:
    registry = yaml.safe_load(f)  # 30ms per call
    
# Optimized: Load once, pass around
registries = load_all_registries_once()
runner = BenchmarkRunner(db_manager, registries=registries)
```

### Priority 3: Solver Capability Caching
```python
# Current: Test every time
def test_solver_availability(solver_name):
    try:
        solver = create_solver(solver_name)  # 300ms test!
        return True
    except:
        return False
        
# Optimized: Static capability matrix or cached results
SOLVER_CAPABILITIES = {
    'cvxpy_clarabel': ['LP', 'QP', 'SOCP', 'SDP'],
    'cvxpy_scs': ['LP', 'QP', 'SOCP', 'SDP'],
    # ...
}
```