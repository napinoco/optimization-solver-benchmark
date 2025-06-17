-- Re-Architected Database Schema: Single Denormalized Table
-- Simplified design for easier querying and historical retention

-- Single denormalized results table with historical retention
CREATE TABLE results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    
    -- Solver information
    solver_name TEXT NOT NULL,
    solver_version TEXT NOT NULL,
    
    -- Problem information  
    problem_library TEXT NOT NULL,        -- 'internal', 'DIMACS', 'SDPLIB'
    problem_name TEXT NOT NULL,
    problem_type TEXT NOT NULL,           -- 'LP', 'QP', 'SOCP', 'SDP'
    
    -- Environment and execution context
    environment_info TEXT NOT NULL,      -- JSON string with system info
    commit_hash TEXT NOT NULL,           -- Git commit hash
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    
    -- Standardized solver results
    solve_time REAL,                     -- Execution time in seconds
    status TEXT,                         -- 'optimal', 'infeasible', 'error', etc.
    primal_objective_value REAL,        -- Primal objective value
    dual_objective_value REAL,          -- Dual objective value (if available)
    duality_gap REAL,                   -- Duality gap
    primal_infeasibility REAL,          -- Primal infeasibility measure
    dual_infeasibility REAL,            -- Dual infeasibility measure
    iterations INTEGER,                  -- Number of solver iterations
    
    -- Unique constraint to prevent exact duplicates
    UNIQUE(solver_name, solver_version, problem_library, problem_name, commit_hash, timestamp)
);

-- Index for efficient latest results queries
CREATE INDEX idx_latest_results ON results(commit_hash, environment_info, timestamp DESC);
CREATE INDEX idx_solver_problem ON results(solver_name, problem_name);
CREATE INDEX idx_problem_type ON results(problem_type);
CREATE INDEX idx_timestamp ON results(timestamp DESC);