-- Optimization Solver Benchmark Database Schema (Complete)
-- This schema includes all tables and enhancements for the benchmark system

-- Benchmarks table stores execution metadata
CREATE TABLE IF NOT EXISTS benchmarks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    environment_info TEXT  -- JSON string containing environment data
);

-- Problems table stores problem metadata with classification data
CREATE TABLE IF NOT EXISTS problems (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    problem_class TEXT NOT NULL,  -- LP, QP, SDP, SOCP
    file_path TEXT NOT NULL,
    metadata TEXT,  -- JSON string for additional metadata
    
    -- Problem dimensions
    n_variables INTEGER DEFAULT 0,
    n_constraints INTEGER DEFAULT 0,
    n_equality_constraints INTEGER DEFAULT 0,
    n_inequality_constraints INTEGER DEFAULT 0,
    n_special_constraints INTEGER DEFAULT 0,
    
    -- Problem characteristics
    complexity_score REAL DEFAULT 0.0,
    difficulty_level TEXT DEFAULT 'Unknown',
    sparsity_ratio REAL DEFAULT 0.0,
    condition_estimate REAL DEFAULT 1.0,
    
    -- Classification metadata
    classification_confidence REAL DEFAULT 1.0,
    matrix_properties TEXT,  -- JSON string
    constraint_properties TEXT,  -- JSON string
    classified_at DATETIME,
    
    UNIQUE(name, problem_class)
);

-- Solvers table stores solver information
CREATE TABLE IF NOT EXISTS solvers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    version TEXT,
    environment TEXT,  -- python, octave, matlab
    metadata TEXT,  -- JSON string for additional metadata
    UNIQUE(name, version, environment)
);

-- Results table stores individual benchmark results
CREATE TABLE IF NOT EXISTS results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    benchmark_id INTEGER NOT NULL,
    solver_name TEXT NOT NULL,
    problem_name TEXT NOT NULL,
    solve_time REAL,
    status TEXT,
    objective_value REAL,
    duality_gap REAL,
    iterations INTEGER,
    error_message TEXT,
    solver_info TEXT,  -- JSON string for solver-specific information
    FOREIGN KEY (benchmark_id) REFERENCES benchmarks (id)
);

-- Problem classifications table for detailed characteristics cache
CREATE TABLE IF NOT EXISTS problem_classifications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    problem_name TEXT NOT NULL,
    problem_type TEXT NOT NULL,
    n_variables INTEGER NOT NULL,
    n_constraints INTEGER NOT NULL,
    n_equality_constraints INTEGER DEFAULT 0,
    n_inequality_constraints INTEGER DEFAULT 0,
    n_special_constraints INTEGER DEFAULT 0,
    complexity_score REAL NOT NULL,
    difficulty_level TEXT NOT NULL,
    sparsity_ratio REAL DEFAULT 0.0,
    condition_estimate REAL DEFAULT 1.0,
    classification_confidence REAL DEFAULT 1.0,
    matrix_properties TEXT,  -- JSON string
    constraint_properties TEXT,  -- JSON string
    classification_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(problem_name)
);

-- Solver recommendations table
CREATE TABLE IF NOT EXISTS solver_recommendations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    problem_name TEXT NOT NULL,
    solver_name TEXT NOT NULL,
    suitability_score REAL NOT NULL,
    recommendation_rank INTEGER NOT NULL,
    recommendation_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Performance indexes
CREATE INDEX IF NOT EXISTS idx_problems_type ON problems(problem_class);
CREATE INDEX IF NOT EXISTS idx_problems_difficulty ON problems(difficulty_level);
CREATE INDEX IF NOT EXISTS idx_classifications_type ON problem_classifications(problem_type);
CREATE INDEX IF NOT EXISTS idx_classifications_difficulty ON problem_classifications(difficulty_level);
CREATE INDEX IF NOT EXISTS idx_recommendations_problem ON solver_recommendations(problem_name);
CREATE INDEX IF NOT EXISTS idx_recommendations_solver ON solver_recommendations(solver_name);
CREATE INDEX IF NOT EXISTS idx_recommendations_score ON solver_recommendations(suitability_score);