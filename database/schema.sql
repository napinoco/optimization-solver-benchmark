-- Optimization Solver Benchmark Database Schema

-- Benchmarks table stores execution metadata
CREATE TABLE IF NOT EXISTS benchmarks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    environment_info TEXT  -- JSON string containing environment data
);

-- Problems table stores problem metadata
CREATE TABLE IF NOT EXISTS problems (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    problem_class TEXT NOT NULL,  -- LP, QP, SDP, SOCP
    file_path TEXT NOT NULL
);

-- Solvers table stores solver information
CREATE TABLE IF NOT EXISTS solvers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    version TEXT,
    environment TEXT  -- python, octave, matlab
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
    FOREIGN KEY (benchmark_id) REFERENCES benchmarks (id)
);