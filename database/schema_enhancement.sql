-- Schema Enhancement for Problem Classification System
-- Add new columns to problems table for classification data

-- Add classification columns to problems table
ALTER TABLE problems ADD COLUMN n_variables INTEGER DEFAULT 0;
ALTER TABLE problems ADD COLUMN n_constraints INTEGER DEFAULT 0;
ALTER TABLE problems ADD COLUMN n_equality_constraints INTEGER DEFAULT 0;
ALTER TABLE problems ADD COLUMN n_inequality_constraints INTEGER DEFAULT 0;
ALTER TABLE problems ADD COLUMN n_special_constraints INTEGER DEFAULT 0;

-- Problem characteristics
ALTER TABLE problems ADD COLUMN complexity_score REAL DEFAULT 0.0;
ALTER TABLE problems ADD COLUMN difficulty_level TEXT DEFAULT 'Unknown';
ALTER TABLE problems ADD COLUMN sparsity_ratio REAL DEFAULT 0.0;
ALTER TABLE problems ADD COLUMN condition_estimate REAL DEFAULT 1.0;

-- Classification metadata
ALTER TABLE problems ADD COLUMN classification_confidence REAL DEFAULT 1.0;
ALTER TABLE problems ADD COLUMN matrix_properties TEXT; -- JSON string
ALTER TABLE problems ADD COLUMN constraint_properties TEXT; -- JSON string

-- Add timestamp for classification
ALTER TABLE problems ADD COLUMN classified_at DATETIME;

-- Create problem_classification_cache table for detailed characteristics
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
    matrix_properties TEXT, -- JSON string
    constraint_properties TEXT, -- JSON string
    classification_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(problem_name)
);

-- Create solver_recommendations table
CREATE TABLE IF NOT EXISTS solver_recommendations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    problem_name TEXT NOT NULL,
    solver_name TEXT NOT NULL,
    suitability_score REAL NOT NULL,
    recommendation_rank INTEGER NOT NULL,
    recommendation_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (problem_name) REFERENCES problems (name)
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_problems_type ON problems(problem_class);
CREATE INDEX IF NOT EXISTS idx_problems_difficulty ON problems(difficulty_level);
CREATE INDEX IF NOT EXISTS idx_classifications_type ON problem_classifications(problem_type);
CREATE INDEX IF NOT EXISTS idx_classifications_difficulty ON problem_classifications(difficulty_level);
CREATE INDEX IF NOT EXISTS idx_recommendations_problem ON solver_recommendations(problem_name);
CREATE INDEX IF NOT EXISTS idx_recommendations_solver ON solver_recommendations(solver_name);
CREATE INDEX IF NOT EXISTS idx_recommendations_score ON solver_recommendations(suitability_score);