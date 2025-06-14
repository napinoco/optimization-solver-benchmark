"""
Database Storage for Problem Classification System
=================================================

Handles storage and retrieval of problem classification data in the benchmark database.
"""

import sqlite3
import json
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import sys

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.utils.problem_classifier import ProblemCharacteristics, ProblemClassifier
from scripts.utils.logger import get_logger

logger = get_logger("classification_storage")


class ClassificationStorage:
    """Handles database operations for problem classification data."""
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize classification storage.
        
        Args:
            db_path: Path to SQLite database file
        """
        if db_path is None:
            db_path = project_root / "database" / "results.db"
        
        self.db_path = str(db_path)
        self.logger = get_logger("classification_storage")
        
        # Ensure database schema is up to date
        self._ensure_schema()
    
    def _ensure_schema(self):
        """Ensure the database schema includes classification tables."""
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check if classification columns exist
                cursor.execute("PRAGMA table_info(problems)")
                columns = [row[1] for row in cursor.fetchall()]
                
                if 'complexity_score' not in columns:
                    self.logger.info("Adding classification columns to database schema")
                    self._create_classification_tables(cursor)
                    
                    # Try to add columns to existing problems table
                    try:
                        column_additions = [
                            "ALTER TABLE problems ADD COLUMN n_variables INTEGER DEFAULT 0",
                            "ALTER TABLE problems ADD COLUMN n_constraints INTEGER DEFAULT 0",
                            "ALTER TABLE problems ADD COLUMN n_equality_constraints INTEGER DEFAULT 0",
                            "ALTER TABLE problems ADD COLUMN n_inequality_constraints INTEGER DEFAULT 0",
                            "ALTER TABLE problems ADD COLUMN n_special_constraints INTEGER DEFAULT 0",
                            "ALTER TABLE problems ADD COLUMN complexity_score REAL DEFAULT 0.0",
                            "ALTER TABLE problems ADD COLUMN difficulty_level TEXT DEFAULT 'Unknown'",
                            "ALTER TABLE problems ADD COLUMN sparsity_ratio REAL DEFAULT 0.0",
                            "ALTER TABLE problems ADD COLUMN condition_estimate REAL DEFAULT 1.0",
                            "ALTER TABLE problems ADD COLUMN classification_confidence REAL DEFAULT 1.0",
                            "ALTER TABLE problems ADD COLUMN matrix_properties TEXT",
                            "ALTER TABLE problems ADD COLUMN constraint_properties TEXT",
                            "ALTER TABLE problems ADD COLUMN classified_at DATETIME"
                        ]
                        
                        for alter_sql in column_additions:
                            try:
                                cursor.execute(alter_sql)
                            except sqlite3.OperationalError as e:
                                if "duplicate column name" not in str(e).lower():
                                    self.logger.debug(f"Column addition skipped: {e}")
                    except Exception as e:
                        self.logger.debug(f"Problems table column addition failed: {e}")
                else:
                    # Ensure classification tables exist even if columns are already there
                    self._create_classification_tables(cursor)
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to ensure schema: {e}")
    
    def _create_classification_tables(self, cursor):
        """Create classification tables manually."""
        
        cursor.execute("""
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
                matrix_properties TEXT,
                constraint_properties TEXT,
                classification_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(problem_name)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS solver_recommendations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                problem_name TEXT NOT NULL,
                solver_name TEXT NOT NULL,
                suitability_score REAL NOT NULL,
                recommendation_rank INTEGER NOT NULL,
                recommendation_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
    
    def _serialize_for_json(self, obj):
        """Convert numpy types to Python types for JSON serialization."""
        import numpy as np
        
        if isinstance(obj, dict):
            return {key: self._serialize_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._serialize_for_json(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def store_classification(self, characteristics: ProblemCharacteristics) -> bool:
        """
        Store problem classification in the database.
        
        Args:
            characteristics: Problem characteristics to store
            
        Returns:
            True if successful, False otherwise
        """
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Store in problem_classifications table
                cursor.execute("""
                    INSERT OR REPLACE INTO problem_classifications (
                        problem_name, problem_type, n_variables, n_constraints,
                        n_equality_constraints, n_inequality_constraints, n_special_constraints,
                        complexity_score, difficulty_level, sparsity_ratio, condition_estimate,
                        classification_confidence, matrix_properties, constraint_properties
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    characteristics.name,
                    characteristics.problem_type.value,
                    characteristics.n_variables,
                    characteristics.n_constraints,
                    characteristics.n_equality_constraints,
                    characteristics.n_inequality_constraints,
                    characteristics.n_special_constraints,
                    characteristics.complexity_score,
                    characteristics.difficulty_level.value,
                    characteristics.sparsity_ratio,
                    characteristics.condition_estimate,
                    characteristics.classification_confidence,
                    json.dumps(self._serialize_for_json(characteristics.matrix_properties)),
                    json.dumps(self._serialize_for_json(characteristics.constraint_properties))
                ))
                
                # Update problems table if it exists
                cursor.execute("""
                    UPDATE problems SET 
                        n_variables = ?,
                        n_constraints = ?,
                        n_equality_constraints = ?,
                        n_inequality_constraints = ?,
                        n_special_constraints = ?,
                        complexity_score = ?,
                        difficulty_level = ?,
                        sparsity_ratio = ?,
                        condition_estimate = ?,
                        classification_confidence = ?,
                        matrix_properties = ?,
                        constraint_properties = ?,
                        classified_at = CURRENT_TIMESTAMP
                    WHERE name = ?
                """, (
                    characteristics.n_variables,
                    characteristics.n_constraints,
                    characteristics.n_equality_constraints,
                    characteristics.n_inequality_constraints,
                    characteristics.n_special_constraints,
                    characteristics.complexity_score,
                    characteristics.difficulty_level.value,
                    characteristics.sparsity_ratio,
                    characteristics.condition_estimate,
                    characteristics.classification_confidence,
                    json.dumps(self._serialize_for_json(characteristics.matrix_properties)),
                    json.dumps(self._serialize_for_json(characteristics.constraint_properties)),
                    characteristics.name
                ))
                
                conn.commit()
                self.logger.debug(f"Stored classification for {characteristics.name}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to store classification for {characteristics.name}: {e}")
            return False
    
    def store_solver_recommendations(self, problem_name: str, 
                                   recommendations: List[Tuple[str, float]]) -> bool:
        """
        Store solver recommendations for a problem.
        
        Args:
            problem_name: Name of the problem
            recommendations: List of (solver_name, suitability_score) tuples
            
        Returns:
            True if successful, False otherwise
        """
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Clear existing recommendations
                cursor.execute("DELETE FROM solver_recommendations WHERE problem_name = ?", 
                             (problem_name,))
                
                # Store new recommendations
                for rank, (solver_name, score) in enumerate(recommendations, 1):
                    cursor.execute("""
                        INSERT INTO solver_recommendations (
                            problem_name, solver_name, suitability_score, recommendation_rank
                        ) VALUES (?, ?, ?, ?)
                    """, (problem_name, solver_name, score, rank))
                
                conn.commit()
                self.logger.debug(f"Stored {len(recommendations)} recommendations for {problem_name}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to store recommendations for {problem_name}: {e}")
            return False
    
    def get_classification(self, problem_name: str) -> Optional[ProblemCharacteristics]:
        """
        Retrieve problem classification from database.
        
        Args:
            problem_name: Name of the problem
            
        Returns:
            ProblemCharacteristics if found, None otherwise
        """
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT problem_type, n_variables, n_constraints,
                           n_equality_constraints, n_inequality_constraints, n_special_constraints,
                           complexity_score, difficulty_level, sparsity_ratio, condition_estimate,
                           classification_confidence, matrix_properties, constraint_properties
                    FROM problem_classifications 
                    WHERE problem_name = ?
                """, (problem_name,))
                
                row = cursor.fetchone()
                if row is None:
                    return None
                
                # Parse the data
                from scripts.utils.problem_classifier import ProblemType, DifficultyLevel
                
                characteristics = ProblemCharacteristics(
                    name=problem_name,
                    problem_type=ProblemType(row[0]),
                    n_variables=row[1],
                    n_constraints=row[2],
                    n_equality_constraints=row[3],
                    n_inequality_constraints=row[4],
                    n_special_constraints=row[5],
                    complexity_score=row[6],
                    difficulty_level=DifficultyLevel(row[7]),
                    sparsity_ratio=row[8],
                    condition_estimate=row[9],
                    classification_confidence=row[10],
                    matrix_properties=json.loads(row[11]) if row[11] else {},
                    constraint_properties=json.loads(row[12]) if row[12] else {}
                )
                
                return characteristics
                
        except Exception as e:
            self.logger.error(f"Failed to get classification for {problem_name}: {e}")
            return None
    
    def get_solver_recommendations(self, problem_name: str) -> List[Tuple[str, float]]:
        """
        Retrieve solver recommendations for a problem.
        
        Args:
            problem_name: Name of the problem
            
        Returns:
            List of (solver_name, suitability_score) tuples, sorted by rank
        """
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT solver_name, suitability_score
                    FROM solver_recommendations 
                    WHERE problem_name = ?
                    ORDER BY recommendation_rank
                """, (problem_name,))
                
                return cursor.fetchall()
                
        except Exception as e:
            self.logger.error(f"Failed to get recommendations for {problem_name}: {e}")
            return []
    
    def get_classification_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all problem classifications.
        
        Returns:
            Dictionary containing classification statistics
        """
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get type distribution
                cursor.execute("""
                    SELECT problem_type, COUNT(*) 
                    FROM problem_classifications 
                    GROUP BY problem_type
                """)
                type_distribution = dict(cursor.fetchall())
                
                # Get difficulty distribution
                cursor.execute("""
                    SELECT difficulty_level, COUNT(*) 
                    FROM problem_classifications 
                    GROUP BY difficulty_level
                """)
                difficulty_distribution = dict(cursor.fetchall())
                
                # Get complexity statistics
                cursor.execute("""
                    SELECT AVG(complexity_score), MIN(complexity_score), MAX(complexity_score),
                           AVG(n_variables), MIN(n_variables), MAX(n_variables),
                           AVG(n_constraints), MIN(n_constraints), MAX(n_constraints)
                    FROM problem_classifications
                """)
                stats = cursor.fetchone()
                
                summary = {
                    "total_problems": sum(type_distribution.values()),
                    "type_distribution": type_distribution,
                    "difficulty_distribution": difficulty_distribution,
                    "complexity_stats": {
                        "avg_complexity": stats[0] if stats[0] else 0,
                        "min_complexity": stats[1] if stats[1] else 0,
                        "max_complexity": stats[2] if stats[2] else 0,
                        "avg_variables": stats[3] if stats[3] else 0,
                        "min_variables": stats[4] if stats[4] else 0,
                        "max_variables": stats[5] if stats[5] else 0,
                        "avg_constraints": stats[6] if stats[6] else 0,
                        "min_constraints": stats[7] if stats[7] else 0,
                        "max_constraints": stats[8] if stats[8] else 0
                    }
                }
                
                return summary
                
        except Exception as e:
            self.logger.error(f"Failed to get classification summary: {e}")
            return {}
    
    def classify_and_store_all_problems(self) -> Dict[str, bool]:
        """
        Classify all problems in the registry and store results.
        
        Returns:
            Dictionary mapping problem names to success status
        """
        
        from scripts.utils.problem_classifier import analyze_problem_registry
        
        self.logger.info("Classifying and storing all problems...")
        
        # Get all problem characteristics
        problem_characteristics = analyze_problem_registry()
        
        # Store each classification
        results = {}
        classifier = ProblemClassifier()
        
        for name, characteristics in problem_characteristics.items():
            # Store classification
            success = self.store_classification(characteristics)
            results[name] = success
            
            if success:
                # Store solver recommendations
                recommendations = classifier.recommend_solvers(characteristics)
                self.store_solver_recommendations(name, recommendations)
        
        successful = sum(1 for success in results.values() if success)
        self.logger.info(f"Successfully classified and stored {successful}/{len(results)} problems")
        
        return results


def update_problem_classifications():
    """Utility function to update all problem classifications in the database."""
    
    storage = ClassificationStorage()
    results = storage.classify_and_store_all_problems()
    
    print("Problem Classification Update Results:")
    print(f"Total problems: {len(results)}")
    print(f"Successful: {sum(1 for success in results.values() if success)}")
    print(f"Failed: {sum(1 for success in results.values() if not success)}")
    
    # Print summary
    summary = storage.get_classification_summary()
    if summary:
        print(f"\nClassification Summary:")
        print(f"Total classified problems: {summary['total_problems']}")
        
        print("\nType distribution:")
        for ptype, count in summary['type_distribution'].items():
            print(f"  {ptype}: {count}")
        
        print("\nDifficulty distribution:")
        for difficulty, count in summary['difficulty_distribution'].items():
            print(f"  {difficulty}: {count}")


if __name__ == "__main__":
    # Test the classification storage system
    print("Testing Classification Storage System...")
    
    # Update all classifications
    update_problem_classifications()
    
    # Test retrieval
    print(f"\n=== Testing Retrieval ===")
    storage = ClassificationStorage()
    
    # Get a specific classification
    characteristics = storage.get_classification("simple_lp")
    if characteristics:
        print(f"Retrieved simple_lp: {characteristics.problem_type.value}, {characteristics.difficulty_level.value}")
        
        # Get recommendations
        recommendations = storage.get_solver_recommendations("simple_lp")
        print(f"Recommendations for simple_lp:")
        for solver, score in recommendations[:3]:
            print(f"  {solver}: {score:.3f}")
    
    print("\n✓ Classification storage system test completed!")