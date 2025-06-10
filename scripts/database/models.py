import sqlite3
import os
from pathlib import Path

def get_database_path():
    """Get the path to the database file."""
    project_root = Path(__file__).parent.parent.parent
    return project_root / "database" / "results.db"

def create_database():
    """Create the database and tables using the schema file."""
    db_path = get_database_path()
    
    # Ensure database directory exists
    db_path.parent.mkdir(exist_ok=True)
    
    # Read schema file
    schema_path = db_path.parent / "schema.sql"
    with open(schema_path, 'r') as f:
        schema_sql = f.read()
    
    # Create database and tables
    with sqlite3.connect(db_path) as conn:
        conn.executescript(schema_sql)
        conn.commit()
    
    return db_path

def get_connection():
    """Get a database connection."""
    db_path = get_database_path()
    return sqlite3.connect(db_path)

if __name__ == "__main__":
    # Test script to create database
    db_path = create_database()
    print(f"Database created at: {db_path}")
    
    # Verify tables exist
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print("Tables created:", [table[0] for table in tables])