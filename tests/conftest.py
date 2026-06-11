"""
Shared pytest configuration for the benchmark system test suite.

Ensures the project root is importable so tests can use the same
`from scripts....` imports as the production code.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
