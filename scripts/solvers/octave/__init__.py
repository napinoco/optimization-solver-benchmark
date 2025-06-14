"""
Octave Solver Package
====================

Provides MATLAB-compatible optimization environment using GNU Octave.
Supports linear programming, quadratic programming, and basic optimization.
"""

from .octave_runner import OctaveSolver

__all__ = ['OctaveSolver']