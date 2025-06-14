"""
Solvers Package
==============

Provides interfaces to various optimization solvers including Python and Octave environments.
"""

from .python.scipy_runner import ScipySolver
from .python.cvxpy_runner import CvxpySolver

# Octave solver (imported conditionally based on availability)
try:
    from .octave.octave_runner import OctaveSolver
    OCTAVE_AVAILABLE = True
except ImportError:
    OCTAVE_AVAILABLE = False
    OctaveSolver = None

__all__ = ['ScipySolver', 'CvxpySolver']

if OCTAVE_AVAILABLE:
    __all__.append('OctaveSolver')