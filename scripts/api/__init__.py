"""
API Package
===========

Provides RESTful API endpoints for accessing benchmark data programmatically.
Includes simple Flask-based server for development and testing.
"""

from .simple_api import BenchmarkAPI

__all__ = ['BenchmarkAPI']