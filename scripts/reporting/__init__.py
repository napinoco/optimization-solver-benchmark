"""
Simplified HTML Report Generation Module

This module provides functionality to generate simplified HTML reports
from benchmark results stored in the database.
"""

from .html_generator import HTMLGenerator
from .result_processor import ResultProcessor
from .data_exporter import DataExporter

__all__ = ['HTMLGenerator', 'ResultProcessor', 'DataExporter']