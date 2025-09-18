"""
Utilities module for multi-agent collaborative learning.

Contains logging, visualization, and helper utilities for the library.
"""

from .logging import setup_logger
from .visualization import plot_learning_curves

__all__ = [
    'setup_logger',
    'plot_learning_curves'
]