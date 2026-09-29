"""Shared solver interface and spin-state optimization."""

from .base import AbstractSolver, SolverResult
from .optimization import SpinOptimizer

__all__ = ['AbstractSolver', 'SolverResult', 'SpinOptimizer']
