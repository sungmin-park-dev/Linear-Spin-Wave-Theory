"""Shared solver interface, classical search and optimization, the Luttinger-Tisza
diagnostic, classical spin dynamics, candidate-state comparison, magnetization curves and
local stability. LSWT, nonlinear spin waves and ED are the subpackages ``lswt``, ``nlswt``
and ``ed``."""

from .base import AbstractSolver, SolverResult
from .dynamics import (ClassicalStructureFactor, ClassicalTorus, ImplicitMidpoint, Langevin,
                       classical_structure_factor, evolve, thermal_samples)
from .monte_carlo import MonteCarlo, ThermalAverages, order_parameter, thermal_averages, twist_derivatives
from .luttinger_tisza import LTReport, LTWaveVector, luttinger_tisza
from .optimization import SpinOptimizer
from .classical import ClassicalSearchResult, classical_energy, classical_search
from .magnetization import MagnetizationCurve, magnetization_curve
from .phase_competition import CandidateReport, compare_states
from .stability import StabilityReport, local_stability

__all__ = ['AbstractSolver', 'SolverResult', 'SpinOptimizer', 'luttinger_tisza', 'LTReport',
           'LTWaveVector', 'ClassicalTorus', 'ImplicitMidpoint', 'Langevin', 'evolve',
           'thermal_samples', 'classical_structure_factor', 'ClassicalStructureFactor', 'MonteCarlo', 'ThermalAverages', 'thermal_averages',
           'order_parameter', 'twist_derivatives',
           'ClassicalSearchResult', 'classical_energy', 'classical_search',
           'MagnetizationCurve', 'magnetization_curve', 'CandidateReport', 'compare_states',
           'StabilityReport', 'local_stability']
