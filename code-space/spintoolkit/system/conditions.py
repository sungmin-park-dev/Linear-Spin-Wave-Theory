"""External conditions of a calculation: applied field and temperature.

Both values are dimensionless, in the energy unit ``E0`` of the model's
coefficients:

- ``field`` is ``b = mu_B B / E0``; the Zeeman term is ``-b^T g_a S_a`` with the
  model's dimensionless g-tensor. When ``g`` is unknown, the model uses the
  identity and ``field`` is the Zeeman energy ``h / E0``.
- ``temperature`` is ``k_B T / E0``.

The field enters the Hamiltonian through the ``zeeman`` terms; the
temperature does not enter the Hamiltonian but sets the statistical ensemble.
Neither is part of :class:`SpinModel`, so changing them never rebuilds the model.

Physical units are tesla and kelvin; converting to and from them is left to
the user, e.g. ``B[T] = field * E0[meV] / MU_B_MEV_PER_T`` and
``T[K] = temperature * E0[meV] / K_BOLTZMANN_MEV``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ExternalConditions:
    """Applied field and temperature of one calculation (dimensionless).

    Parameters
    ----------
    field : array_like, shape (3,), optional
        ``mu_B B`` in units of E0, in the global spin frame (default zero).
    temperature : float, optional
        ``k_B T`` in units of E0 (default zero).
    """

    field: Any = (0.0, 0.0, 0.0)
    temperature: float = 0.0

    def __post_init__(self):
        if np.iscomplexobj(self.field):
            raise TypeError("field must be real")
        field = np.array(self.field, dtype=float, copy=True)
        field.setflags(write=False)
        object.__setattr__(self, "field", field)
        object.__setattr__(self, "temperature", float(self.temperature))
        problems = []
        if field.shape != (3,) or not np.all(np.isfinite(field)):
            problems.append("field must be finite with shape (3,)")
        if not (np.isfinite(self.temperature) and self.temperature >= 0):
            problems.append(f"temperature must be finite and non-negative, got {self.temperature}")
        if problems:
            raise ValueError("; ".join(problems))
