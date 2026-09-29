"""External conditions of a calculation: applied field and temperature.

The field ``B`` enters the Hamiltonian through the g-tensor of each ``zeeman``
term (``-mu_B B^T g S``); the temperature ``T`` does not enter the Hamiltonian
but sets the statistical ensemble. Neither is part of :class:`SpinModel`, so
changing them never rebuilds the model.

Unit rules
----------
- meV models take ``B`` in tesla with ``mu_B = MU_B_MEV_PER_T`` and ``T`` in
  kelvin with ``k_B = K_BOLTZMANN_MEV``.
- Relative-unit models take ``B`` and ``T`` in the model's energy unit
  (``mu_B = k_B = 1``). Tesla or kelvin are accepted only when the model
  declares ``energy_scale_meV``; unknown conversion factors are not guessed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from spintoolkit.definitions.constants import K_BOLTZMANN_MEV, MU_B_MEV_PER_T
from spintoolkit.system.model import Units

FIELD_UNITS = ("T", "relative")
TEMPERATURE_UNITS = ("K", "relative")


@dataclass(frozen=True)
class ExternalConditions:
    """Applied field and temperature of one calculation.

    Parameters
    ----------
    B : array_like, shape (3,), optional
        Applied field in the global frame (default zero).
    B_unit : {"relative", "T"}, optional
    T : float, optional
        Temperature (default zero).
    T_unit : {"relative", "K"}, optional
    """

    B: Any = (0.0, 0.0, 0.0)
    B_unit: str = "relative"
    T: float = 0.0
    T_unit: str = "relative"

    def __post_init__(self):
        if np.iscomplexobj(self.B):
            raise TypeError("B must be real")
        field = np.array(self.B, dtype=float, copy=True)
        field.setflags(write=False)
        object.__setattr__(self, "B", field)
        object.__setattr__(self, "T", float(self.T))
        problems = []
        if field.shape != (3,) or not np.all(np.isfinite(field)):
            problems.append("B must be finite with shape (3,)")
        if self.B_unit not in FIELD_UNITS:
            problems.append(f"B_unit {self.B_unit!r} is not one of {FIELD_UNITS}")
        if self.T_unit not in TEMPERATURE_UNITS:
            problems.append(f"T_unit {self.T_unit!r} is not one of {TEMPERATURE_UNITS}")
        if not (np.isfinite(self.T) and self.T >= 0):
            problems.append(f"T must be finite and non-negative, got {self.T}")
        if problems:
            raise ValueError("; ".join(problems))

    def zeeman_field(self, units: Units) -> np.ndarray:
        """Return ``mu_B B`` in the model's energy unit.

        Raises
        ------
        ValueError
            If the field unit cannot be converted to the model's energy unit.
        """
        return self.B * _scale(units, self.B_unit, "T", MU_B_MEV_PER_T, "B")

    def thermal_energy(self, units: Units) -> float:
        """Return ``k_B T`` in the model's energy unit.

        Raises
        ------
        ValueError
            If the temperature unit cannot be converted to the model's energy unit.
        """
        return self.T * _scale(units, self.T_unit, "K", K_BOLTZMANN_MEV, "T")


def _scale(units, given, physical, constant_mev, name):
    if units.energy == "meV":
        if given == physical:
            return constant_mev
        raise ValueError(f"meV models need {name} in {physical}, got {given!r}")
    if given == "relative":
        return 1.0
    if units.energy_scale_meV is None:
        raise ValueError(f"{name} in {physical} needs energy_scale_meV for a relative-unit model")
    return constant_mev / units.energy_scale_meV
