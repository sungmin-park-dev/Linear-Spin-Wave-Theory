"""Existing ladder-to-Cartesian spin basis convention."""

import numpy as np

# Ladder operators to Cartesian: (S+, S-, Sz) -> (Sx, Sy, Sz)
Mat_C = np.array([
    [1 / np.sqrt(2), 1 / np.sqrt(2), 0],       # (S+ + S-)/2   -> x
    [1j / np.sqrt(2), -1j / np.sqrt(2), 0],     # (S+ - S-)/2j  -> y
    [0, 0, 1],                                    # Sz            -> z
])
