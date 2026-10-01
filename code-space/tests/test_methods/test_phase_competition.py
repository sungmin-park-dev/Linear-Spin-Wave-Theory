"""Candidate-state comparison at harmonic order (D42).

Triangular Heisenberg antiferromagnet, S = 1/2: the 120 degree state is
stable with E_cl + E_zp = -0.5388 J per spin (Chubukov, Sachdev and Senthil,
J. Phys. Cond. Matt. 6, 8891 (1994): E = -0.5388 J); the collinear stripe
(a saddle) and the ferromagnet (a maximum) are unstable and carry no
zero-point energy; meshes of different cells have matched density.
"""

import numpy as np
import pytest

from spintoolkit.methods.lswt import LSWTSettings, solve_lswt
from spintoolkit.methods.phase_competition import compare_states, magnetic_mesh
from spintoolkit.models import neel_state, polarized_state, state_120, triangular_heisenberg


def test_triangular_heisenberg_ranking():
    model = triangular_heisenberg(J=1.0, S=0.5)
    candidates = {"fm": polarized_state(model), "stripe": neel_state(model, (1, 0, 0)),
                  "120": state_120(model)}
    reports = compare_states(model, candidates, k_density=36)
    assert [r.name for r in reports] == ["120", "stripe", "fm"]
    best = reports[0]
    assert best.status == "stable" and best.max_torque < 1e-12
    assert best.classical_energy == pytest.approx(-0.375, abs=1e-14)
    assert best.harmonic_energy == pytest.approx(-0.5388, abs=2e-4)
    direct = solve_lswt(model, state_120(model), settings=LSWTSettings(mesh=best.mesh))
    assert best.harmonic_energy == pytest.approx(direct.ground_state_energy, abs=1e-14)
    assert np.isnan(best.skyrmion.charge)                       # coplanar 120: undefined
    for r in reports[1:]:
        assert r.status == "unstable" and np.isnan(r.zero_point_energy)
        assert "negative mode" in r.message
    fm = reports[2]
    np.testing.assert_allclose(fm.magnetization, [0, 0, 0.5])
    assert fm.skyrmion.integer == 0
    assert reports[1].to_dict()["harmonic_energy"] is None


def test_mesh_density_is_matched():
    model = triangular_heisenberg()
    assert magnetic_mesh(polarized_state(model), 24) == (24, 24)
    assert magnetic_mesh(state_120(model), 24) == (14, 14)          # 24 / sqrt(3) -> 14
    assert magnetic_mesh(neel_state(model), 24) == (17, 17)
