"""SpinModel construction, validation and fingerprint (transfer contract)."""

import numpy as np
import pytest

from spintoolkit.system.model import (
    Site, SpinModel, SpinModelError, Term, Units,
)

SQUARE = [[1.0, 0.0], [0.0, 1.0]]
META = {"model_id": "test"}


def make(terms, sites=None, lattice=SQUARE, units=None, metadata=META):
    sites = sites or [Site("A", (0, 0), 0.5), Site("B", (0.5, 0.5), 0.5)]
    return SpinModel(lattice, sites, terms, units or Units(), metadata)


def violations(**kwargs):
    with pytest.raises(SpinModelError) as info:
        make(**kwargs)
    return info.value.violations


def test_valid_model_exposes_sites_and_terms():
    J = np.diag([1.0, 1.0, 0.5])
    model = make([Term.bilinear(("A", (0, 0)), ("B", (0, 0)), J, "NN"),
                  Term.zeeman("A", 2 * np.eye(3))])
    assert model.site_ids == ("A", "B")
    assert model.site("B").spin == 0.5
    assert len(model.terms_of_kind("bilinear")) == 1
    np.testing.assert_allclose(model.cartesian_position("B", (1, 0)), [1.5, 0.5])
    assert model.metadata["parameters"] == {}


@pytest.mark.parametrize("units, expected", [
    (Units(energy="eV"), "energy unit"),
    (Units(length="nm"), "length unit"),
    (Units(energy="meV", energy_scale_meV=2.0), "only allowed with relative"),
    (Units(energy_scale_meV=-1.0), "positive and finite"),
])
def test_invalid_units_are_rejected(units, expected):
    found = violations(terms=[], units=units)
    assert any(expected in v for v in found)


def test_duplicate_zeeman_term_is_rejected():
    found = violations(terms=[Term.zeeman("A", np.eye(3)), Term.zeeman("A", 2 * np.eye(3))])
    assert any("more than one zeeman" in v for v in found)


def test_zeeman_term_needs_zero_offset():
    term = Term("zeeman", (("A", (1, 0)),), np.eye(3))
    assert any("(0, 0)" in v for v in violations(terms=[term]))


def test_unsupported_kind_is_rejected_not_ignored():
    term = Term("biquadratic", (("A", (0, 0)), ("B", (0, 0))), np.ones((3, 3)))
    assert any("not supported" in v for v in violations(terms=[term]))


def test_missing_endpoint_is_rejected():
    term = Term.bilinear(("A", (0, 0)), ("C", (0, 0)), np.eye(3))
    assert any("'C' is not a site" in v for v in violations(terms=[term]))


def test_reversed_bilinear_record_is_a_duplicate():
    first = Term.bilinear(("A", (0, 0)), ("B", (1, 0)), np.eye(3))
    reversed_ = Term.bilinear(("B", (0, 0)), ("A", (-1, 0)), np.eye(3))
    assert any("duplicates term 0" in v for v in violations(terms=[first, reversed_]))


def test_translated_bilinear_record_is_a_duplicate():
    first = Term.bilinear(("A", (0, 0)), ("B", (-1, 0)), np.eye(3))
    translated = Term.bilinear(("A", (2, 3)), ("B", (1, 3)), np.eye(3))
    assert any("duplicates term 0" in v for v in violations(terms=[first, translated]))


def test_same_physical_site_twice_is_rejected():
    term = Term.bilinear(("A", (1, 1)), ("A", (1, 1)), np.eye(3))
    assert any("same physical site" in v for v in violations(terms=[term]))


def test_fractional_cell_offset_is_rejected_not_rounded():
    term = Term.bilinear(("A", (0, 0)), ("B", (0.5, 0)), np.eye(3))
    assert any("exact integers" in v for v in violations(terms=[term]))


def test_asymmetric_exchange_and_same_basis_intercell_bonds_are_allowed():
    dm = np.array([[0, 0.3, 0], [-0.3, 0, 0], [0, 0, 0]])
    model = make([Term.bilinear(("A", (0, 0)), ("A", (1, 0)), np.eye(3) + dm),
                  Term.bilinear(("A", (0, 0)), ("A", (0, 1)), np.eye(3))])
    assert not np.allclose(model.terms[0].coefficient, model.terms[0].coefficient.T)


def test_inputs_are_copied_and_model_arrays_are_read_only():
    J = np.eye(3)
    position = np.array([0.0, 0.0])
    model = make([Term.bilinear(("A", (0, 0)), ("A", (1, 0)), J)],
                 sites=[Site("A", position, 0.5)])
    J[0, 0] = 99.0
    position[0] = 0.25
    assert model.terms[0].coefficient[0, 0] == 1.0
    assert model.sites[0].position[0] == 0.0
    with pytest.raises(ValueError):
        model.terms[0].coefficient[0, 0] = 5.0
    with pytest.raises(ValueError):
        model.lattice[0, 0] = 2.0


def test_all_violations_are_reported_together():
    found = violations(
        terms=[Term.bilinear(("A", (0, 0)), ("C", (0, 0)), np.eye(3)),
               Term.zeeman("A", np.eye(3)), Term.zeeman("A", np.eye(3))],
        sites=[Site("A", (0, 0), 0.3), Site("A", (0, 0), 0.5)],
        lattice=[[1, 0], [2, 0]],
        metadata={},
    )
    for expected in ["model_id", "linearly independent", "half-integer",
                     "duplicated", "'C' is not a site", "more than one zeeman"]:
        assert any(expected in v for v in found), expected


def test_complex_and_non_finite_coefficients():
    with pytest.raises(TypeError):
        Term.bilinear(("A", (0, 0)), ("B", (0, 0)), np.eye(3) * 1j)
    bad = Term.bilinear(("A", (0, 0)), ("B", (0, 0)), np.full((3, 3), np.nan))
    assert any("non-finite" in v for v in violations(terms=[bad]))


def test_fingerprint_depends_only_on_physical_content():
    J = np.array([[1.0, 0.2, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.5]])
    forward = Term.bilinear(("A", (0, 0)), ("B", (1, 0)), J, "x")
    reversed_ = Term.bilinear(("B", (2, 0)), ("A", (1, 0)), J.T, "other label")
    zeeman = Term.zeeman("B", np.eye(3))
    base = make([forward, zeeman])
    assert base.fingerprint() == make([zeeman, forward]).fingerprint()
    assert base.fingerprint() == make([reversed_, zeeman]).fingerprint()
    assert base.fingerprint() == make([forward, zeeman],
                                      metadata={"model_id": "renamed"}).fingerprint()
    changed = Term.bilinear(("A", (0, 0)), ("B", (1, 0)), J * 1.0000001)
    assert base.fingerprint() != make([changed, zeeman]).fingerprint()
    assert base.fingerprint() != make([forward, zeeman],
                                      units=Units(energy="meV")).fingerprint()
