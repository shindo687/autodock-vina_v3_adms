import math
from pathlib import Path

import pytest

import vina_ad


PAIR = ((0.0, 0.0, 0.0), (3.0, 0.0, 0.0))
TYPES = (0, 0)  # XS_TYPE_C_H, XS_TYPE_C_H
WEIGHTS = vina_ad.DEFAULT_VINA_WEIGHTS


def source_formula_pair(radius=3.0, types=TYPES, weights=WEIGHTS, torsions=0.0):
    # Independent transcription of potentials.h for the C_H/C_H pair.  The
    # test intentionally does not call sidecar private helpers.
    optimal = 1.9 + 1.9
    delta = radius - optimal
    terms = (
        math.exp(-((delta / 0.5) ** 2)),
        math.exp(-(((delta - 3.0) / 2.0) ** 2)),
        delta * delta if delta <= 0 else 0.0,
        1.0,  # slope_step(1.5, 0.5, r - optimal), delta=-0.8 <= good
        0.0,
        0.0,
    )
    energy = sum(weights[k] * terms[k] for k in range(6))
    return energy / (1.0 + weights[6] * torsions), terms


def test_source_formula_oracle_and_default_weights():
    expected, terms = source_formula_pair()
    actual = vina_ad.score_coordinates(PAIR, TYPES)
    assert actual == pytest.approx(expected, abs=1e-12)
    assert vina_ad.DEFAULT_VINA_WEIGHTS == (-0.035579, -0.005156, 0.840245, -0.035069, -0.587439, 50.0, 0.05846)
    assert terms[0] > 0 and terms[2] > 0


def test_real_vina_binding_oracle(tmp_path: Path):
    vina = pytest.importorskip("vina")
    atom = lambda serial, x: f"ATOM  {serial:5d}  C   LIG A   1    {x:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00     0.000 C \n"
    receptor = tmp_path / "receptor.pdbqt"
    ligand = tmp_path / "ligand.pdbqt"
    receptor.write_text(atom(1, 0.0))
    ligand.write_text("ROOT\n" + atom(1, 3.0) + "ENDROOT\nTORSDOF 0\n")
    real = vina.Vina(sf_name="vina", cpu=1, seed=1, verbosity=0, no_refine=True)
    real.set_receptor(str(receptor))
    real.set_ligand_from_file(str(ligand))
    real.compute_vina_maps(center=[1.5, 0.0, 0.0], box_size=[10.0, 10.0, 10.0], spacing=0.375)
    oracle = float(real.score()[0])
    replay = vina_ad.score_coordinates(PAIR, TYPES)
    assert abs(oracle - replay) <= 0.05


def test_typed_potentials_and_cutoff():
    # Hydrophobic C_H/C_H and donor/acceptor N_D/O_A terms are active; all
    # pair terms are zero at the official 8 A potential cutoff (except glue).
    value = vina_ad.score_coordinates(((0., 0., 0.), (4., 0., 0.)), (0, 0))
    assert math.isfinite(value)
    with pytest.raises(vina_ad.NonDifferentiablePoint):
        vina_ad.score_coordinates(((0., 0., 0.), (8., 0., 0.)), (0, 0))
    assert vina_ad.score_coordinates(((0., 0., 0.), (21., 0., 0.)), (0, 0)) == 0.0
    donor_acceptor = vina_ad.score_coordinates(((0., 0., 0.), (2., 0., 0.)), (3, 8))
    assert donor_acceptor != 0.0


def test_torsion_correction_and_custom_pairs():
    untorsioned = vina_ad.score_coordinates(PAIR, TYPES, torsion_count=0)
    torsioned = vina_ad.score_coordinates(PAIR, TYPES, torsion_count=2)
    assert torsioned == pytest.approx(untorsioned / (1 + WEIGHTS[6] * 2))
    coords = ((0., 0., 0.), (3., 0., 0.), (0., 4., 0.))
    assert vina_ad.score_coordinates(coords, (0, 0, 0), pairs=((0, 1),)) == pytest.approx(untorsioned)


def test_analytic_jvp_vjp_and_finite_difference():
    tangent = ((0.2, -0.1, 0.3), (-0.4, 0.2, 0.1))
    value, jvp_out = vina_ad.jvp(vina_ad.score_coordinates, PAIR, TYPES, tangents={"coordinates": tangent})
    assert value == vina_ad.score_coordinates(PAIR, TYPES)
    h = 1e-6
    plus = [list(row) for row in PAIR]
    minus = [list(row) for row in PAIR]
    for i in range(2):
        for k in range(3):
            plus[i][k] += h * tangent[i][k]
            minus[i][k] -= h * tangent[i][k]
    finite = (vina_ad.score_coordinates(plus, TYPES) - vina_ad.score_coordinates(minus, TYPES)) / (2 * h)
    assert jvp_out == pytest.approx(finite, abs=2e-5)
    value, pullback = vina_ad.vjp(vina_ad.score_coordinates, PAIR, TYPES, wrt=("coordinates", "weights"))
    result = pullback(1.7)
    assert value == vina_ad.score_coordinates(PAIR, TYPES)
    assert set(result) == {"coordinates", "weights"}
    assert result == pullback(1.7)


def test_weight_derivative_and_aliases():
    direction = (0.25, -0.5, 0.75, 0.0, 0.0, 0.0, 0.0)
    _, tangent = vina_ad.jvp(vina_ad.score_coordinates, PAIR, TYPES, tangents={"weights": direction})
    expected = sum(source_formula_pair()[1][k] * direction[k] for k in range(6))
    assert tangent == pytest.approx(expected)
    assert vina_ad.score is vina_ad.score_coordinates
    assert vina_ad.energy is vina_ad.score_coordinates


def test_errors_and_fallback_contract():
    with pytest.raises(ValueError, match="at least two"):
        vina_ad.score_coordinates(((0, 0, 0),), (0,))
    with pytest.raises(ValueError, match="shape"):
        vina_ad.score_coordinates(((0, 0), (1, 1, 1)), (0, 0))
    with pytest.raises(ValueError, match="finite"):
        vina_ad.score_coordinates(((0, 0, 0), (math.inf, 0, 0)), (0, 0))
    with pytest.raises(vina_ad.NonDifferentiablePoint):
        vina_ad.grad(vina_ad.score_coordinates, ((0, 0, 0), (0, 0, 0)), (0, 0), wrt="coordinates")
    with pytest.raises(TypeError, match="Unknown tangent"):
        vina_ad.jvp(vina_ad.score_coordinates, PAIR, TYPES, tangents={"other": 1.0})
    with pytest.raises(vina_ad.UnsupportedWrt) as error:
        vina_ad.vjp(vina_ad.score_coordinates, PAIR, TYPES, wrt="atom_types")
    assert error.value.function is vina_ad.score_coordinates
    assert error.value.requested == ("atom_types",)
    assert set(error.value.supported) == {"coordinates", "weights"}
    with pytest.raises(ValueError, match="length 7"):
        vina_ad.score_coordinates(PAIR, TYPES, weights=(1, 2, 3))
