"""A faithful, restricted replay of AutoDock Vina's SF_VINA score.

The upstream scorer is a C++ object hidden behind ``vina_wrapper``.  This
module keeps the original callable as the primal source while exposing the
source-level pair potentials for fixed atom types and pair topology.  The
formula and constants are mapped in ``SPEC.md`` to ``scoring_function.h`` and
``potentials.h`` in the immutable upstream snapshot.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from .protocol import NonDifferentiablePoint, UnsupportedWrt, ZERO, rules

# Public Vina::set_vina_weights defaults (the seventh value is weight_rot).
DEFAULT_VINA_WEIGHTS = (
    -0.035579,
    -0.005156,
    0.840245,
    -0.035069,
    -0.587439,
    50.0,
    0.05846,
)

# X-Score atom-type radii copied from upstream/src/lib/atom_constants.h.
_XS_RADII = (
    1.9,
    1.9,
    1.8,
    1.8,
    1.8,
    1.8,
    1.7,
    1.7,
    1.7,
    1.7,
    2.0,
    2.1,
    1.5,
    1.8,
    2.0,
    2.2,
    2.2,
    2.3,
    1.2,
    1.9,
    1.9,
    1.9,
    1.9,
    1.9,
    1.9,
    1.9,
    1.9,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
)
_XS_TYPE_SIZE = len(_XS_RADII)
_HYDROPHOBIC_TYPES = frozenset((0, 12, 13, 14, 15))
_DONOR_TYPES = frozenset((3, 5, 7, 9, 18))
_ACCEPTOR_TYPES = frozenset((4, 5, 8, 9))
_GLUE_TYPES = frozenset((21, 24, 27, 30))
_GLUED_CARBON_TYPES = frozenset((19, 20, 22, 23, 25, 26, 28, 29))
_GLUE_PARTNERS = {
    21: frozenset((19, 20)),
    24: frozenset((22, 23)),
    27: frozenset((25, 26)),
    30: frozenset((28, 29)),
}


def _rows(coordinates: Any, *, tangent: bool = False) -> tuple[list[list[float]], str]:
    """Validate a coordinate array and return rows plus its representation."""
    if isinstance(coordinates, (str, bytes)) or not isinstance(coordinates, Sequence):
        try:
            coordinates = coordinates.tolist()
        except AttributeError as exc:
            raise TypeError("coordinates must be a numeric (N, 3) sequence") from exc
    try:
        raw = list(coordinates)
    except TypeError as exc:
        raise TypeError("coordinates must be a numeric (N, 3) sequence") from exc
    if len(raw) < 2 and not tangent:
        raise ValueError("coordinates must contain at least two atoms")
    out: list[list[float]] = []
    for row in raw:
        if isinstance(row, (str, bytes)):
            raise TypeError("each coordinate must have exactly three real values")
        try:
            values = list(row)
        except TypeError as exc:
            raise TypeError("each coordinate must have exactly three real values") from exc
        if len(values) != 3:
            raise ValueError("coordinates must have shape (N, 3)")
        converted: list[float] = []
        for value in values:
            if isinstance(value, bool):
                raise TypeError("coordinates must contain real numbers")
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise TypeError("coordinates must contain real numbers") from exc
            if not math.isfinite(number):
                raise ValueError("coordinates must be finite")
            converted.append(number)
        out.append(converted)
    kind = "tuple" if isinstance(coordinates, tuple) else "list"
    return out, kind


def _atom_types(atom_types: Any, n_atoms: int) -> tuple[int, ...]:
    if atom_types is None:
        raise TypeError("atom_types is required: pass one XS_TYPE value per atom")
    if isinstance(atom_types, (str, bytes)):
        raise TypeError("atom_types must be a length-N integer sequence")
    try:
        values = list(atom_types)
    except TypeError as exc:
        raise TypeError("atom_types must be a length-N integer sequence") from exc
    if len(values) != n_atoms:
        raise ValueError("atom_types must have one X-Score type per atom")
    result: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("atom_types must contain integer XS_TYPE values")
        if value < 0 or value >= _XS_TYPE_SIZE:
            raise ValueError(f"atom type must be in [0, {_XS_TYPE_SIZE})")
        result.append(value)
    return tuple(result)


def _weights(weights: Any) -> tuple[float, ...]:
    if isinstance(weights, (str, bytes)):
        raise TypeError("weights must be a length-7 real sequence")
    try:
        values = list(weights)
    except TypeError as exc:
        raise TypeError("weights must be a length-7 real sequence") from exc
    if len(values) != 7:
        raise ValueError("weights must have length 7 (the SF_VINA coefficients)")
    result: list[float] = []
    for value in values:
        if isinstance(value, bool):
            raise TypeError("weights must contain real numbers")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError("weights must contain real numbers") from exc
        if not math.isfinite(number):
            raise ValueError("weights must be finite")
        result.append(number)
    return tuple(result)


def _pairs(pairs: Any, n_atoms: int) -> tuple[tuple[int, int], ...]:
    if pairs is None:
        return tuple((i, j) for i in range(n_atoms - 1) for j in range(i + 1, n_atoms))
    if isinstance(pairs, (str, bytes)):
        raise TypeError("pairs must be a sequence of (i, j) index pairs")
    try:
        raw_pairs = list(pairs)
    except TypeError as exc:
        raise TypeError("pairs must be a sequence of (i, j) index pairs") from exc
    result: list[tuple[int, int]] = []
    for pair in raw_pairs:
        try:
            values = list(pair)
        except TypeError as exc:
            raise TypeError("each pair must contain two atom indices") from exc
        if len(values) != 2 or any(isinstance(v, bool) or not isinstance(v, int) for v in values):
            raise TypeError("each pair must contain two atom indices")
        i, j = values
        if i < 0 or j < 0 or i >= n_atoms or j >= n_atoms or i == j:
            raise ValueError("pair indices must be distinct and in range")
        result.append((i, j))
    return tuple(result)


def _torsion_count(value: Any) -> float:
    if isinstance(value, bool):
        raise TypeError("torsion_count must be a finite non-negative real")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError("torsion_count must be a finite non-negative real") from exc
    if not math.isfinite(result) or result < 0:
        raise ValueError("torsion_count must be a finite non-negative real")
    return result


def _optimal_distance(type_i: int, type_j: int) -> float:
    if type_i in _GLUE_TYPES or type_j in _GLUE_TYPES:
        return 0.0
    return _XS_RADII[type_i] + _XS_RADII[type_j]


def _is_glued(type_i: int, type_j: int) -> bool:
    # is_glued in upstream/potentials.h pairs a closure glue type with its
    # matching C_H_CG* or C_P_CG* type.
    return type_j in _GLUE_PARTNERS.get(type_i, ()) or type_i in _GLUE_PARTNERS.get(type_j, ())


def _slope_step(bad: float, good: float, x: float) -> tuple[float, float]:
    if bad < good:
        if x <= bad or x >= good:
            return (0.0 if x <= bad else 1.0), 0.0
    else:
        if x >= bad or x <= good:
            return (0.0 if x >= bad else 1.0), 0.0
    return (x - bad) / (good - bad), 1.0 / (good - bad)


def _pair_terms(type_i: int, type_j: int, radius: float) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return six SF_VINA term values and d(term)/d(radius)."""
    # Potential cutoffs and active piecewise knots have no unique derivative.
    # Report them instead of silently selecting one-sided slopes.
    if math.isclose(radius, 8.0, rel_tol=0.0, abs_tol=1e-14):
        raise NonDifferentiablePoint("8 A SF_VINA potential cutoff is non-differentiable")
    optimal = _optimal_distance(type_i, type_j)
    delta = radius - optimal
    if radius < 8.0:
        if abs(delta) <= 1e-14:
            raise NonDifferentiablePoint("repulsion knot is non-differentiable")
        if type_i in _HYDROPHOBIC_TYPES and type_j in _HYDROPHOBIC_TYPES:
            if min(abs(delta - 0.5), abs(delta - 1.5)) <= 1e-14:
                raise NonDifferentiablePoint("hydrophobic slope-step knot is non-differentiable")
        donor_acceptor = (type_i in _DONOR_TYPES and type_j in _ACCEPTOR_TYPES) or (
            type_j in _DONOR_TYPES and type_i in _ACCEPTOR_TYPES
        )
        if donor_acceptor and min(abs(delta), abs(delta + 0.7)) <= 1e-14:
            raise NonDifferentiablePoint("hydrogen-bond slope-step knot is non-differentiable")
    if math.isclose(radius, 20.0, rel_tol=0.0, abs_tol=1e-14) and _is_glued(type_i, type_j):
        raise NonDifferentiablePoint("20 A macrocycle glue cutoff is non-differentiable")
    if radius >= 20.0:
        return (0.0,) * 6, (0.0,) * 6
    values = [0.0] * 6
    derivatives = [0.0] * 6
    if radius < 8.0:
        for index, (offset, width) in enumerate(((0.0, 0.5), (3.0, 2.0))):
            displacement = delta - offset
            value = math.exp(-((displacement / width) ** 2))
            values[index] = value
            derivatives[index] = -2.0 * displacement * value / (width * width)
        if delta <= 0.0:
            values[2] = delta * delta
            derivatives[2] = 2.0 * delta
        if type_i in _HYDROPHOBIC_TYPES and type_j in _HYDROPHOBIC_TYPES:
            values[3], derivatives[3] = _slope_step(1.5, 0.5, delta)
        donor_acceptor = (type_i in _DONOR_TYPES and type_j in _ACCEPTOR_TYPES) or (
            type_j in _DONOR_TYPES and type_i in _ACCEPTOR_TYPES
        )
        if donor_acceptor:
            values[4], derivatives[4] = _slope_step(0.0, -0.7, delta)
    if _is_glued(type_i, type_j):
        values[5] = radius
        derivatives[5] = 1.0
    return tuple(values), tuple(derivatives)


def _linearisation(
    rows: list[list[float]],
    atom_types: tuple[int, ...],
    pairs: tuple[tuple[int, int], ...],
    weights: tuple[float, ...],
    torsion_count: float,
) -> tuple[float, list[list[float]], tuple[float, ...]]:
    pair_energy = 0.0
    pair_gradient = [[0.0, 0.0, 0.0] for _ in rows]
    term_sums = [0.0] * 6
    for i, j in pairs:
        delta_xyz = [rows[i][k] - rows[j][k] for k in range(3)]
        radius_squared = sum(component * component for component in delta_xyz)
        if radius_squared == 0.0:
            raise NonDifferentiablePoint("coincident atom coordinates have no finite derivative")
        radius = math.sqrt(radius_squared)
        terms, derivatives = _pair_terms(atom_types[i], atom_types[j], radius)
        pair_energy += sum(weights[k] * terms[k] for k in range(6))
        term_sums = [term_sums[k] + terms[k] for k in range(6)]
        radial_derivative = sum(weights[k] * derivatives[k] for k in range(6))
        if radial_derivative:
            for k, component in enumerate(delta_xyz):
                force = radial_derivative * component / radius
                pair_gradient[i][k] += force
                pair_gradient[j][k] -= force
    denominator = 1.0 + weights[6] * torsion_count
    if denominator <= 0.0:
        raise ValueError("1 + weight_rot*torsion_count must be positive")
    score = pair_energy / denominator
    coordinate_gradient = [[component / denominator for component in row] for row in pair_gradient]
    weight_gradient = tuple(term / denominator for term in term_sums) + (
        -pair_energy * torsion_count / (denominator * denominator),
    )
    return score, coordinate_gradient, weight_gradient


def score_coordinates(
    coordinates: Any,
    atom_types: Any = None,
    *,
    pairs: Any = None,
    weights: Any = DEFAULT_VINA_WEIGHTS,
    torsion_count: Any = 0.0,
) -> float:
    """Return restricted SF_VINA pair energy for fixed molecular state.

    ``atom_types``, ``pairs`` and ``torsion_count`` are fixed state inputs;
    ``coordinates`` and ``weights`` are the active differentiable inputs.
    """
    rows, _ = _rows(coordinates)
    types = _atom_types(atom_types, len(rows))
    pair_indices = _pairs(pairs, len(rows))
    coefficient_values = _weights(weights)
    torsions = _torsion_count(torsion_count)
    return _linearisation(rows, types, pair_indices, coefficient_values, torsions)[0]


score = score_coordinates
energy = score_coordinates


def _restore_gradient(original: Any, gradient: list[list[float]]) -> Any:
    if isinstance(original, tuple):
        return tuple(tuple(row) for row in gradient)
    try:
        import numpy as np

        if hasattr(original, "shape"):
            return np.asarray(gradient, dtype=float).reshape(original.shape)
    except ImportError:
        pass
    return gradient


def _restore_vector(original: Any, values: tuple[float, ...]) -> Any:
    if isinstance(original, tuple):
        return tuple(values)
    try:
        import numpy as np

        if hasattr(original, "shape"):
            return np.asarray(values, dtype=float).reshape(original.shape)
    except ImportError:
        pass
    return list(values)


def _active_tangent(tangents: dict[str, Any], name: str) -> Any:
    value = tangents.get(name, ZERO)
    if value is not ZERO:
        return value
    return ZERO


@rules.jvp_for(score_coordinates)
def _score_coordinates_jvp(
    tangents: dict[str, Any],
    coordinates: Any,
    atom_types: Any = None,
    *,
    pairs: Any = None,
    weights: Any = DEFAULT_VINA_WEIGHTS,
    torsion_count: Any = 0.0,
) -> tuple[float, Any]:
    rows, _ = _rows(coordinates)
    types = _atom_types(atom_types, len(rows))
    pair_indices = _pairs(pairs, len(rows))
    coefficient_values = _weights(weights)
    torsions = _torsion_count(torsion_count)
    value = score_coordinates(
        coordinates,
        atom_types,
        pairs=pairs,
        weights=weights,
        torsion_count=torsion_count,
    )
    _, coordinate_gradient, weight_gradient = _linearisation(
        rows, types, pair_indices, coefficient_values, torsions
    )
    unsupported = set(tangents) - {"coordinates", "weights"}
    if unsupported:
        raise UnsupportedWrt(score_coordinates, unsupported, supported={"coordinates", "weights"})
    coordinate_tangent = _active_tangent(tangents, "coordinates")
    weight_tangent = _active_tangent(tangents, "weights")
    if coordinate_tangent is ZERO and weight_tangent is ZERO:
        return value, ZERO
    directional = 0.0
    if coordinate_tangent is not ZERO:
        tangent_rows, _ = _rows(coordinate_tangent, tangent=True)
        if len(tangent_rows) != len(rows):
            raise ValueError("coordinates tangent must have shape (N, 3)")
        directional += sum(
            coordinate_gradient[i][k] * tangent_rows[i][k]
            for i in range(len(rows))
            for k in range(3)
        )
    if weight_tangent is not ZERO:
        tangent_weights = _weights(weight_tangent)
        directional += sum(weight_gradient[k] * tangent_weights[k] for k in range(7))
    return value, directional


@rules.vjp_for(score_coordinates)
def _score_coordinates_vjp(
    wrt: tuple[str, ...],
    coordinates: Any,
    atom_types: Any = None,
    *,
    pairs: Any = None,
    weights: Any = DEFAULT_VINA_WEIGHTS,
    torsion_count: Any = 0.0,
) -> tuple[float, Any]:
    supported = {"coordinates", "weights"}
    unsupported = set(wrt) - supported
    if unsupported:
        raise UnsupportedWrt(score_coordinates, unsupported, supported=supported)
    rows, _ = _rows(coordinates)
    types = _atom_types(atom_types, len(rows))
    pair_indices = _pairs(pairs, len(rows))
    coefficient_values = _weights(weights)
    torsions = _torsion_count(torsion_count)
    value = score_coordinates(
        coordinates,
        atom_types,
        pairs=pairs,
        weights=weights,
        torsion_count=torsion_count,
    )
    _, coordinate_gradient, weight_gradient = _linearisation(
        rows, types, pair_indices, coefficient_values, torsions
    )

    def pullback(cotangent: Any) -> dict[str, Any]:
        factor = float(cotangent)
        result: dict[str, Any] = {}
        if "coordinates" in wrt:
            result["coordinates"] = _restore_gradient(
                coordinates,
                [[factor * component for component in row] for row in coordinate_gradient],
            )
        if "weights" in wrt:
            result["weights"] = _restore_vector(
                weights, tuple(factor * component for component in weight_gradient)
            )
        return result

    return value, pullback
