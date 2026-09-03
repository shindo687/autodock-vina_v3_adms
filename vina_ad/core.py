"""Continuous coordinate scoring replay and its explicit ChainRules rules.

The function in this module is intentionally small and dependency-free.  It is
the primal callable used by both rules; the derivative implementation only
evaluates its analytic linearisation and never finite-differences production
inputs.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from .protocol import NonDifferentiablePoint, UnsupportedWrt, ZERO, rules

DEFAULT_WEIGHTS = (1.0, -0.5, 0.2)


def _rows(coordinates: Any) -> tuple[list[list[float]], str]:
    """Validate coordinates and return rows plus a representation hint."""
    if isinstance(coordinates, (str, bytes)) or not isinstance(coordinates, Sequence):
        # numpy arrays are not Sequence on all supported Python versions.
        try:
            coordinates = coordinates.tolist()
        except AttributeError as exc:
            raise TypeError("coordinates must be a numeric (N, 3) sequence") from exc
    try:
        raw = list(coordinates)
    except TypeError as exc:
        raise TypeError("coordinates must be a numeric (N, 3) sequence") from exc
    if len(raw) < 2:
        raise ValueError("coordinates must contain at least two points")
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


def _weights(weights: Any) -> tuple[float, float, float]:
    if isinstance(weights, (str, bytes)):
        raise TypeError("weights must be a length-3 real sequence")
    try:
        values = list(weights)
    except TypeError as exc:
        raise TypeError("weights must be a length-3 real sequence") from exc
    if len(values) != 3:
        raise ValueError("weights must have length 3")
    result = []
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
    return result[0], result[1], result[2]


def _pair_linearisation(rows: list[list[float]], weights: tuple[float, float, float],
                        tangent: list[list[float]] | None = None) -> tuple[float, list[list[float]]]:
    """Return score and (optionally) coordinate directional derivative.

    The second return value is a coordinate-shaped gradient when ``tangent`` is
    None, and a one-row ``[[directional_value]]`` result otherwise.  Keeping
    this one shared kernel makes JVP and VJP use the same mathematical mapping.
    """
    score = 0.0
    gradient = [[0.0, 0.0, 0.0] for _ in rows]
    w0, w1, w2 = weights
    for i in range(len(rows) - 1):
        for j in range(i + 1, len(rows)):
            delta = [rows[i][k] - rows[j][k] for k in range(3)]
            r2 = sum(x * x for x in delta)
            if r2 == 0.0:
                raise NonDifferentiablePoint("coincident coordinates have no finite derivative")
            r = math.sqrt(r2)
            z0 = (r - 3.0) / 0.5
            z1 = (r - 5.0) / 1.5
            f0, f1, f2 = math.exp(-(z0 * z0)), math.exp(-(z1 * z1)), 1.0 / r
            score += w0 * f0 + w1 * f1 + w2 * f2
            d_dr = w0 * f0 * (-2.0 * (r - 3.0) / (0.5 * 0.5))
            d_dr += w1 * f1 * (-2.0 * (r - 5.0) / (1.5 * 1.5)) - w2 / (r * r)
            unit = [x / r for x in delta]
            for k in range(3):
                component = d_dr * unit[k]
                gradient[i][k] += component
                gradient[j][k] -= component
    if tangent is None:
        return score, gradient
    return score, [[sum(gradient[i][k] * tangent[i][k] for k in range(3)) for i in range(len(rows))]]


def score_coordinates(coordinates: Any, weights: Any = DEFAULT_WEIGHTS) -> float:
    """Return the continuous pair-distance score for an ``(N, 3)`` pose."""
    rows, _ = _rows(coordinates)
    return _pair_linearisation(rows, _weights(weights))[0]


score = score_coordinates
energy = score_coordinates


def _restore_gradient(original: Any, gradient: list[list[float]]) -> Any:
    """Preserve common list/tuple/NumPy-array conventions for cotangents."""
    if isinstance(original, tuple):
        return tuple(tuple(row) for row in gradient)
    try:
        import numpy as np  # optional convenience; not a runtime requirement
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


@rules.jvp_for(score_coordinates)
def _score_coordinates_jvp(tangents: dict[str, Any], coordinates: Any,
                           weights: Any = DEFAULT_WEIGHTS) -> tuple[float, Any]:
    rows, _ = _rows(coordinates)
    weight_values = _weights(weights)
    # Always obtain the primal through the public callable.  The linearisation
    # below supplies only derivative data; it is not an alternate primal API.
    value = score_coordinates(coordinates, weights)
    _, gradient = _pair_linearisation(rows, weight_values)
    active_coordinates = tangents.get("coordinates", ZERO)
    active_weights = tangents.get("weights", ZERO)
    if active_coordinates is ZERO and active_weights is ZERO:
        return value, ZERO
    directional = 0.0
    if active_coordinates is not ZERO:
        tangent_rows, _ = _rows(active_coordinates)
        if len(tangent_rows) != len(rows):
            raise ValueError("coordinates tangent must have shape (N, 3)")
        directional += sum(gradient[i][k] * tangent_rows[i][k]
                           for i in range(len(rows)) for k in range(3))
    if active_weights is not ZERO:
        dw = _weights(active_weights)
        # Feature sums are exactly the partial derivatives wrt weights.
        feature_sums = [0.0, 0.0, 0.0]
        for i in range(len(rows) - 1):
            for j in range(i + 1, len(rows)):
                d = [rows[i][k] - rows[j][k] for k in range(3)]
                r = math.sqrt(sum(x * x for x in d))
                feature_sums[0] += math.exp(-((r - 3.0) / 0.5) ** 2)
                feature_sums[1] += math.exp(-((r - 5.0) / 1.5) ** 2)
                feature_sums[2] += 1.0 / r
        directional += sum(feature_sums[k] * dw[k] for k in range(3))
    return value, directional


@rules.vjp_for(score_coordinates)
def _score_coordinates_vjp(wrt: tuple[str, ...], coordinates: Any,
                           weights: Any = DEFAULT_WEIGHTS) -> tuple[float, Any]:
    supported = {"coordinates", "weights"}
    unsupported = set(wrt) - supported
    if unsupported:
        raise UnsupportedWrt(score_coordinates, unsupported, supported=supported)
    rows, _ = _rows(coordinates)
    weight_values = _weights(weights)
    # Preserve the original callable as the sole primal source.
    value = score_coordinates(coordinates, weights)
    _, coordinate_gradient = _pair_linearisation(rows, weight_values)
    feature_sums = [0.0, 0.0, 0.0]
    for i in range(len(rows) - 1):
        for j in range(i + 1, len(rows)):
            r = math.sqrt(sum((rows[i][k] - rows[j][k]) ** 2 for k in range(3)))
            feature_sums[0] += math.exp(-((r - 3.0) / 0.5) ** 2)
            feature_sums[1] += math.exp(-((r - 5.0) / 1.5) ** 2)
            feature_sums[2] += 1.0 / r

    def pullback(cotangent: Any) -> dict[str, Any]:
        result: dict[str, Any] = {}
        factor = float(cotangent)
        if "coordinates" in wrt:
            result["coordinates"] = _restore_gradient(coordinates,
                                                        [[factor * x for x in row]
                                                         for row in coordinate_gradient])
        if "weights" in wrt:
            result["weights"] = _restore_vector(weights, tuple(factor * x for x in feature_sums))
        return result
    return value, pullback
