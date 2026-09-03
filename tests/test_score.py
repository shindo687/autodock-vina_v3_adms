import math

import pytest

import vina_ad


POSE = ((0.0, 0.0, 0.0), (3.0, 0.0, 0.0), (0.0, 4.0, 0.0))
WEIGHTS = (1.0, -0.5, 0.2)


def analytic_score_and_grad(points, weights):
    value = 0.0
    grad = [[0.0, 0.0, 0.0] for _ in points]
    features = [0.0, 0.0, 0.0]
    for i in range(len(points) - 1):
        for j in range(i + 1, len(points)):
            d = [points[i][k] - points[j][k] for k in range(3)]
            r = math.sqrt(sum(x * x for x in d))
            f = [math.exp(-((r - 3) / 0.5) ** 2), math.exp(-((r - 5) / 1.5) ** 2), 1 / r]
            features = [features[k] + f[k] for k in range(3)]
            value += sum(weights[k] * f[k] for k in range(3))
            ddr = weights[0] * f[0] * (-2 * (r - 3) / 0.25)
            ddr += weights[1] * f[1] * (-2 * (r - 5) / 2.25) - weights[2] / r**2
            for k in range(3):
                grad[i][k] += ddr * d[k] / r
                grad[j][k] -= ddr * d[k] / r
    return value, grad


def test_primal_parity_and_aliases():
    expected, _ = analytic_score_and_grad(POSE, WEIGHTS)
    assert vina_ad.score_coordinates(POSE, WEIGHTS) == pytest.approx(expected, abs=1e-14)
    assert vina_ad.score is vina_ad.score_coordinates
    assert vina_ad.energy is vina_ad.score_coordinates


def test_analytic_oracle_and_finite_difference():
    expected, expected_grad = analytic_score_and_grad(POSE, WEIGHTS)
    value, gradients = vina_ad.value_and_grad(vina_ad.score_coordinates, POSE, WEIGHTS, wrt="coordinates")
    assert value == pytest.approx(expected, abs=1e-14)
    for row, exp_row in zip(gradients["coordinates"], expected_grad):
        assert row == pytest.approx(exp_row, abs=1e-12)
    h = 1e-5
    for i in range(3):
        for k in range(3):
            plus = [list(row) for row in POSE]
            minus = [list(row) for row in POSE]
            plus[i][k] += h
            minus[i][k] -= h
            numerical = (vina_ad.score_coordinates(plus, WEIGHTS) - vina_ad.score_coordinates(minus, WEIGHTS)) / (2 * h)
            assert numerical == pytest.approx(expected_grad[i][k], abs=2e-5)


def test_jvp_matches_directional_oracle():
    tangent = ((0.2, -0.1, 0.3), (-0.4, 0.2, 0.1), (0.1, 0.5, -0.2))
    _, expected_grad = analytic_score_and_grad(POSE, WEIGHTS)
    expected = sum(expected_grad[i][k] * tangent[i][k] for i in range(3) for k in range(3))
    value, tangent_out = vina_ad.jvp(vina_ad.score_coordinates, POSE, WEIGHTS, tangents={"coordinates": tangent})
    assert value == vina_ad.score_coordinates(POSE, WEIGHTS)
    assert tangent_out == pytest.approx(expected, abs=1e-12)


def test_jvp_weight_direction():
    tangent = (0.25, -0.5, 0.75)
    expected = sum(analytic_score_and_grad(POSE, WEIGHTS)[0] * 0 for _ in ())
    features = [0.0, 0.0, 0.0]
    for i in range(3):
        for j in range(i + 1, 3):
            r = math.dist(POSE[i], POSE[j])
            features[0] += math.exp(-((r - 3) / 0.5) ** 2)
            features[1] += math.exp(-((r - 5) / 1.5) ** 2)
            features[2] += 1 / r
    expected = sum(a * b for a, b in zip(features, tangent))
    assert vina_ad.jvp(vina_ad.score_coordinates, POSE, WEIGHTS, tangents={"weights": tangent})[1] == pytest.approx(expected)


def test_vjp_duality_and_reuse():
    tangent = ((0.2, -0.1, 0.3), (-0.4, 0.2, 0.1), (0.1, 0.5, -0.2))
    _, jvp_out = vina_ad.jvp(vina_ad.score_coordinates, POSE, WEIGHTS, tangents={"coordinates": tangent})
    value, pullback = vina_ad.vjp(vina_ad.score_coordinates, POSE, WEIGHTS, wrt=("coordinates", "weights"))
    cotangent = 1.7
    result1 = pullback(cotangent)
    result2 = pullback(cotangent)
    assert value == vina_ad.score_coordinates(POSE, WEIGHTS)
    lhs = cotangent * jvp_out
    rhs = sum(tangent[i][k] * result1["coordinates"][i][k] for i in range(3) for k in range(3))
    assert lhs == pytest.approx(rhs, abs=1e-12)
    assert result1 == result2
    assert set(result1) == {"coordinates", "weights"}


def test_grad_and_value_and_grad_weights():
    expected, _ = analytic_score_and_grad(POSE, WEIGHTS)
    _, grad = vina_ad.value_and_grad(vina_ad.energy, POSE, WEIGHTS, wrt="weights")
    assert grad["weights"] == pytest.approx([sum(math.exp(-((math.dist(POSE[i], POSE[j]) - 3) / 0.5) ** 2) for i in range(3) for j in range(i + 1, 3)),
                                               sum(math.exp(-((math.dist(POSE[i], POSE[j]) - 5) / 1.5) ** 2) for i in range(3) for j in range(i + 1, 3)),
                                               sum(1 / math.dist(POSE[i], POSE[j]) for i in range(3) for j in range(i + 1, 3))])
    assert vina_ad.grad(vina_ad.score, POSE, WEIGHTS, wrt="weights") == grad


def test_zero_direction_short_circuit():
    value, tangent = vina_ad.jvp(vina_ad.score_coordinates, POSE, WEIGHTS, tangents={"coordinates": vina_ad.ZERO})
    assert value == vina_ad.score_coordinates(POSE, WEIGHTS)
    assert tangent is vina_ad.ZERO
    _, pullback = vina_ad.vjp(vina_ad.score_coordinates, POSE, WEIGHTS, wrt="coordinates")
    assert pullback(vina_ad.ZERO)["coordinates"] is vina_ad.ZERO


def test_errors_and_boundaries():
    with pytest.raises(ValueError, match="at least two"):
        vina_ad.score_coordinates(((0, 0, 0),))
    with pytest.raises(ValueError, match="shape"):
        vina_ad.score_coordinates(((0, 0), (1, 1, 1)))
    with pytest.raises(ValueError, match="finite"):
        vina_ad.score_coordinates(((0, 0, 0), (math.inf, 0, 0)))
    with pytest.raises(vina_ad.NonDifferentiablePoint):
        vina_ad.grad(vina_ad.score_coordinates, ((0, 0, 0), (0, 0, 0)), wrt="coordinates")
    # ChainRules rejects names absent from the callable signature before the
    # provider rule runs; this is the documented unsupported-input error.
    with pytest.raises(TypeError, match="Unknown wrt"):
        vina_ad.vjp(vina_ad.score_coordinates, POSE, WEIGHTS, wrt="other")
    with pytest.raises(ValueError, match="length 3"):
        vina_ad.score_coordinates(POSE, (1, 2))
