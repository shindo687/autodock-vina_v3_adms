# Support status (vina-ad 0.1.0, round-2 scope)

| API family | Status | Evidence / reason |
| --- | --- | --- |
| `score_coordinates` (`score`, `energy` aliases) | implemented restricted subset | `core.py`; independent SF_VINA source-term oracle, real `vina.Vina` binding smoke comparison, and sourced cross-pair workflow |
| JVP for `coordinates`, `weights` | implemented | `core.py::_score_coordinates_jvp`; finite-difference oracle, weight analytic oracle, JVP/VJP duality, zero-direction, and active-pruning assertions in `tests/test_score.py` |
| VJP for `coordinates`, `weights` | implemented | `core.py::_score_coordinates_vjp`; coordinate/weight finite-difference oracles, JVP/VJP duality, pullback reuse, zero-cotangent, and active-pruning assertions in `tests/test_score.py` |
| `grad`, `value_and_grad` | implemented | delegated to ChainRules 0.1.0 (or the conformance fallback); both coordinate and weight gradients are compared with independent finite differences in `tests/test_score.py` |
| `vina.Vina.__init__`, configuration and text methods | not suitable | state, text, parsing, or file I/O; no continuous map |
| map, pose and energy I/O methods | not suitable/deferred | file serialization or compiled binding state |
| `vina.Vina.score` | deferred | upstream implementation is in unavailable `vina_wrapper`; restricted pair replay requires atom types/pairs and is explicitly separate |
| `vina.Vina.optimize`, `dock`, `randomize` | not suitable/deferred | iterative/stochastic/discrete search; no stable local derivative |

Unsupported upstream APIs are not registered. Calling ChainRules on an
unregistered callable raises `RuleNotFound`; unknown tangent/wrt names are
rejected before rule dispatch with contextual errors (including the
dependency-free fallback), and coincident/piecewise knots raise
`NonDifferentiablePoint` only when coordinates are active. Weights-only rules
remain defined at coordinate singularities and the zero-direction/cotangent
protocol is tested. The old coordinate-only three-feature toy is no longer
part of the public contract. `vina_ad.workflow` runs the sourced official
multi-atom example through real Vina and public `value_and_grad`/`jvp`, reports
quantitative pair, primal, derivative, finite-difference, and duality metrics,
and lists full grid/search derivatives as remaining coverage; it reports
`deferred` only when the real binding or input files are unavailable. Its
labelled toy diagnostic is not real-workflow evidence.
