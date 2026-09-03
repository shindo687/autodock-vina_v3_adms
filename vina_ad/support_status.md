# Support status (vina-ad 0.1.0)

| API family | Status | Evidence / reason |
| --- | --- | --- |
| `score_coordinates` (`score`, `energy` aliases) | implemented | `core.py`; analytic oracle, finite-difference and duality tests |
| JVP for `coordinates`, `weights` | implemented | `core.py::_score_coordinates_jvp`; `tests/test_score.py` |
| VJP for `coordinates`, `weights` | implemented | `core.py::_score_coordinates_vjp`; pullback reuse and duality tests |
| `grad`, `value_and_grad` | implemented | delegated to ChainRules 0.1.0; scalar output checked by protocol |
| `vina.Vina.__init__`, configuration and text methods | not suitable | state, text, parsing, or file I/O; no continuous map |
| map, pose and energy I/O methods | not suitable/deferred | file serialization or compiled binding state |
| `vina.Vina.score` | deferred | upstream implementation is in unavailable `vina_wrapper`; replay is explicitly separate |
| `vina.Vina.optimize`, `dock`, `randomize` | not suitable/deferred | iterative/stochastic/discrete search; no stable local derivative |

Unsupported upstream APIs are not registered. Calling ChainRules on an
unregistered callable raises `RuleNotFound`; unknown `wrt` names are rejected by
ChainRules, and coincident coordinates raise `NonDifferentiablePoint`.
