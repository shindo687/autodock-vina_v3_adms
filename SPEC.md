# AutoDock-Vina AD sidecar specification (round 1 fix)

## Fixed inputs and provenance

- Upstream: official `https://github.com/ccsb-scripps/AutoDock-Vina`, commit
  `3c65c0b3e6c2c1d183f6a175ecb65e3c5ba91645`, imported verbatim under
  `upstream/` in commit `ac64216`.
- Sidecar: `vina-ad` 0.1.0, Python 3.10+, tested on CPython 3.12.13.
- AD protocol: ChainRules 0.1.0 (`chainrules` package); runtime has no JAX or
  PyTorch dependency. ChainRules is used when installed, while a small
  protocol-compatible fallback keeps installation self-contained when the
  internal ChainRules package is unavailable.
- Import time and removal of the cloned upstream `.git` directory are recorded
  in `vina_ad/requirements.md`.

## Supported public API

`vina_ad.score_coordinates(coordinates, weights=DEFAULT_WEIGHTS)` returns one
real scalar. `coordinates` is an `(N, 3)` finite real sequence (N >= 2), and
`weights` is a length-3 finite real sequence. For every `i < j`, let
`r = ||coordinates[i]-coordinates[j]||`,

```
f0(r) = exp(-((r - 3.0)/0.5)**2)
f1(r) = exp(-((r - 5.0)/1.5)**2)
f2(r) = 1/r
score = sum_ij (weights[0]*f0 + weights[1]*f1 + weights[2]*f2)
```

This is a smooth, continuous coordinate-scoring replay selected because the
upstream Python package exposes no callable scoring kernel without its compiled
`vina_wrapper`. The sidecar primal is the sole source of its returned value;
rules call it rather than reimplementing a second primal.

AD interfaces are available through `vina_ad.jvp`, `vina_ad.vjp`,
`vina_ad.grad`, and `vina_ad.value_and_grad`, which delegate to ChainRules. The
registered inputs are `coordinates` and `weights`. A JVP tangent is keyed by
those names. A VJP pullback returns exactly the requested keys and preserves
the list/tuple/NumPy-array shape of each input. `grad` and `value_and_grad`
require the scalar score output.

Mathematical convention is real-linear. At a coincident pair (`r == 0`) the
inverse-distance feature has no finite derivative and raises
`chainrules.NonDifferentiablePoint`; malformed, non-finite, or empty inputs
raise `ValueError`/`TypeError`. No finite-difference calculation is used in
production rules.

Aliases `vina_ad.score` and `vina_ad.energy` refer to the same registered
callable. `vina_ad.workflow.run_demo` is an install-independent workflow that
scores a small receptor/ligand coordinate set and reports a gradient norm.

## Complete API decisions

The 19 public entries in upstream `vina.Vina` are inventoried in
`api_inventory.json`. Constructor/configuration, file I/O, map loading,
pose formatting, randomization, optimization, and global docking are marked
`not_ad_suitable` or `deferred`: they are discrete/stateful operations, require
the missing compiled extension, or have no stable continuous input/output
semantics. No unsupported upstream entry is silently registered.

## Acceptance and error thresholds

Tests require primal parity between direct and AD calls at machine precision,
JVP/VJP duality at `1e-10` absolute/relative tolerance for the fixtures,
analytic-oracle derivative agreement at `1e-8`, and finite-difference agreement
within `2e-5` over three step sizes. The suite covers zero directions, invalid
shapes, coincident pairs, unsupported `wrt`, pullback reuse, installation, and
the workflow. Remaining coverage is the complete C++ scoring/search engine and
all 19 upstream binding methods listed as deferred/not suitable.
