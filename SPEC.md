# AutoDock-Vina AD sidecar specification (round 2 scope)

> **Round-2 scope decision (formed before implementation).** The old generic
> three-feature replay described below is superseded by the restricted
> SF_VINA specification in the final section of this file.  The implementation
> and its tests are committed separately after this scope/inventory commit.

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

## Round-2 restricted SF_VINA scope

The only implemented sidecar callable is a restricted, pair-list replay of the
official `SF_VINA` scoring formula. It requires Cartesian coordinates,
X-Score atom types, a fixed interacting-pair topology, and a torsion count.
Atom types, pair topology, and torsion count are fixed state; only coordinates
and the seven public Vina weights are differentiable. A complete `vina.Vina`
docking workflow remains deferred because this snapshot's Python binding
imports the absent compiled `vina.vina_wrapper` extension.

`vina_ad.score_coordinates(coordinates, atom_types, *, pairs=None,
weights=DEFAULT_VINA_WEIGHTS, torsion_count=0.0)` returns pair energy in
kcal/mol. `pairs=None` means every `i < j`; otherwise each `(i, j)` is a fixed
pair. Atom types are integer `XS_TYPE_*` values from
`upstream/src/lib/atom_constants.h` (0 through 31).

For each pair with `r = ||x_i-x_j|| < 20`, `d = r -
optimal_distance(xs_i,xs_j)`, and `g(o,w)=exp(-((r-optimal_distance-o)/w)**2)`,
the six terms are exactly those in `upstream/src/lib/potentials.h`:

1. `vina_gaussian(0, 0.5, 8)` -> `g(0, 0.5)`;
2. `vina_gaussian(3, 2, 8)` -> `g(3, 2)`;
3. `vina_repulsion(0, 8)` -> `d*d` when `d <= 0`, otherwise zero;
4. `vina_hydrophobic(0.5, 1.5, 8)` -> `slope_step(1.5, 0.5, d)` for
   hydrophobic X-Score pairs;
5. `vina_non_dir_h_bond(-0.7, 0, 8)` -> `slope_step(0, -0.7, d)` for donor /
   acceptor pairs;
6. `linearattraction(20)` -> `r` only for macrocycle glue pairs.

The first six weights are coefficients of these terms. The seventh is the
public `weight_rot` from `Vina::set_vina_weights`; the Vina `num_tors_div`
correction is `E/(1 + weight_rot*torsion_count)`. This mapping is sourced to
`upstream/src/lib/scoring_function.h:48-59`,
`upstream/src/lib/potentials.h:134-210,495-514`, and
`upstream/src/lib/conf_independent.cpp:146-149`.

The registered JVP/VJP/gradient interfaces are real-linear in coordinates and
weights. `r == 0` and piecewise knots raise `NonDifferentiablePoint`;
malformed values, unknown tangent names, and unsupported active inputs fail
with contextual errors. Rules call the public primal and share one analytic
pair linearisation.

Round-2 evidence compares exact source-level `ScoringFunction::eval` terms and
also runs a real installed `vina.Vina` binding on a sourced one-atom
receptor/ligand PDBQT pair. The binding's rounded `Vina.score()[0]` is compared
within a documented 0.05 kcal/mol grid-interpolation bound. The official
`upstream/example/python_scripting/first_example.py` is recorded as deferred
when run directly from this snapshot because `vina.vina_wrapper` is absent;
the former hard-coded `run_demo` is retained only as a labelled toy diagnostic,
never as real-workflow evidence.
