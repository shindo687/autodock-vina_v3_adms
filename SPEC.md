# AutoDock-Vina AD sidecar specification (round 2)

This scope was committed before implementation in commit `0923ce7`; the
implementation and tests follow in the round-2 fix commit. The reviewed
implementation was `f00535b` (review round 2).

## Upstream and boundary

- Official source: `https://github.com/ccsb-scripps/AutoDock-Vina.git`, commit
  `3c65c0b3e6c2c1d183f6a175ecb65e3c5ba91645`, copied verbatim under
  `upstream/` by import commit `ac64216`.
- The snapshot's Python binding imports the compiled `vina.vina_wrapper`,
  which is absent from the snapshot. Full `vina.Vina` docking is therefore
  deferred. A labelled toy diagnostic is never used as workflow evidence.
- The sidecar implements one faithful restricted subset: a fixed pair-list
  replay of official `SF_VINA`. It requires Cartesian coordinates, X-Score
  atom types, fixed pair topology, and a torsion count. Atom types, pair
  topology, and torsion count are fixed state; coordinates and seven public
  Vina weights are differentiable.

## Restricted SF_VINA callable

`vina_ad.score_coordinates(coordinates, atom_types, *, pairs=None,
weights=DEFAULT_VINA_WEIGHTS, torsion_count=0.0)` returns pair energy in
kcal/mol. `pairs=None` means every `i < j`; otherwise every `(i, j)` is a
fixed pair. Atom types are integer `XS_TYPE_*` values 0 through 31 from
`upstream/src/lib/atom_constants.h`.

For every pair with `r = ||x_i-x_j|| < 20`, let
`d = r - optimal_distance(xs_i,xs_j)` and
`g(o,w)=exp(-((d-o)/w)**2)`. The six terms are exactly the upstream classes:

1. `vina_gaussian(0, 0.5, 8)` -> `g(0, 0.5)`;
2. `vina_gaussian(3, 2, 8)` -> `g(3, 2)`;
3. `vina_repulsion(0, 8)` -> `d*d` when `d <= 0`, else zero;
4. `vina_hydrophobic(0.5, 1.5, 8)` -> `slope_step(1.5, 0.5, d)` for
   hydrophobic X-Score type pairs;
5. `vina_non_dir_h_bond(-0.7, 0, 8)` -> `slope_step(0, -0.7, d)` for donor /
   acceptor pairs;
6. `linearattraction(20)` -> `r` only for matching macrocycle glue pairs.

The first six public weights multiply these terms. The seventh is the public
`weight_rot` from `Vina::set_vina_weights`; after pair sum `E`, Vina's
`num_tors_div` correction is `E/(1 + weight_rot*torsion_count)`. The mapping is
sourced to `upstream/src/lib/scoring_function.h:48-59`,
`upstream/src/lib/potentials.h:134-210,495-514`, and
`upstream/src/lib/conf_independent.cpp:146-149`.

JVP/VJP/gradient rules are real-linear in coordinates and weights. Coincident
radii, the 8/20 A cutoffs, and piecewise knots are reported as
`NonDifferentiablePoint`. Invalid values and unsupported active inputs fail
with contextual errors. Rules call the public primal and share one analytic
pair linearisation.

## Oracle and workflow

Tests compare independent source-formula transcription to `ScoringFunction`
values (absolute error <= `1e-12`) and run a real installed `vina.Vina`
Python binding on a sourced one-atom receptor/ligand PDBQT pair. The binding
rounds to three decimals and interpolates maps; the restricted pair replay is
required to agree within `0.05` kcal/mol. The official
`upstream/example/python_scripting/first_example.py` is run and its missing
`vina.vina_wrapper` failure is recorded as deferred. `run_official_workflow`
reports that status for the multi-atom source files rather than claiming full
grid-score parity.

## Complete API inventory and decisions

All 19 public `vina.Vina` methods are recorded in `api_inventory.json`. Every
entry has primal semantics, differentiable and fixed inputs, derivative
interface, mathematical convention, reusable computation/dependency chain,
oracle, and explicit decision evidence. Constructor/configuration, file I/O,
map generation, pose selection, optimization, and stochastic docking remain
deferred or not suitable for AD.

## Acceptance thresholds and remaining scope

- Source-term oracle: absolute error <= `1e-12`.
- Real binding smoke oracle: absolute deviation <= `0.05` kcal/mol.
- Analytic derivatives: finite-difference error <= `2e-5` away from knots;
  JVP/VJP duality <= `1e-10`.
- Fresh no-dependency installs must preserve ChainRules 0.1.0 tangent/wrt
  validation and contextual `UnsupportedWrt` attributes.

The complete C++ grid/search engine, grid interpolation derivatives, topology
construction, and all deferred/not-suitable methods are outside this scope.
