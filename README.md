# vina-ad

`vina-ad` is a separately installable sidecar for AutoDock Vina. It exposes an
explicit ChainRules-compatible API for a restricted, atom-typed `SF_VINA`
coordinate pair kernel: `vina_ad.score_coordinates` (aliases: `score`,
`energy`) and the wrappers `jvp`, `vjp`, `grad`, and `value_and_grad`.

```python
import vina_ad
coordinates = [[0., 0., 0.], [3., 0., 0.], [0., 4., 0.]]
atom_types = [0, 0, 0]  # XS_TYPE_C_H values from upstream atom_constants.h
value, tangent = vina_ad.jvp(
    vina_ad.score_coordinates, coordinates,
    atom_types,
    tangents={"coordinates": [[1., 0., 0.], [0., 0., 0.], [0., 0., 0.]]},
)
value, gradients = vina_ad.value_and_grad(
    vina_ad.score_coordinates, coordinates, atom_types, wrt="coordinates"
)
```

The kernel maps Vina's two Gaussians, repulsion, hydrophobic, hydrogen-bond,
macrocycle glue, seven public weights, 8/20 A cutoffs, and torsion correction
from the immutable upstream source. Atom types, fixed interacting pairs, and
torsion count are state inputs; only coordinates and weights are active. It is
usable without the compiled upstream `vina_wrapper` and does not claim to
differentiate the complete C++ docking/search engine. See `SPEC.md`,
`api_inventory.json`, and `vina_ad/requirements.md` for provenance, scope, and
the complete support table. `python -m vina_ad.workflow` runs the sourced
workflow, including public `value_and_grad`/`jvp` calls and quantitative
primal/derivative metrics, when a real binding and PDBQT files are available.
In an installed environment pass the source inputs explicitly:

```bash
python -m vina_ad.workflow \
  --receptor /path/to/1iep_receptor.pdbqt \
  --ligand /path/to/1iep_ligand.pdbqt
```

If the real binding or inputs are unavailable it reports the capability as
deferred; the labelled `run_demo` is only a toy diagnostic.
