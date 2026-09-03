# vina-ad

`vina-ad` is a separately installable sidecar for AutoDock Vina. It exposes an
explicit ChainRules-compatible API for a continuous coordinate scoring kernel:
`vina_ad.score_coordinates` (aliases: `score`, `energy`) and the wrappers
`jvp`, `vjp`, `grad`, and `value_and_grad`.

```python
import vina_ad
coordinates = [[0., 0., 0.], [3., 0., 0.], [0., 4., 0.]]
value, tangent = vina_ad.jvp(
    vina_ad.score_coordinates, coordinates,
    tangents={"coordinates": [[1., 0., 0.], [0., 0., 0.], [0., 0., 0.]]},
)
value, gradients = vina_ad.value_and_grad(
    vina_ad.score_coordinates, coordinates, wrt="coordinates"
)
```

The kernel is a documented Python replay of three smooth pair-distance
features. It is usable without the compiled upstream `vina_wrapper`; it does
not claim to differentiate the complete C++ docking/search engine. See
`SPEC.md`, `api_inventory.json`, and `vina_ad/requirements.md` for provenance,
scope, and the complete support table. Run the representative workflow with
`python -m vina_ad.workflow`.
