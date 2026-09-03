# Round-1 fix receipt

Reviewed commit: `ac64216` (review round 1). Fix commit is the commit containing
this receipt.

Environment: Linux x86_64, CPython 3.12.13, pytest 8.4.2, ChainRules 0.1.0.

Commands and observed totals:

| command | result | exit |
| --- | --- | ---: |
| `pytest -q` | 11 passed, 0 failed | 0 |
| `python -m pytest -q` | 11 passed, 0 failed | 0 |
| `pytest -q -p no:cacheprovider tests` | 11 passed, 0 failed | 0 |
| `python -m pip install --no-deps --target /tmp/vina_ad_review_install2 .` | wheel built and installed | 0 |
| `PYTHONPATH=/tmp/vina_ad_review_install2 python -c 'import vina_ad; ...'` | version `0.1.0`, score `0.9821600089636336` | 0 |
| fresh venv `pip install --no-deps --no-build-isolation .` | installed `vina-ad-0.1.0` | 0 |
| fresh venv `python -m vina_ad.workflow` | score `0.269885566173`, gradient L2 `0.679126205494`, 3 atoms | 0 |

Analytic-oracle maximum absolute error: below `1e-12`; central finite-difference
checks use `h=1e-5` and tolerance `2e-5`; JVP/VJP duality tolerance is `1e-12`.
No production finite-difference fallback is present. Complete C++ scoring,
search, map generation and the 19 upstream binding methods remain deferred or
not suitable as listed in `../support_status.md` and `api_inventory.json`.
