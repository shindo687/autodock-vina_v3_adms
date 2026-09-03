import json
import pathlib
import subprocess
import sys

import vina_ad
from vina_ad.workflow import run_demo


def test_representative_workflow_quantitative():
    result = run_demo()
    assert result["n_atoms"] == 3.0
    assert result["score"] == vina_ad.score_coordinates(((0., 0., 0.), (3., 0., 0.), (0., 4., 0.)))
    assert result["gradient_l2"] > 0


def test_inventory_is_complete_and_valid():
    data = json.loads(pathlib.Path("api_inventory.json").read_text())
    assert data["coverage_totals"]["upstream_public_entries"] == len(data["upstream_public_api"]) == 19
    assert {entry["status"] for entry in data["upstream_public_api"]} <= {"implemented", "deferred", "not_ad_suitable"}


def test_module_workflow_entrypoint():
    proc = subprocess.run([sys.executable, "-m", "vina_ad.workflow"], check=True, capture_output=True, text=True)
    assert "score=" in proc.stdout and "gradient_l2=" in proc.stdout
