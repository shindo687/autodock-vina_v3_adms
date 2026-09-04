import json
import math
import os
import pathlib
import subprocess
import sys

import pytest

import vina_ad
from vina_ad.workflow import run_demo, run_official_workflow


def test_representative_workflow_quantitative():
    result = run_demo()
    assert result["status"] == "toy-diagnostic"
    assert result["n_atoms"] == 3.0
    assert result["score"] == vina_ad.score_coordinates(((0., 0., 0.), (3., 0., 0.), (0., 4., 0.)), (0, 0, 0))
    assert result["gradient_l2"] > 0


def test_official_workflow_runs_public_ad_on_sourced_example():
    root = pathlib.Path(__file__).resolve().parents[1]
    receptor = root / "upstream/example/python_scripting/1iep_receptor.pdbqt"
    ligand = root / "upstream/example/python_scripting/1iep_ligand.pdbqt"
    result = run_official_workflow(receptor, ligand)
    if result["status"] == "deferred":
        pytest.skip(result["reason"])
    assert result["status"] == "completed"
    assert math.isfinite(result["real_vina_optimized_score"])
    assert result["vina_optimize_max_steps"] == 1.0
    assert result["receptor_atoms"] > 1000
    assert result["ligand_atoms"] > 10
    assert result["interaction_pairs"] == result["receptor_atoms"] * result["ligand_atoms"]
    assert result["workflow_scale"]["interaction_pairs"] == result["interaction_pairs"]
    assert result["iterations"]["ad_value_and_grad"] == 1.0
    assert result["ad_value_and_grad_evaluations"] == 1.0
    assert result["ad_jvp_evaluations"] == 1.0
    assert result["ad_coordinate_gradient_l2"] > 0.0
    assert result["ad_weight_gradient_l2"] > 0.0
    assert result["derivative_abs_error"] <= 2e-5
    assert result["duality_abs_error"] <= 1e-10
    assert result["jvp_primal"] == pytest.approx(result["ad_primal"], abs=1e-12)
    assert "grid interpolation" in result["remaining_coverage"]


def test_official_workflow_default_is_quantified_or_deferred():
    result = run_official_workflow()
    assert result["status"] in {"completed", "deferred"}
    if result["status"] == "completed":
        assert result["ad_value_and_grad_evaluations"] == 1.0
        assert result["derivative_abs_error"] <= 2e-5


def test_fresh_target_install_runs_sourced_ad_workflow(tmp_path):
    """The representative workflow must use the installed sidecar package."""
    pytest.importorskip("vina")
    root = pathlib.Path(__file__).resolve().parents[1]
    target = tmp_path / "target"
    install = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--no-deps",
            "--no-build-isolation",
            "--target",
            str(target),
            str(root),
        ],
        capture_output=True,
        text=True,
    )
    assert install.returncode == 0, install.stderr
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(target)
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "vina_ad.workflow",
            "--receptor",
            str(root / "upstream/example/python_scripting/1iep_receptor.pdbqt"),
            "--ligand",
            str(root / "upstream/example/python_scripting/1iep_ligand.pdbqt"),
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.startswith("status=completed ")
    result = json.loads(proc.stdout.split(" ", 1)[1])
    assert result["ad_value_and_grad_evaluations"] == 1.0
    assert result["ad_jvp_evaluations"] == 1.0
    assert result["derivative_abs_error"] <= 2e-5
    assert result["receptor_atoms"] > 1000


def test_inventory_is_complete_and_valid():
    data = json.loads(pathlib.Path("api_inventory.json").read_text())
    assert data["coverage_totals"]["upstream_public_entries"] == len(data["upstream_public_api"]) == 19
    assert {entry["status"] for entry in data["upstream_public_api"]} <= {"implemented", "deferred", "not_ad_suitable"}


def test_module_workflow_entrypoint():
    proc = subprocess.run([sys.executable, "-m", "vina_ad.workflow"], check=True, capture_output=True, text=True)
    assert "status=" in proc.stdout
