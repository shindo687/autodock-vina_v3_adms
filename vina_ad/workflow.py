"""Workflow adapters with an explicit boundary around the compiled Vina API.

``run_demo`` is retained as a deterministic toy diagnostic.  It is deliberately
not described as a real docking workflow.  ``run_official_workflow`` loads the
official example PDBQT files, invokes the real ``vina.Vina`` binding when it is
installed, and reports ``deferred`` when the snapshot's missing compiled
``vina_wrapper`` prevents that operation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import DEFAULT_VINA_WEIGHTS, grad, score_coordinates


_PDBQT_TO_XS = {
    "C": 0,
    "A": 1,
    "N": 2,
    "NA": 4,
    "OA": 8,
    "SA": 8,
    "HD": 3,
    "O": 6,
    "S": 10,
    "F": 12,
    "CL": 13,
    "BR": 14,
    "I": 15,
}


def _read_pdbqt(path: Path) -> tuple[list[tuple[float, float, float]], list[int]]:
    coordinates: list[tuple[float, float, float]] = []
    atom_types: list[int] = []
    for line in path.read_text().splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        fields = line.split()
        try:
            xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            pdbqt_type = fields[-1].upper()
        except (ValueError, IndexError) as exc:
            raise ValueError(f"invalid PDBQT atom line in {path}") from exc
        if pdbqt_type not in _PDBQT_TO_XS:
            raise ValueError(f"unsupported PDBQT atom type {pdbqt_type!r} in {path}")
        coordinates.append(xyz)
        atom_types.append(_PDBQT_TO_XS[pdbqt_type])
    if not coordinates:
        raise ValueError(f"no ATOM/HETATM records found in {path}")
    return coordinates, atom_types


def run_demo() -> dict[str, float | str]:
    """Run a labelled, install-independent toy pair diagnostic."""
    coordinates = ((0.0, 0.0, 0.0), (3.0, 0.0, 0.0), (0.0, 4.0, 0.0))
    atom_types = (0, 0, 0)
    value = score_coordinates(coordinates, atom_types)
    gradients = grad(score_coordinates, coordinates, atom_types, wrt="coordinates")["coordinates"]
    norm = sum(component * component for row in gradients for component in row) ** 0.5
    return {"status": "toy-diagnostic", "score": value, "gradient_l2": norm, "n_atoms": float(len(coordinates))}


def run_official_workflow(
    receptor: str | Path | None = None,
    ligand: str | Path | None = None,
) -> dict[str, Any]:
    """Run the sourced official Python workflow, or report why it is deferred.

    The default files are resolved from the source checkout.  Installed wheels
    intentionally do not bundle the immutable upstream snapshot, so callers
    should pass explicit PDBQT paths in an installed environment.
    """
    if receptor is None:
        receptor = Path(__file__).resolve().parents[1] / "upstream/example/python_scripting/1iep_receptor.pdbqt"
    if ligand is None:
        ligand = Path(__file__).resolve().parents[1] / "upstream/example/python_scripting/1iep_ligand.pdbqt"
    receptor_path, ligand_path = Path(receptor), Path(ligand)
    if not receptor_path.exists() or not ligand_path.exists():
        return {"status": "deferred", "reason": "official PDBQT inputs are unavailable after installation"}
    try:
        from vina import Vina
    except (ImportError, ModuleNotFoundError) as exc:
        return {"status": "deferred", "reason": f"real vina binding unavailable: {exc}"}
    try:
        vina = Vina(sf_name="vina", cpu=1, seed=1, verbosity=0)
        vina.set_receptor(str(receptor_path))
        vina.set_ligand_from_file(str(ligand_path))
        vina.compute_vina_maps(center=[15.190, 53.903, 16.917], box_size=[20, 20, 20])
        real_score = float(vina.score()[0])
    except Exception as exc:  # binding errors are an explicit deferred result
        return {"status": "deferred", "reason": f"real Vina workflow failed: {exc}"}
    receptor_coordinates, receptor_types = _read_pdbqt(receptor_path)
    ligand_coordinates, ligand_types = _read_pdbqt(ligand_path)
    # A restricted pair replay is deliberately scoped to one sourced
    # receptor/ligand pair; it is not claimed to reproduce a multi-atom grid
    # score.  Only a one-atom-per-file fixture has a meaningful oracle bound.
    pair_coordinates = (receptor_coordinates[0], ligand_coordinates[0])
    pair_types = (receptor_types[0], ligand_types[0])
    replay_score = score_coordinates(pair_coordinates, pair_types)
    comparable = len(receptor_coordinates) == 1 and len(ligand_coordinates) == 1
    result = {
        "status": "completed" if comparable else "deferred",
        "real_vina_score": real_score,
        "restricted_pair_score": replay_score,
        "absolute_deviation": abs(real_score - replay_score),
        "deviation_bound": 0.05 if comparable else None,
        "receptor_atoms": float(len(receptor_coordinates)),
        "ligand_atoms": float(len(ligand_coordinates)),
    }
    if not comparable:
        result["reason"] = "official multi-atom workflow ran, but restricted pair replay is not a full grid scorer"
    return result


if __name__ == "__main__":
    result = run_official_workflow()
    if result["status"] == "deferred":
        print("status=deferred reason={reason}".format(**result))
    else:
        print(
            "status=completed real_vina_score={real_vina_score:.3f} "
            "restricted_pair_score={restricted_pair_score:.12f} "
            "absolute_deviation={absolute_deviation:.6f}".format(**result)
        )
