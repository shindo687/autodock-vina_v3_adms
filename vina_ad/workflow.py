"""A deterministic, install-independent representative scoring workflow."""

from . import grad, score_coordinates


def run_demo() -> dict[str, float]:
    receptor = ((0.0, 0.0, 0.0), (3.0, 0.0, 0.0))
    ligand = ((0.0, 4.0, 0.0),)
    coordinates = receptor + ligand
    value = score_coordinates(coordinates)
    gradients = grad(score_coordinates, coordinates, wrt="coordinates")["coordinates"]
    norm = sum(component * component for row in gradients for component in row) ** 0.5
    return {"score": value, "gradient_l2": norm, "n_atoms": float(len(coordinates))}


if __name__ == "__main__":
    result = run_demo()
    print("score={score:.12f} gradient_l2={gradient_l2:.12f} n_atoms={n_atoms:.0f}".format(**result))
