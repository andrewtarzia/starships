"""Utilities for cage processing."""

import json
import logging
import pathlib

import pywindow as pw
import stk

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def calculate_pywindow_data(
    name: str,
    calculation_path: pathlib.Path,
    molecule: stk.Molecule | stk.BuildingBlock | stk.ConstructedMolecule,
) -> dict[str, float]:
    """Run pywindow."""
    json_file = calculation_path / f"{name}_pw.json"

    if json_file.exists():
        with json_file.open("r") as f:
            results = json.load(f)
    else:
        logging.info("running pywindow on %s", name)

        molsys = pw.MolecularSystem.load_rdkit_mol(molecule.to_rdkit_mol())
        mol = molsys.system_to_molecule()

        try:
            mol.calculate_pore_diameter_opt()
            mol.calculate_pore_volume_opt()
            mol.calculate_maximum_diameter()

            results = {
                "pore_diameter_opt": (
                    mol.properties["pore_diameter_opt"]["diameter"]
                ),
                "pore_volume_opt": mol.properties["pore_volume_opt"],
                "maximum_diameter": mol.properties["maximum_diameter"][
                    "diameter"
                ],
            }

        except ValueError:
            results = {
                "pore_diameter_opt": 0,
                "pore_volume_opt": 0,
                "maximum_diameter": 0,
            }

        with json_file.open("w") as f:
            json.dump(results, f)
    return results
