import argparse
import logging
import pathlib

import stk

import cage_construct


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--working_path",
        type=str,
        help=("Path to working directory to save outputs."),
    )

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    logging.info("saving to %s", args.working_path)
    working_path = pathlib.Path(args.working_path).resolve()
    ligand_path = working_path / "ligands"
    ligand_path.mkdir(parents=True, exist_ok=True)

    # For the POC.
    ditopic_bb1 = stk.BuildingBlock(
        smiles="BrC=NC1CCCCC1N=CBr",
        functional_groups=(stk.BromoFactory(),),
    )
    ditopic_bb1.write(ligand_path / "ditopic_bb1_unopt.mol")

    tritopic_bb2 = stk.BuildingBlock(
        smiles="C1=C(C=C(C=C1Br)Br)Br",
        functional_groups=(stk.BromoFactory(),),
    )
    tritopic_bb2.write(ligand_path / "tritopic_bb2_unopt.mol")

    # For the MOCs.
    ditopic_bb3 = stk.BuildingBlock(
        smiles="C1=CC(=CC=C1C2=CC=C(C=C2)Br)Br",
        functional_groups=(stk.BromoFactory(),),
    )
    ditopic_bb3.write(ligand_path / "ditopic_bb3_unopt.mol")

    # Get the lowest energy conformer for all and write energy to file.
    bb1_lowe_conf, bb1_min_energy = (
        cage_construct.utilities.get_lowest_energy_conformer(ditopic_bb1)
    )
    bb1_lowe_conf.write(ligand_path / "ditopic_bb1_lowe.mol")
    with (ligand_path / "ditopic_bb1_lowe.ey").open("w") as f:
        f.write(f"{bb1_min_energy}\n")

    bb2_lowe_conf, bb2_min_energy = (
        cage_construct.utilities.get_lowest_energy_conformer(tritopic_bb2)
    )
    bb2_lowe_conf.write(ligand_path / "tritopic_bb2_lowe.mol")
    with (ligand_path / "tritopic_bb2_lowe.ey").open("w") as f:
        f.write(f"{bb1_min_energy}\n")

    bb3_lowe_conf, bb3_min_energy = (
        cage_construct.utilities.get_lowest_energy_conformer(ditopic_bb3)
    )
    bb3_lowe_conf.write(ligand_path / "ditopic_bb3_lowe.mol")
    with (ligand_path / "ditopic_bb3_lowe.ey").open("w") as f:
        f.write(f"{bb1_min_energy}\n")

    # Here you would often have to get the prepared conformer, but in this
    # case they all work out.

    # For example for bent-ditopic ligands, you normally have to align their
    # building blocks using ``bbprep.DitopicFitter``.


if __name__ == "__main__":
    main()
