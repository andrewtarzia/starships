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
    parser.add_argument(
        "--xtb_path",
        type=str,
        help=("Path to xtb software."),
    )

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    logging.info("saving to %s", args.working_path)
    working_path = pathlib.Path(args.working_path).resolve()
    ligand_path = working_path / "ligands"
    complex_path = working_path / "complexes"
    cage_path = working_path / "cages"
    cage_path.mkdir(parents=True, exist_ok=True)
    calculations_path = cage_path / "calculations"
    calculations_path.mkdir(parents=True, exist_ok=True)

    # Cage 1: CC3, POC.
    ditopic_bb1 = stk.BuildingBlock.init_from_file(
        path=ligand_path / "ditopic_bb1_lowe.mol",
        functional_groups=(stk.BromoFactory(),),
    )
    tritopic_bb = stk.BuildingBlock.init_from_file(
        path=ligand_path / "tritopic_bb2_lowe.mol",
        functional_groups=(stk.BromoFactory(),),
    )

    name = "POC"
    opt_file = cage_path / f"{name}_optc.mol"
    if not opt_file.exists():
        logging.info("building %s", name)
        cage = stk.ConstructedMolecule(
            stk.cage.FourPlusSix(
                building_blocks=(ditopic_bb1, tritopic_bb),
                # Use smaller target than default to go straight to xtb opt.
                # Normally, I would recommend using another
                # step with Gulp or OpenMM before xTB.
                optimizer=stk.MCHammer(target_bond_length=1.0),
            )
        )
        cage.write(cage_path / f"{name}_unopt.mol")
        # Always check this structure makes sense before moving on to a costly
        # optimisation!

        cage = cage_construct.utilities.cage_optimisation(
            name=name,
            calculation_path=calculations_path,
            molecule=cage,
            charge=0,
            xtb_path=pathlib.Path(args.xtb_path),
            metal=False,
        )
        cage.write(opt_file)

    # Cage 2: MOC.
    ditopic_bb3 = stk.BuildingBlock.init_from_file(
        path=ligand_path / "ditopic_bb3_lowe.mol",
        functional_groups=(stk.BromoFactory(),),
    )
    tritopic_complex = stk.BuildingBlock.init_from_file(
        path=complex_path / "lcomplex.mol",
        functional_groups=(stk.BromoFactory(),),
    )

    name = "MOC"
    opt_file = cage_path / f"{name}_optc.mol"
    if not opt_file.exists():
        logging.info("building %s", name)
        cage = stk.ConstructedMolecule(
            stk.cage.M4L6TetrahedronSpacer(
                building_blocks={
                    tritopic_complex: (0, 1, 2, 3),
                    ditopic_bb3: (4, 5, 6, 7, 8, 9),
                },
                # Use differnt target and weaker epsilon than default to go
                # straight to xtb opt. Normally, I would recommend using
                # another step with Gulp or OpenMM before xTB.
                optimizer=stk.MCHammer(
                    target_bond_length=1.5,
                    num_steps=2000,
                    nonbond_epsilon=5,
                ),
            )
        )
        cage.write(cage_path / f"{name}_unopt.mol")
        # Always check this structure makes sense before moving on to a costly
        # optimisation!

        cage = cage_construct.utilities.cage_optimisation(
            name=name,
            calculation_path=calculations_path,
            molecule=cage,
            charge=2 * 4,
            xtb_path=pathlib.Path(args.xtb_path),
            metal=True,
        )
        cage.write(opt_file)


if __name__ == "__main__":
    main()
