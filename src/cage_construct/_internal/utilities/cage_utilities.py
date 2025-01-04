"""Utilities for cage processing."""

import logging
import pathlib

import stk
import stko

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def cage_optimisation(  # noqa: PLR0913
    name: str,
    calculation_path: pathlib.Path,
    molecule: stk.Molecule | stk.BuildingBlock | stk.ConstructedMolecule,
    charge: int,
    metal: bool,  # noqa: FBT001
    xtb_path: pathlib.Path,
) -> stk.Molecule:
    """Run optimisation workflow.

    Currently this is an xtb-based workflow. However, ``stko`` allows using
    ``GULP``, ``xtb``, ``OpenMM/OpenFF`` for these processes and you should
    implement what you need.

    """
    rdkit_output = calculation_path / f"{name}_rdkit.mol"
    xtbffopt_output = calculation_path / f"{name}_xtbff.mol"

    if not xtb_path.exists():
        msg = f"xtb is not installed here: {xtb_path}"
        raise ValueError(msg)

    if metal:
        if not rdkit_output.exists():
            output_dir = calculation_path / f"{name}_rdkitopt"
            logging.info("    rdkit optimisation of %s", name)
            rdkit_opt = stko.MetalOptimizer(max_iterations=2000)
            molecule = rdkit_opt.optimize(mol=molecule)
            molecule.write(rdkit_output)

        else:
            logging.info("    loading %s", rdkit_output)
            molecule = molecule.with_structure_from_file(str(rdkit_output))

    if not xtbffopt_output.exists():
        output_dir = calculation_path / f"{name}_xtbffopt"
        logging.info("    xtbff optimisation of %s", name)
        xtb_opt = stko.XTBFF(
            xtb_path=xtb_path,
            output_dir=output_dir,
            num_cores=6,
            charge=charge,
            opt_level="normal",
            unlimited_memory=True,
        )
        molecule = xtb_opt.optimize(mol=molecule)
        molecule.write(xtbffopt_output)

    else:
        logging.info("    loading %s", xtbffopt_output)
        molecule = molecule.with_structure_from_file(str(xtbffopt_output))

    return molecule.with_structure_from_file(str(xtbffopt_output))
