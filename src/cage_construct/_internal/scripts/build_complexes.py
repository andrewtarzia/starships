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
    complex_path = working_path / "complexes"
    complex_path.mkdir(parents=True, exist_ok=True)

    # Build metal atom.
    metal = stk.BuildingBlock(
        smiles="[Fe+2]",
        functional_groups=(
            stk.SingleAtom(stk.Fe(0, charge=2)) for i in range(6)
        ),
        position_matrix=[[0, 0, 0]],
    )
    bidentate = stk.BuildingBlock(
        smiles="C1=CC=NC(=C1)C=NBr",
        functional_groups=[
            stk.SmartsFunctionalGroupFactory(
                smarts="[#6]~[#7X2]~[#35]",
                bonders=(1,),
                deleters=(),
            ),
            stk.SmartsFunctionalGroupFactory(
                smarts="[#6]~[#7X2]~[#6]",
                bonders=(1,),
                deleters=(),
            ),
        ],
    )
    bidentate = cage_construct.utilities.ensure_nccnbr_torsion(bidentate)
    bidentate.write(complex_path / "bidentate.mol")

    # Build only one sterochemistry for now.
    lcomplex = stk.ConstructedMolecule(
        topology_graph=stk.metal_complex.OctahedralLambda(
            metals=metal,
            ligands=bidentate,
            optimizer=stk.MCHammer(),
        )
    )
    lcomplex.write(complex_path / "lcomplex.mol")


if __name__ == "__main__":
    main()
