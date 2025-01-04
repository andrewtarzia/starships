import argparse
import logging
import pathlib

import chemiscope
import stk
import stko

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
    cage_path = working_path / "cages"
    calculations_path = cage_path / "calculations"

    # Load molecules.
    cages = {
        "POC": stk.BuildingBlock.init_from_file(
            path=cage_path / "POC_optc.mol"
        ),
        "MOC": stk.BuildingBlock.init_from_file(
            path=cage_path / "MOC_optc.mol"
        ),
    }

    shape_calc = stko.ShapeCalculator()
    structures = []
    properties = {
        "asphericity": [],
        "pore_diameter_opt / AA": [],
        "pore_volume_opt / AA^3": [],
        "maximum_diameter / AA": [],
    }
    for name, cage_structure in cages.items():
        logging.info("analysing %s", name)
        structures.append(cage_structure)

        asphericity = shape_calc.get_results(cage_structure).get_asphericity()
        properties["asphericity"].append(asphericity)

        pywindow_data = cage_construct.utilities.calculate_pywindow_data(
            name=name,
            calculation_path=calculations_path,
            molecule=cage_structure,
        )

        properties["pore_diameter_opt / AA"].append(
            pywindow_data["pore_diameter_opt"]
        )
        properties["pore_volume_opt / AA^3"].append(
            pywindow_data["pore_volume_opt"]
        )
        properties["maximum_diameter / AA"].append(
            pywindow_data["maximum_diameter"]
        )

    # A simple chemiscope interface, load into https://chemiscope.org/
    chemiscope.write_input(
        path=str(cage_path / "example.json.gz"),
        frames=structures,
        properties=properties,
        meta={"name": "Example database."},
        settings=chemiscope.quick_settings(
            x="pore_diameter_opt / AA",
            y="pore_volume_opt / AA^3",
            color="asphericity",
        ),
    )


if __name__ == "__main__":
    main()
