"""Perform crest analysis on ligand."""

import logging
import pathlib
from collections import Counter

import cgexplore as cgx
import matplotlib.pyplot as plt
import numpy as np
import stk
import stko

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def plot_xy(
    xproperty: str,
    ensemble: dict[str, dict],  # type: ignore[type-arg]
    min_energy: float,
    figure_dir: pathlib.Path,
    ligand_name: str,
) -> None:
    """Make an xy plot of properties."""
    fig, ax = plt.subplots(figsize=(8, 5))

    if xproperty in ("binder_angles",):
        ax.scatter(
            [ensemble[i][xproperty][0] for i in ensemble],
            [(ensemble[i]["energy"] - min_energy) * 2625.5 for i in ensemble],
            edgecolor="k",
            s=80,
        )
        ax.scatter(
            [ensemble[i][xproperty][1] for i in ensemble],
            [(ensemble[i]["energy"] - min_energy) * 2625.5 for i in ensemble],
            edgecolor="k",
            marker="D",
            s=80,
        )
    elif xproperty in ("torsion_state",):
        xs = [Counter(ensemble[i][xproperty]) for i in ensemble]

        xs = [i.get("b", 0) for i in xs]  # type: ignore[misc]

        ax.scatter(
            xs,  # type: ignore[arg-type]
            [(ensemble[i]["energy"] - min_energy) * 2625.5 for i in ensemble],
            edgecolor="k",
            marker="D",
            s=80,
        )
    else:
        ax.scatter(
            [ensemble[i][xproperty] for i in ensemble],
            [(ensemble[i]["energy"] - min_energy) * 2625.5 for i in ensemble],
            edgecolor="k",
            s=80,
        )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel(xproperty, fontsize=16)
    ax.set_ylabel("relative energy [kJ/mol]", fontsize=16)
    if xproperty == "binder_adjacent_torsion":
        ax.set_xlim(-180, 180)

    if xproperty == "binder_angles":
        ax.set_xlim(0, 180)

    if xproperty == "binder_binder_angle":
        ax.set_xlim(0, 180)

    ax.set_ylim(0, 20)

    fig.tight_layout()
    fig.savefig(
        figure_dir / f"xy_{xproperty}_{ligand_name}.png",
        dpi=360,
        bbox_inches="tight",
    )
    plt.close()


def main() -> None:
    """Run script."""
    wd = pathlib.Path("/home/atarzia/workingspace/starships/")
    ligand_dir = wd / "ligands"
    ligand_dir.mkdir(exist_ok=True)
    figure_dir = wd / "figures" / "crest"
    figure_dir.mkdir(exist_ok=True)
    calculation_dir = wd / "calculations"
    calculation_dir.mkdir(exist_ok=True)
    crest_path = pathlib.Path("/home/atarzia/software/crest_301/crest")
    xtb_path = pathlib.Path("/home/atarzia/miniforge3/envs/tscram/bin/xtb")

    ligands = {
        "st5": {"input": ligand_dir / "st5_manual.mol"},
        "la": {
            "smiles": "C1=C(C2C=CC3C4C=CC(C5=CC=CN=C5)=CC=4C(=O)C=3C=2)C=NC=C1"
        },
        "las": {
            "smiles": (
                "C1=CC=C(C=C1)N2C3=C(C=CC(=C3)C4=CN=CC=C4)C5=C2C=C(C=C5)C6=CN=CC=C6"
            ),
        },
        "c1": {
            "smiles": (
                "O=C(O[C@H]1CO[C@H]2[C@H](CO[C@@H]12)OC(=O)C3=CC=NC=C3)C4=C"
                "C=NC=C4"
            )
        },
    }

    for ligand, ligand_dict in ligands.items():
        logging.info("doing %s", ligand)
        if "smiles" in ligand_dict:  # type: ignore[operator]
            molecule = stk.BuildingBlock(ligand_dict["smiles"])  # type: ignore[index]
        elif "input" in ligand_dict:  # type: ignore[operator]
            molecule = stk.BuildingBlock.init_from_file(ligand_dict["input"])  # type: ignore[index]
        else:
            raise NotImplementedError

        molecule.write(ligand_dir / f"{ligand}_unopt.mol")

        ensemble = cgx.atomistic.run_conformer_analysis(
            ligand_name=ligand,
            molecule=molecule,
            ligand_dir=ligand_dir,
            calculation_dir=calculation_dir,
            functional_group_factories=(
                stko.functional_groups.ThreeSiteFactory("[#6]~[#7X2]~[#6]"),
            ),
            crest_path=crest_path,
            xtb_path=xtb_path,
        )

        min_energy = min([ensemble[i]["energy"] for i in ensemble])

        # Plot.
        fig, ax = plt.subplots(figsize=(8, 5))
        xwidth = 0.5
        relative_energies_kjmol = [
            (ensemble[i]["energy"] - min_energy) * 2625.5 for i in ensemble
        ]
        xmin = min(relative_energies_kjmol)
        xmax = 20
        xbins = np.arange(xmin - xwidth, xmax + xwidth, xwidth)
        ax.hist(
            x=relative_energies_kjmol,
            bins=list(xbins),
            density=False,
            histtype="stepfilled",
            stacked=True,
            linewidth=1.0,
            edgecolor="k",
        )
        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.set_xlabel("relative energy [kJ/mol]", fontsize=16)
        ax.set_ylabel("count", fontsize=16)
        fig.tight_layout()
        fig.savefig(
            figure_dir / f"dist_energy_{ligand}.png",
            dpi=360,
            bbox_inches="tight",
        )
        plt.close()

        for x_property in (
            "binder_angles",
            "binder_binder_angle",
            "binder_distance",
            "binder_adjacent_torsion",
            "adjacent_distance",
        ):
            plot_xy(
                xproperty=x_property,
                ensemble=ensemble,
                min_energy=min_energy,
                figure_dir=figure_dir,
                ligand_name=ligand,
            )

        _ = cgx.atomistic.get_ditopic_aligned_bb(
            path=wd / "ligands" / f"{ligand}_prep.mol",
            optl_path=wd / "ligands" / f"{ligand}_optl.mol",
        )


if __name__ == "__main__":
    main()
