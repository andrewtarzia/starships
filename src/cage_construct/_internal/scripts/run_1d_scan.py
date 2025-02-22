"""Script to generate and optimise CG models."""

import argparse
import itertools as it
import logging
import pathlib

import cgexplore as cgx
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import stk
from openmm import OpenMMException
from rdkit import RDLogger

from .utilities import (
    abead_c,
    abead_d,
    binder_bead,
    cbead_c,
    cbead_d,
    eb_str,
    ebead_c,
    precursors_to_forcefield,
    tetra_bead,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
RDLogger.DisableLog("rdApp.*")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="store_true",
        help="set to iterate through structure functions",
    )
    return parser.parse_args()


def analyse_cage(
    database_path: pathlib.Path,
    name: str,
    forcefield: cgx.forcefields.ForceField,
    num_building_blocks: int,
) -> None:
    """Analyse a toy model cage."""
    database = cgx.utilities.AtomliteDatabase(database_path)
    properties = database.get_entry(key=name).properties

    database.add_properties(
        key=name,
        property_dict={
            "forcefield_dict": forcefield.get_forcefield_dictionary(),  # pyright: ignore[]
            "energy_per_bb": cgx.utilities.get_energy_per_bb(
                energy_decomposition=properties["energy_decomposition"],
                number_building_blocks=num_building_blocks,
            ),
        },
    )


def make_plot(
    database_path: pathlib.Path,
    figure_dir: pathlib.Path,
    filename: str,
    scans: dict[str, dict[str, list[float] | str]],
) -> None:
    """Visualise energies."""
    fig, axs = plt.subplots(ncols=2, sharey=True, figsize=(10, 5))
    cg_scale = 2
    vmin = 0
    vmax = 0.6
    scans_to_ys = {i: j for j, i in enumerate(scans.keys())}

    for scan, sdict in scans.items():
        y = scans_to_ys[scan]
        if scan == "aa":
            xoption = "a_c"
            scaler = cg_scale * 2
        else:
            xoption = "_".join(list(scan))
            scaler = cg_scale

        if len(xoption.split("_")) == 2:  # noqa: PLR2004
            ax = axs[0]
            tx = sdict["r"][0] / scaler
            ax.axvline(x=0, c="k", zorder=-2)
            relative = True

        else:
            ax = axs[1]
            tx = sdict["r"][0]
            ax.plot(
                (tx, tx),
                (y - 0.3, y + 0.3),
                c="k",
                zorder=2,
            )
            relative = False

        for entry in cgx.utilities.AtomliteDatabase(
            database_path
        ).get_entries():
            if scan != entry.key.split("_")[1]:
                continue

            x = float(entry.properties["forcefield_dict"]["v_dict"][xoption])
            if relative:
                x = x - tx

            c = float(entry.properties["energy_per_bb"])

            ax.scatter(
                x,
                y,
                c=c,
                vmin=vmin,
                vmax=vmax,
                alpha=1.0,
                edgecolor="k",
                s=50,
                marker="s",
                cmap="Blues_r",
            )

    axs[0].tick_params(axis="both", which="major", labelsize=16)
    axs[0].set_xlabel(r"value-target [$\mathrm{\AA}$]", fontsize=16)
    axs[0].set_yticks(list(scans_to_ys.values()))
    axs[0].set_yticklabels([f"${i}$" for i in scans_to_ys])
    axs[0].set_ylim(-0.5, 4.5)
    axs[1].tick_params(axis="both", which="major", labelsize=16)
    axs[1].set_xlabel("value [$^\\circ$]", fontsize=16)

    cbar_ax = fig.add_axes([1.01, 0.2, 0.02, 0.7])  # type: ignore[call-overload]
    cmap = mpl.cm.Blues_r  # type: ignore[attr-defined]
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cbar_ax,
        orientation="vertical",
    )
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label(eb_str(), fontsize=16)

    fig.tight_layout()
    fig.savefig(
        figure_dir / filename,
        dpi=360,
        bbox_inches="tight",
    )
    fig.savefig(
        figure_dir / filename.replace(".png", ".pdf"),
        dpi=360,
        bbox_inches="tight",
    )
    plt.close()


def main() -> None:  # noqa: PLR0915
    """Run script."""
    args = _parse_args()
    wd = pathlib.Path("/home/atarzia/workingspace/starships/")
    calculation_dir = wd / "oned_calculations"
    calculation_dir.mkdir(exist_ok=True)
    structure_dir = wd / "oned_structures"
    structure_dir.mkdir(exist_ok=True)
    ligand_dir = wd / "oned_ligands"
    ligand_dir.mkdir(exist_ok=True)
    data_dir = wd / "oned_data"
    data_dir.mkdir(exist_ok=True)
    figure_dir = wd / "figures"
    figure_dir.mkdir(exist_ok=True)
    database_path = data_dir / "oned.db"

    converging = cgx.molecular.SixBead(
        bead=cbead_c,
        abead1=abead_c,
        abead2=ebead_c,
    )
    converging_name = "la"
    diverging = cgx.molecular.TwoC1Arm(bead=cbead_d, abead1=abead_d)
    diverging_name = "st5"
    tetra = cgx.molecular.FourC1Arm(bead=tetra_bead, abead1=binder_bead)

    scans = {
        "aa": {
            "r": [
                3.9,
                4.0,
                4.2,
                4.4,
                4.6,
                4.8,
                5.0,
                5.2,
                5.4,
                5.6,
                5.8,
                6.0,
                3.8,
                3.6,
                3.4,
                3.2,
                3.0,
                2.8,
                2.6,
                2.4,
                2.2,
                2.0,
            ],
            "c": "st5",
        },
        "dd": {
            "r": [
                7.0,
                7.2,
                7.4,
                7.6,
                7.8,
                8.0,
                8.2,
                8.4,
                8.6,
                8.8,
                9.0,
                6.8,
                6.6,
                6.4,
                6.2,
                6.0,
                5.8,
                5.6,
                5.4,
                5.2,
                5.0,
            ],
            "c": "la",
        },
        "de": {
            "r": [
                1.5,
                1.6,
                1.7,
                1.8,
                1.9,
                2.0,
                2.1,
                2.2,
                2.3,
                2.4,
                2.5,
                1.4,
                1.3,
                1.2,
                1.1,
                1.0,
                0.9,
                0.8,
                0.7,
                0.6,
                0.5,
            ],
            "c": "la",
        },
        "bac": {
            "r": [
                120,
                125,
                130,
                135,
                140,
                145,
                150,
                155,
                160,
                165,
                170,
                175,
                115,
                110,
                105,
                100,
                95,
                90,
            ],
            "c": "st5",
        },
        "dde": {
            "r": [
                170,
                175,
                165,
                160,
                155,
                150,
                145,
                140,
                135,
                130,
                125,
                120,
                115,
                110,
                105,
                100,
                95,
                90,
            ],
            "c": "la",
        },
    }
    if args.run:
        for cname, pair_range_dict in scans.items():
            # Rewrite each time.
            ligand_measures = {
                "la": {"dd": 7.0, "de": 1.5, "dde": 170, "eg": 1.4, "gb": 1.4},
                "st5": {"ba": 2.8, "aa": 3.9, "bac": 120, "bacab": 180},
            }

            for i, xp in enumerate(pair_range_dict["r"]):
                ligand_measures[pair_range_dict["c"]][cname] = xp

                forcefield = precursors_to_forcefield(
                    pair="1dscan",
                    diverging=diverging,
                    converging=converging,
                    conv_meas=ligand_measures[converging_name],
                    dive_meas=ligand_measures[diverging_name],
                )

                converging_bb = cgx.utilities.optimise_ligand(
                    molecule=converging.get_building_block(),
                    name=f"{converging.get_name()}_f{forcefield.get_identifier()}",
                    output_dir=calculation_dir,
                    forcefield=forcefield,
                    platform=None,
                )
                converging_bb = converging_bb.clone()

                tetra_bb = cgx.utilities.optimise_ligand(
                    molecule=tetra.get_building_block(),
                    name=f"{tetra.get_name()}_f{forcefield.get_identifier()}",
                    output_dir=calculation_dir,
                    forcefield=forcefield,
                    platform=None,
                )
                tetra_bb = tetra_bb.clone()

                diverging_bb = cgx.utilities.optimise_ligand(
                    molecule=diverging.get_building_block(),
                    name=f"{diverging.get_name()}_f{forcefield.get_identifier()}",
                    output_dir=calculation_dir,
                    forcefield=forcefield,
                    platform=None,
                )
                diverging_bb = diverging_bb.clone()

                name = f"onedscan_{cname}_{i}"
                logging.info("building %s", name)

                cage = stk.ConstructedMolecule(
                    stk.cage.M3L6(
                        building_blocks={
                            tetra_bb: (0, 1, 2),
                            converging_bb: (3, 4, 5, 6),
                            diverging_bb: (7, 8),
                        },
                        vertex_positions=None,
                    )
                )
                cage.write(str(structure_dir / f"{name}_unopt.mol"))

                si = name.split("_")[2]
                potential_names = []
                for cstr, neg in it.product(scans, range(10)):
                    potential_names.append(f"onedscan_{cstr}_{int(si) - neg}")

                try:
                    conformer = cgx.scram.optimise_cage(
                        molecule=cage,
                        name=name,
                        output_dir=calculation_dir,
                        forcefield=forcefield,
                        platform=None,
                        database_path=database_path,
                        potential_names=potential_names,
                    )
                    if conformer is not None:
                        conformer.molecule.with_centroid(
                            np.array((0, 0, 0))
                        ).write(str(structure_dir / f"{name}_optc.mol"))

                    analyse_cage(
                        database_path=database_path,
                        name=name,
                        forcefield=forcefield,
                        num_building_blocks=9,
                    )

                except OpenMMException:
                    pass

            # Rescan over the surface for improved energies.
            for i, xp in enumerate(pair_range_dict["r"]):
                ligand_measures[pair_range_dict["c"]][cname] = xp

                forcefield = precursors_to_forcefield(
                    pair="1dscan",
                    diverging=diverging,
                    converging=converging,
                    conv_meas=ligand_measures[converging_name],
                    dive_meas=ligand_measures[diverging_name],
                )

                name = f"onedscan_{cname}_{i}"
                logging.info("rescanning %s", name)

                current_cage = stk.BuildingBlock.init_from_file(
                    structure_dir / f"{name}_optc.mol"
                )

                potential_names = []
                x_indices_of_interest = [
                    pair_range_dict["r"].index(x)
                    for _, x in sorted(
                        zip(
                            [abs(i - xp) for i in pair_range_dict["r"]],
                            pair_range_dict["r"],
                            strict=False,
                        )
                    )
                ][:6]
                for cstr, xidx in it.product(scans, x_indices_of_interest):
                    potential_names.append(f"onedscan_{cstr}_{xidx}")

                conformer = cgx.scram.optimise_from_files(
                    molecule=current_cage,
                    name=name,
                    output_dir=calculation_dir,
                    forcefield=forcefield,
                    platform=None,
                    database_path=database_path,
                    potential_names=potential_names,
                )

                conformer.molecule.with_centroid(np.array((0, 0, 0))).write(
                    str(structure_dir / f"{name}_optc.mol")
                )

                analyse_cage(
                    database_path=database_path,
                    name=name,
                    forcefield=forcefield,
                    num_building_blocks=9,
                )

    make_plot(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="1dscan_1.png",
        scans=scans,
    )


if __name__ == "__main__":
    main()
