"""Script to generate and optimise CG models."""

import argparse
import logging
import pathlib

import cgexplore as cgx
import matplotlib as mpl
import matplotlib.pyplot as plt
import stko
from openmm import OpenMMException
from rdkit import RDLogger

from .utilities import (
    Scrambler,
    abead_c,
    abead_d,
    binder_bead,
    cbead_c,
    cbead_d,
    eb_str,
    ebead_c,
    isomer_energy,
    name_parser,
    precursors_to_forcefield,
    simple_beeswarm2,
    tetra_bead,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
RDLogger.DisableLog("rdApp.*")


def analyse_cage(
    database_path: pathlib.Path,
    name: str,
    forcefield: cgx.forcefields.ForceField,
    iterator: Scrambler,
    topology_code: cgx.scram.TopologyCode,
) -> None:
    """Analyse a toy model cage."""
    database = cgx.utilities.AtomliteDatabase(database_path)
    properties = database.get_entry(key=name).properties
    if "topology_code_vmap" not in properties:
        num_components = len(
            stko.Network.init_from_molecule(
                database.get_molecule(key=name)
            ).get_connected_components()
        )

        splits = name.split("_")
        if len(splits) == 4:  # noqa: PLR2004
            multiplier = name.split("_")[2]
            pairname = name.split("_")[0] + "_" + name.split("_")[1]
        elif len(splits) == 5:  # noqa: PLR2004
            multiplier = name.split("_")[3]
            pairname = (
                name.split("_")[0]
                + "_"
                + name.split("_")[1]
                + "_"
                + name.split("_")[2]
            )

        database.add_properties(
            key=name,
            property_dict={
                "forcefield_dict": forcefield.get_forcefield_dictionary(),
                "energy_per_bb": cgx.utilities.get_energy_per_bb(
                    energy_decomposition=properties["energy_decomposition"],
                    number_building_blocks=iterator.get_num_building_blocks(),
                ),
                "pair": pairname,
                "num_components": num_components,
                "multiplier": multiplier,
                "topology_code_vmap": tuple(
                    (int(i[0]), int(i[1])) for i in topology_code.vertex_map
                ),
            },
        )


def make_plot(
    pair: str,
    database_path: pathlib.Path,
    structure_dir: pathlib.Path,
    figure_dir: pathlib.Path,
    filename: str,
) -> None:
    """Visualise energies."""
    fig, ax = plt.subplots(figsize=(8, 5))
    energies: dict[str, list[tuple[float, str]]] = {}
    cmap = {
        "1": "tab:blue",
        "2": "tab:orange",
        "3": "tab:green",
        "4": "tab:red",
    }

    for entry in cgx.utilities.AtomliteDatabase(database_path).get_entries():
        if "pair" not in entry.properties:
            continue

        if pair != entry.properties["pair"]:
            continue

        multi = entry.properties["multiplier"]
        energy = entry.properties["energy_per_bb"]

        if multi not in energies:
            energies[multi] = []

        if entry.properties["num_components"] > 1:
            continue
        energies[multi].append((round(energy, 4), entry.key))

    with (figure_dir / f"min_{pair}.txt").open("w") as f:
        for multi, edata in energies.items():
            if len(edata) == 0:
                continue

            sorted_energies = sorted(edata, key=lambda p: p[0])
            min_energy = sorted_energies[0]

            ax.plot(
                [i[0] for i in edata],
                marker="o",
                c=cmap[multi],
                markersize=4,
                label=f"{multi}: {round(min_energy[0], 3)} @ {min_energy[1]}",
            )

            opt_file = structure_dir / f"{min_energy[1]}_optc.mol"
            f.write(f"{opt_file} ")

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_ylabel(eb_str(), fontsize=16)
    ax.set_yscale("log")
    ax.set_ylim(0.01, 1000)
    ax.axhline(y=isomer_energy(), c="k", ls="--")
    ax.legend(ncols=1, fontsize=16)
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


def make_summary_plot(
    database_path: pathlib.Path,
    figure_dir: pathlib.Path,
    filename: str,
) -> None:
    """Visualise energies."""
    fig, ax = plt.subplots(figsize=(5, 5))
    energies: dict[tuple[str, str], list[tuple[float, str]]] = {}

    xs = ["1", "2", "3", "4"]
    ys = ["la_st5", "la_st52", "la_c1"]
    ys.reverse()

    for entry in cgx.utilities.AtomliteDatabase(database_path).get_entries():
        if "pair" not in entry.properties:
            continue

        multi = entry.properties["multiplier"]
        pair = entry.properties["pair"]
        if "_11" in pair:
            continue
        if pair not in ys:
            continue

        vstr = entry.key.split("_")[-1]
        energy = entry.properties["energy_per_bb"]

        if (pair, multi) not in energies:
            energies[(pair, multi)] = []

        if entry.properties["num_components"] > 1:
            continue
        energies[(pair, multi)].append((round(energy, 4), vstr))

    vmin = 0
    vmax = 1
    for (pair, multi), edata in energies.items():
        sorted_energies = sorted(edata, key=lambda p: p[0])
        min_energy = sorted_energies[0]

        x = xs.index(multi)
        y = ys.index(pair)

        ax.scatter(
            x,
            y,
            c=min_energy[0],
            vmin=vmin,
            vmax=vmax,
            alpha=1.0,
            edgecolor="k",
            s=200,
            marker="s",
            cmap="Blues_r",
        )
        ax.text(
            x=x,
            y=y,
            s=min_energy[1],
            horizontalalignment="center",
            verticalalignment="center_baseline",
            color="w" if min_energy[0] < 0.5 else "k",  # noqa: PLR2004
            fontsize=12,
        )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel("multiplier", fontsize=16)
    ax.set_xticks(list(range(len(xs))))
    ax.set_xticklabels(xs)
    ax.set_yticks(list(range(len(ys))))
    ax.set_yticklabels(ys)

    cbar_ax = fig.add_axes([1.01, 0.2, 0.02, 0.7])  # type: ignore[call-overload]
    cmap = mpl.cm.Blues_r  # type: ignore[attr-defined]
    norm = mpl.colors.Normalize(vmin=0, vmax=vmax)
    cbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cbar_ax,
        orientation="vertical",
    )
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label(f"4:2:3 {eb_str()}", fontsize=16)

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


def make_summary_plot2(  # noqa: C901
    database_path: pathlib.Path,
    figure_dir: pathlib.Path,
    filename: str,
) -> None:
    """Visualise energies."""
    st5 = name_parser("st5")
    c1 = name_parser("c1")
    if "main" in filename:
        fig, ax = plt.subplots(figsize=(8, 3))
        systems = {
            ("la_st5", "1"): {"name": "s-1", "data": []},
            ("la_st5", "2"): {"name": "s-2", "data": []},
            ("la_st5", "4"): {"name": "s-4", "data": []},
            ("la_st52", "1"): {"name": "l-1", "data": []},
            ("la_st52", "2"): {"name": "l-2", "data": []},
            ("la_st52", "4"): {"name": "l-4", "data": []},
            ("la_st5_11", "1"): {"name": "s,1:1-1", "data": []},
            ("la_st5_11", "2"): {"name": "s,1:1-2", "data": []},
            ("la_st5_11", "3"): {"name": "s,1:1-3", "data": []},
        }
        ax.axvline(x=2 + 0.5, c="gray")
        ax.axvline(x=5 + 0.5, c="gray")

    elif "si" in filename:
        fig, ax = plt.subplots(figsize=(8, 5))
        systems = {
            ("la_st5", "1"): {"name": f"{st5}-s-1", "data": []},
            ("la_st5", "2"): {"name": f"{st5}-s-2", "data": []},
            ("la_st5", "4"): {"name": f"{st5}-s-4", "data": []},
            ("la_st52", "1"): {"name": f"{st5}-l-1", "data": []},
            ("la_st52", "2"): {"name": f"{st5}-l-2", "data": []},
            ("la_st52", "4"): {"name": f"{st5}-l-4", "data": []},
            ("la_c1", "1"): {"name": f"{c1}-1", "data": []},
            ("la_c1", "2"): {"name": f"{c1}-2", "data": []},
            ("la_c1", "4"): {"name": f"{c1}-4", "data": []},
            ("la_st5_11", "1"): {"name": f"{st5}-s,1:1-1", "data": []},
            ("la_st5_11", "2"): {"name": f"{st5}-s,1:1-2", "data": []},
            ("la_st5_11", "3"): {"name": f"{st5}-s,1:1-3", "data": []},
            ("la_st52_11", "1"): {"name": f"{st5}-l,1:1-1", "data": []},
            ("la_st52_11", "2"): {"name": f"{st5}-l,1:1-2", "data": []},
            ("la_st52_11", "3"): {"name": f"{st5}-l,1:1-3", "data": []},
        }
        ax.axvline(x=2 + 0.5, c="gray")
        ax.axvline(x=5 + 0.5, c="gray")
        ax.axvline(x=8 + 0.5, c="gray")
        ax.axvline(x=11 + 0.5, c="gray")

    count_423 = 0
    count_111 = 0
    for entry in cgx.utilities.AtomliteDatabase(database_path).get_entries():
        if "pair" not in entry.properties:
            continue

        multi = entry.properties["multiplier"]
        pair = entry.properties["pair"]
        if (pair, multi) not in systems:
            continue
        energy = entry.properties["energy_per_bb"]

        if entry.properties["num_components"] > 1:
            continue

        systems[(pair, multi)]["data"].append(energy)  # type: ignore[attr-defined]

        if pair == "la_st5_11":
            count_111 += 1
        elif pair == "la_st5":
            count_423 += 1

    if "si" in filename:
        logging.info(
            "structures built for la_st5, 4:2:3 %s, 1:1:1 %s",
            count_423,
            count_111,
        )

    for i, (pair, multi) in enumerate(systems):
        if len(systems[(pair, multi)]["data"]) == 0:
            continue
        min_energy = min(systems[(pair, multi)]["data"])

        ax.scatter(
            simple_beeswarm2(systems[(pair, multi)]["data"], width=0.3) + i,
            systems[(pair, multi)]["data"],
            c="tab:blue",
            alpha=0.2,
            edgecolor="none",
            s=30,
            marker="o",
            zorder=1,
        )
        ax.scatter(
            i,
            min_energy,
            c="none",
            alpha=1.0,
            edgecolor="k",
            s=80,
            marker="o",
            zorder=2,
        )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xticks(list(range(len(systems))))
    rot = 0 if "main" in filename else 90
    ax.set_xticklabels([systems[i]["name"] for i in systems], rotation=rot)  # type: ignore[misc]
    ax.set_ylabel(eb_str(), fontsize=16)
    ax.set_yscale("log")
    ax.set_ylim(0.01, None)

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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="store_true",
        help="set to iterate through structure functions",
    )

    return parser.parse_args()


def main() -> None:  # noqa: PLR0915
    """Run script."""
    args = _parse_args()

    wd = pathlib.Path("/home/atarzia/workingspace/starships/")
    calculation_dir = wd / "rerun_calculations"
    calculation_dir.mkdir(exist_ok=True)
    structure_dir = wd / "rerun_structures"
    structure_dir.mkdir(exist_ok=True)
    ligand_dir = wd / "rerun_ligands"
    ligand_dir.mkdir(exist_ok=True)
    data_dir = wd / "rerun_data"
    data_dir.mkdir(exist_ok=True)
    figure_dir = wd / "figures"
    figure_dir.mkdir(exist_ok=True)

    database_path = data_dir / "rerun.db"

    ligand_measures = {
        "la": {"dd": 7.0, "de": 1.5, "dde": 170, "eg": 1.4, "gb": 1.4},
        "st5": {"ba": 2.8, "aa": 3.9, "bac": 120, "bacab": 180},
        "st52": {"ba": 2.8, "aa": 5.0, "bac": 110, "bacab": 180},
        "c1": {"ba": 2.8, "aa": 3.4, "bac": 90, "bacab": 180},
    }

    pairs = {
        "la_st5": {
            "converging_name": "la",
            "diverging_name": "st5",
            "stoichiometry_L_L_M": (4, 2, 3),
            "converging": cgx.molecular.SixBead(
                bead=cbead_c,
                abead1=abead_c,
                abead2=ebead_c,
            ),
            "diverging": cgx.molecular.TwoC1Arm(
                bead=cbead_d,
                abead1=abead_d,
            ),
            "tetra": cgx.molecular.FourC1Arm(
                bead=tetra_bead,
                abead1=binder_bead,
            ),
            "multipliers": (1, 2, 4),
        },
        "la_st52": {
            "converging_name": "la",
            "diverging_name": "st52",
            "stoichiometry_L_L_M": (4, 2, 3),
            "converging": cgx.molecular.SixBead(
                bead=cbead_c,
                abead1=abead_c,
                abead2=ebead_c,
            ),
            "diverging": cgx.molecular.TwoC1Arm(
                bead=cbead_d,
                abead1=abead_d,
            ),
            "tetra": cgx.molecular.FourC1Arm(
                bead=tetra_bead,
                abead1=binder_bead,
            ),
            "multipliers": (1, 2, 4),
        },
        "la_c1": {
            "converging_name": "la",
            "diverging_name": "c1",
            "stoichiometry_L_L_M": (4, 2, 3),
            "converging": cgx.molecular.SixBead(
                bead=cbead_c,
                abead1=abead_c,
                abead2=ebead_c,
            ),
            "diverging": cgx.molecular.TwoC1Arm(
                bead=cbead_d,
                abead1=abead_d,
            ),
            "tetra": cgx.molecular.FourC1Arm(
                bead=tetra_bead,
                abead1=binder_bead,
            ),
            "multipliers": (1, 2, 4),
        },
        "la_st5_11": {
            "converging_name": "la",
            "diverging_name": "st5",
            "stoichiometry_L_L_M": (1, 1, 1),
            "converging": cgx.molecular.SixBead(
                bead=cbead_c,
                abead1=abead_c,
                abead2=ebead_c,
            ),
            "diverging": cgx.molecular.TwoC1Arm(
                bead=cbead_d,
                abead1=abead_d,
            ),
            "tetra": cgx.molecular.FourC1Arm(
                bead=tetra_bead,
                abead1=binder_bead,
            ),
            "multipliers": (1, 2, 3),
        },
        "la_st52_11": {
            "converging_name": "la",
            "diverging_name": "st52",
            "stoichiometry_L_L_M": (1, 1, 1),
            "converging": cgx.molecular.SixBead(
                bead=cbead_c,
                abead1=abead_c,
                abead2=ebead_c,
            ),
            "diverging": cgx.molecular.TwoC1Arm(
                bead=cbead_d,
                abead1=abead_d,
            ),
            "tetra": cgx.molecular.FourC1Arm(
                bead=tetra_bead,
                abead1=binder_bead,
            ),
            "multipliers": (1, 2, 3),
        },
    }

    if args.run:
        for pair in pairs:
            converging_name = pairs[pair]["converging_name"]
            diverging_name = pairs[pair]["diverging_name"]
            converging = pairs[pair]["converging"]
            diverging = pairs[pair]["diverging"]
            tetra = pairs[pair]["tetra"]

            forcefield = precursors_to_forcefield(
                pair=pair,
                diverging=diverging,
                converging=converging,
                conv_meas=ligand_measures[converging_name],
                dive_meas=ligand_measures[diverging_name],
            )

            converging_name = (
                f"{converging.get_name()}_f{forcefield.get_identifier()}"
            )
            converging_bb = cgx.utilities.optimise_ligand(
                molecule=converging.get_building_block(),
                name=converging_name,
                output_dir=calculation_dir,
                forcefield=forcefield,
                platform=None,
            )
            converging_bb.write(
                str(ligand_dir / f"{converging_name}_optl.mol")
            )
            converging_bb = converging_bb.clone()

            tetra_name = f"{tetra.get_name()}_f{forcefield.get_identifier()}"
            tetra_bb = cgx.utilities.optimise_ligand(
                molecule=tetra.get_building_block(),
                name=tetra_name,
                output_dir=calculation_dir,
                forcefield=forcefield,
                platform=None,
            )
            tetra_bb.write(str(ligand_dir / f"{tetra_name}_optl.mol"))
            tetra_bb = tetra_bb.clone()

            diverging_name = (
                f"{diverging.get_name()}_f{forcefield.get_identifier()}"
            )
            diverging_bb = cgx.utilities.optimise_ligand(
                molecule=diverging.get_building_block(),
                name=diverging_name,
                output_dir=calculation_dir,
                forcefield=forcefield,
                platform=None,
            )
            diverging_bb.write(str(ligand_dir / f"{diverging_name}_optl.mol"))
            diverging_bb = diverging_bb.clone()

            for multiplier in pairs[pair]["multipliers"]:
                # Define a connectivity based on a multiplier.
                iterator = Scrambler(
                    multiplier=multiplier,
                    stoichiometry=pairs[pair]["stoichiometry_L_L_M"],
                    tetra_bb=tetra_bb,
                    converging_bb=converging_bb,
                    diverging_bb=diverging_bb,
                )
                logging.info("doing: pair %s, multi %s", pair, multiplier)
                for constructed in iterator.get_constructed_molecules():
                    idx = constructed.idx
                    acage = constructed.constructed_molecule
                    name = f"{pair}_{multiplier}_{idx}"
                    acage.write(structure_dir / f"{name}_unopt.mol")

                    num_components = len(
                        stko.Network.init_from_molecule(
                            acage
                        ).get_connected_components()
                    )
                    if num_components != 1:
                        continue

                    # Optimise and save.
                    logging.info("building %s", name)

                    try:
                        conformer = cgx.scram.optimise_cage(
                            molecule=acage,
                            name=name,
                            output_dir=calculation_dir,
                            forcefield=forcefield,
                            platform=None,
                            database_path=database_path,
                        )
                        if conformer is not None:
                            conformer.molecule.with_centroid((0, 0, 0)).write(
                                str(structure_dir / f"{name}_optc.mol")
                            )

                        analyse_cage(
                            database_path=database_path,
                            name=name,
                            forcefield=forcefield,
                            iterator=iterator,
                            topology_code=constructed.topology_code,
                        )

                    except OpenMMException:
                        pass

                make_plot(
                    database_path=database_path,
                    pair=pair,
                    structure_dir=structure_dir,
                    figure_dir=figure_dir,
                    filename=f"rerun_1_{pair}.png",
                )

    make_summary_plot(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="rerun_3.png",
    )
    make_summary_plot2(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="rerun_4_main.png",
    )
    make_summary_plot2(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="rerun_4_si.png",
    )
    raise SystemExit
    for pair in pairs:
        make_plot(
            database_path=database_path,
            pair=pair,
            structure_dir=structure_dir,
            figure_dir=figure_dir,
            filename=f"rerun_1_{pair}.png",
        )


if __name__ == "__main__":
    main()
