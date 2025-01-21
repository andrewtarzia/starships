"""Script to generate and optimise CG models."""

import argparse
import itertools as it
import logging
import pathlib

import cgexplore as cgx
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
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
    isomer_energy,
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


def make_plot(  # noqa: PLR0915
    database_path: pathlib.Path,
    figure_dir: pathlib.Path,
    filename: str,
) -> None:
    """Visualise energies."""
    fig, ax = plt.subplots(figsize=(5, 5))
    cg_scale = 2

    if filename == "scan_1.png":
        xoption = "a_c"
        xoption2 = "a_a"
        yoption = "b_a_c"
        red_x = [
            3.4 / (2 * cg_scale),
            5.0 / (2 * cg_scale),
            3.9 / (2 * cg_scale),
        ]
        red_y = [90, 110, 120]
        xlbl = r"$a$-$c$  [$\mathrm{\AA}$]"
        ylbl = "$b$-$a$-$c$  [$^\\circ$]"
        ax.axhline(y=90, c="k", ls="--", alpha=0.5)
        ax.axhline(y=120, c="k", ls="--", alpha=0.5)
        ax.axvline(x=3.4 / (2 * cg_scale), c="k", ls="--", alpha=0.5)
        ax.axvline(x=5.0 / (2 * cg_scale), c="k", ls="--", alpha=0.5)
        xtarget = (3.4 / (2 * cg_scale), 5.0 / (2 * cg_scale))
        ytarget = (90, 120)
    elif filename == "scan_4.png":
        xoption = "d_d_e"
        xoption2 = None
        yoption = "b_a_c"
        red_x = [170] * 3
        red_y = [90, 110, 120]
        xlbl = "$d$-$d$-$e$  [$^\\circ$]"
        ylbl = "$b$-$a$-$c$  [$^\\circ$]"
        ax.axhline(y=90, c="k", ls="--", alpha=0.5)
        ax.axhline(y=120, c="k", ls="--", alpha=0.5)
        ax.axvline(x=170, c="k", ls="--", alpha=0.5)
        xtarget = (170,)
        ytarget = (90, 120)

    elif filename == "scan_7.png":
        xoption = "d_d_e"
        xoption2 = None
        yoption = "d_d"
        red_x = [170]
        red_y = [7 / cg_scale]
        xlbl = "$d$-$d$-$e$  [$^\\circ$]"
        ylbl = r"$d$-$d$  [$\mathrm{\AA}$]"
        ax.axhline(y=7 / cg_scale, c="k", ls="--", alpha=0.5)
        ax.axvline(x=170, c="k", ls="--", alpha=0.5)
        xtarget = (170,)
        ytarget = (7 / cg_scale,)

    else:
        raise NotImplementedError

    vmin = 0
    vmax = 0.6

    for entry in cgx.utilities.AtomliteDatabase(database_path).get_entries():
        if xoption2 is None:
            cname = f"{yoption.replace('_', '')}-{xoption.replace('_', '')}"
        else:
            cname = f"{yoption.replace('_', '')}-{xoption2.replace('_', '')}"
        if cname not in entry.key:
            continue
        x = float(entry.properties["forcefield_dict"]["v_dict"][xoption])
        y = float(entry.properties["forcefield_dict"]["v_dict"][yoption])
        c = float(entry.properties["energy_per_bb"])

        if x in xtarget and y in ytarget:
            logging.info(
                "target (x:%s, y:%s) E=%s",
                round(x, 2),
                round(y, 2),
                round(c, 2),
            )

        ax.scatter(
            x,
            y,
            c=c,
            vmin=vmin,
            vmax=vmax,
            alpha=1.0,
            edgecolor="k",
            s=100,
            marker="s",
            cmap="Blues_r",
        )

    ax.scatter(
        red_x,
        red_y,
        c="tab:red",
        alpha=1.0,
        edgecolor="k",
        s=80,
        marker="X",
    )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel(xlbl, fontsize=16)
    ax.set_ylabel(ylbl, fontsize=16)

    cbar_ax = fig.add_axes([1.01, 0.2, 0.02, 0.7])  # type: ignore[call-overload]
    cmap = mpl.cm.Blues_r  # type: ignore[attr-defined]
    norm = mpl.colors.Normalize(vmin=0, vmax=vmax)
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


def make_contour_plot(  # noqa: PLR0915
    database_path: pathlib.Path,
    figure_dir: pathlib.Path,
    filename: str,
) -> None:
    """Visualise energies."""
    fig, ax = plt.subplots(figsize=(8, 5))
    cg_scale = 2

    if filename in ("scan_1.png", "scan_1c.png"):
        xoption = "a_c"
        xoption2 = "a_a"
        yoption = "b_a_c"
        red_x = [
            3.4 / (2 * cg_scale),
            5.0 / (2 * cg_scale),
            3.9 / (2 * cg_scale),
        ]
        red_y = [90, 110, 120]
        xlbl = r"$a$-$c$  [$\mathrm{\AA}$]"
        ylbl = "$b$-$a$-$c$  [$^\\circ$]"
        ax.axhline(y=90, c="k", ls="--", alpha=0.5)
        ax.axhline(y=120, c="k", ls="--", alpha=0.5)
        ax.axvline(x=3.4 / (2 * cg_scale), c="k", ls="--", alpha=0.5)
        ax.axvline(x=5.0 / (2 * cg_scale), c="k", ls="--", alpha=0.5)

    elif filename in ("scan_4.png", "scan_4c.png"):
        xoption = "d_d_e"
        xoption2 = None
        yoption = "b_a_c"
        red_x = [170] * 3
        red_y = [90, 110, 120]
        xlbl = "$d$-$d$-$e$  [$^\\circ$]"
        ylbl = "$b$-$a$-$c$  [$^\\circ$]"
        ax.axhline(y=90, c="k", ls="--", alpha=0.5)
        ax.axhline(y=120, c="k", ls="--", alpha=0.5)
        ax.axvline(x=170, c="k", ls="--", alpha=0.5)

    elif filename in ("scan_7.png", "scan_7c.png"):
        xoption = "d_d_e"
        xoption2 = None
        yoption = "d_d"
        red_x = [170]
        red_y = [7 / cg_scale]
        xlbl = "$d$-$d$-$e$  [$^\\circ$]"
        ylbl = r"$d$-$d$  [$\mathrm{\AA}$]"
        ax.axhline(y=7 / cg_scale, c="k", ls="--", alpha=0.5)
        ax.axvline(x=170, c="k", ls="--", alpha=0.5)

    else:
        raise NotImplementedError

    if xoption2 is None:
        cname = f"{yoption.replace('_', '')}-{xoption.replace('_', '')}"
    else:
        cname = f"{yoption.replace('_', '')}-{xoption2.replace('_', '')}"

    frame = cgx.utilities.AtomliteDatabase(database_path).get_property_df(
        properties=[
            "$.energy_per_bb",
            f"$.forcefield_dict.v_dict.{xoption}",
            f"$.forcefield_dict.v_dict.{yoption}",
        ]
    )
    frame = frame.filter(pl.col("key").str.contains(cname))

    # Plot the underlying grid.
    ax.scatter(
        frame[f"$.forcefield_dict.v_dict.{xoption}"],
        frame[f"$.forcefield_dict.v_dict.{yoption}"],
        c="none",
        alpha=0.1,
        edgecolor="k",
        s=50,
        marker="s",
        zorder=2,
    )

    xs = set(frame[f"$.forcefield_dict.v_dict.{xoption}"])
    ys = set(frame[f"$.forcefield_dict.v_dict.{yoption}"])
    frame = frame.sort(pl.col(f"$.forcefield_dict.v_dict.{xoption}")).sort(
        pl.col(f"$.forcefield_dict.v_dict.{yoption}")
    )
    frame = frame.group_by(
        f"$.forcefield_dict.v_dict.{xoption}", maintain_order=True
    ).agg(pl.col("$.energy_per_bb"))

    zs = np.array(frame["$.energy_per_bb"].to_list()).T
    xs, ys = np.meshgrid(sorted(set(xs)), sorted(set(ys)))

    cs = ax.contourf(
        xs, ys, zs, levels=[0.0, 0.1, 0.3, 0.6, 1.0], cmap="Blues_r"
    )

    ax.scatter(
        red_x,
        red_y,
        c="tab:red",
        alpha=1.0,
        edgecolor="k",
        s=80,
        marker="X",
    )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel(xlbl, fontsize=16)
    ax.set_ylabel(ylbl, fontsize=16)

    cbar = fig.colorbar(cs)
    cbar.ax.tick_params(labelsize=16)
    cbar.ax.set_ylabel(eb_str(), fontsize=16)

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


def make_singular_plot(  # noqa: PLR0913
    database_path: pathlib.Path,
    figure_dir: pathlib.Path,
    filename: str,
    cname: str,
    xoption: str,
    xlbl: str,
) -> None:
    """Visualise energies."""
    fig, ax = plt.subplots(figsize=(3, 5))

    xs = []
    cs = []
    energies = []
    for entry in cgx.utilities.AtomliteDatabase(database_path).get_entries():
        if cname not in entry.key:
            continue
        x = float(entry.properties["forcefield_dict"]["v_dict"][xoption])

        xs.append(x)
        energies.append(entry.properties["energy_per_bb"])
        cs.append(
            "tab:blue"
            if float(entry.properties["energy_per_bb"]) < isomer_energy()
            else "tab:gray"
        )

    ax.scatter(xs, energies, c=cs, s=80, alpha=1.0, marker="o", ec="k")
    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel(xlbl, fontsize=16)
    ax.set_ylabel(eb_str(), fontsize=16)

    ax.set_yscale("log")

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
    calculation_dir = wd / "scan_calculations"
    calculation_dir.mkdir(exist_ok=True)
    structure_dir = wd / "scan_structures"
    structure_dir.mkdir(exist_ok=True)
    ligand_dir = wd / "scan_ligands"
    ligand_dir.mkdir(exist_ok=True)
    data_dir = wd / "scan_data"
    data_dir.mkdir(exist_ok=True)
    figure_dir = wd / "figures"
    figure_dir.mkdir(exist_ok=True)
    database_path = data_dir / "scan.db"

    aa_range = [3.9, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0]
    bac_range = [120, 90, 95, 100, 105, 110, 115, 125, 130, 135, 140, 145, 150]
    dde_range = [170, 120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 175]
    dd_range = [7.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 9.0]

    pair = "la_st5"
    converging = cgx.molecular.SixBead(
        bead=cbead_c,
        abead1=abead_c,
        abead2=ebead_c,
    )
    converging_name = "la"
    diverging = cgx.molecular.TwoC1Arm(bead=cbead_d, abead1=abead_d)
    diverging_name = "st5"
    tetra = cgx.molecular.FourC1Arm(bead=tetra_bead, abead1=binder_bead)

    combos = {
        "bac-aa": {"yr": bac_range, "xr": aa_range, "yl": "st5", "xl": "st5"},
        "bac-dde": {"yr": bac_range, "xr": dde_range, "yl": "st5", "xl": "la"},
        "dd-dde": {"yr": dd_range, "xr": dde_range, "yl": "la", "xl": "la"},
    }
    if args.run:
        for cname, pair_range_dict in combos.items():
            # Rewrite each time.
            ligand_measures = {
                "la": {"dd": 7.0, "de": 1.5, "dde": 170, "eg": 1.4, "gb": 1.4},
                "st5": {"ba": 2.8, "aa": 3.9, "bac": 120, "bacab": 180},
            }

            for (i, xp), (j, yp) in it.product(
                enumerate(pair_range_dict["xr"]),
                enumerate(pair_range_dict["yr"]),
            ):
                ypname, xpname = cname.split("-")
                ligand_measures[pair_range_dict["xl"]][xpname] = xp
                ligand_measures[pair_range_dict["yl"]][ypname] = yp

                forcefield = precursors_to_forcefield(
                    pair=f"{pair}",
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

                name = f"scan_{cname}_{i}-{j}"
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

                si, sj = name.split("_")[2].split("-")
                potential_names = []
                for cstr in combos:
                    potential_names.extend(
                        [
                            f"scan_{cstr}_{int(si) - 2}-{int(sj) - 2}",
                            f"scan_{cstr}_{int(si) - 1}-{int(sj) - 2}",
                            f"scan_{cstr}_{int(si)}-{int(sj) - 2}",
                            f"scan_{cstr}_{int(si) - 2}-{int(sj) - 1}",
                            f"scan_{cstr}_{int(si) - 1}-{int(sj) - 1}",
                            f"scan_{cstr}_{int(si)}-{int(sj) - 1}",
                            f"scan_{cstr}_{int(si) - 2}-{int(sj)}",
                            f"scan_{cstr}_{int(si) - 1}-{int(sj)}",
                        ]
                    )

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
            for (i, xp), (j, yp) in it.product(
                enumerate(pair_range_dict["xr"]),
                enumerate(pair_range_dict["yr"]),
            ):
                ypname, xpname = cname.split("-")
                ligand_measures[pair_range_dict["xl"]][xpname] = xp
                ligand_measures[pair_range_dict["yl"]][ypname] = yp

                forcefield = precursors_to_forcefield(
                    pair=f"{pair}",
                    diverging=diverging,
                    converging=converging,
                    conv_meas=ligand_measures[converging_name],
                    dive_meas=ligand_measures[diverging_name],
                )

                name = f"scan_{cname}_{i}-{j}"
                logging.info("rescanning %s", name)

                current_cage = stk.BuildingBlock.init_from_file(
                    structure_dir / f"{name}_optc.mol"
                )

                potential_names = []

                x_indices_of_interest = [
                    pair_range_dict["xr"].index(x)
                    for _, x in sorted(
                        zip(
                            [abs(i - xp) for i in pair_range_dict["xr"]],
                            pair_range_dict["xr"],
                            strict=False,
                        )
                    )
                ][:3]
                y_indices_of_interest = [
                    pair_range_dict["yr"].index(x)
                    for _, x in sorted(
                        zip(
                            [abs(i - yp) for i in pair_range_dict["yr"]],
                            pair_range_dict["yr"],
                            strict=False,
                        )
                    )
                ][:3]

                for cstr, xidx, yidx in it.product(
                    combos, x_indices_of_interest, y_indices_of_interest
                ):
                    potential_names.append(f"scan_{cstr}_{xidx}-{yidx}")

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
        filename="scan_1.png",
    )
    make_contour_plot(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="scan_1c.png",
    )
    make_singular_plot(
        database_path=database_path,
        figure_dir=figure_dir,
        cname="bac-aa",
        xoption="a_c",
        xlbl=r"$a$-$c$  [$\mathrm{\AA}$]",
        filename="scan_2.png",
    )
    make_singular_plot(
        database_path=database_path,
        figure_dir=figure_dir,
        cname="bac-aa",
        xoption="b_a_c",
        xlbl="$b$-$a$-$c$  [$^\\circ$]",
        filename="scan_3.png",
    )

    make_plot(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="scan_4.png",
    )
    make_contour_plot(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="scan_4c.png",
    )
    make_singular_plot(
        database_path=database_path,
        figure_dir=figure_dir,
        cname="bac-dde",
        xoption="d_d_e",
        xlbl="$d$-$d$-$e$  [$^\\circ$]",
        filename="scan_5.png",
    )
    make_singular_plot(
        database_path=database_path,
        figure_dir=figure_dir,
        cname="bac-dde",
        xoption="b_a_c",
        xlbl="$b$-$a$-$c$  [$^\\circ$]",
        filename="scan_6.png",
    )

    make_plot(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="scan_7.png",
    )
    make_contour_plot(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="scan_7c.png",
    )
    make_singular_plot(
        database_path=database_path,
        figure_dir=figure_dir,
        cname="dd-dde",
        xoption="d_d",
        xlbl=r"$d$-$d$  [$\mathrm{\AA}$]",
        filename="scan_8.png",
    )
    make_singular_plot(
        database_path=database_path,
        figure_dir=figure_dir,
        cname="dd-dde",
        xoption="d_d_e",
        xlbl="$d$-$d$-$e$  [$^\\circ$]",
        filename="scan_9.png",
    )


if __name__ == "__main__":
    main()
