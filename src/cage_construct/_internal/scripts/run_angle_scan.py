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
import stko
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


def save_to_chemiscope(
    database_path: pathlib.Path, figure_dir: pathlib.Path, cname: str
) -> None:
    """Save one grid to chemiscope."""
    xoption = "d_d_e"
    yoption = "b_a_c"

    properties_to_get = (
        "E_b / kjmol-1",
        "bac / deg",
        "dde / deg",
        "avg. BA_r / deg",
        "avg. BA_t / deg",
    )

    database = cgx.utilities.AtomliteDatabase(database_path)
    structures = []
    properties = {i: [] for i in properties_to_get}
    for entry in database.get_entries():
        if cname not in entry.key:
            continue
        properties["dde / deg"].append(
            float(entry.properties["forcefield_dict"]["v_dict"][xoption])
        )
        properties["bac / deg"].append(
            float(entry.properties["forcefield_dict"]["v_dict"][yoption])
        )
        properties["E_b / kjmol-1"].append(
            float(entry.properties["energy_per_bb"])
        )
        properties["avg. BA_r / deg"].append(
            np.mean(entry.properties["diverging_binder_binder_angles"])
        )
        properties["avg. BA_t / deg"].append(
            np.mean(entry.properties["converging_binder_binder_angles"])
        )

        structures.append(database.get_molecule(entry.key))

    logging.info(
        "structures: %s, properties: %s",
        len(structures),
        len(properties),
    )
    cgx.utilities.write_chemiscope_json(
        json_file=figure_dir / "cs_bac-dde_starships.json.gz",
        structures=structures,
        properties=properties,
        bonds_as_shapes=True,
        meta_dict={
            "name": "CGGeom: starshsips",
            "description": ("Minimal models in starship topology."),
            "authors": ["Andrew Tarzia"],
            "references": [],
        },
        x_axis_dict={"property": "dde / deg"},
        y_axis_dict={"property": "bac / deg"},
        z_axis_dict={"property": ""},
        color_dict={"property": "E_b / kjmol-1", "min": 0, "max": 1.0},
        bond_hex_colour="#919294",
    )


def analyse_cage(
    database_path: pathlib.Path,
    name: str,
    forcefield: cgx.forcefields.ForceField,
    num_building_blocks: int,
) -> None:
    """Analyse a toy model cage."""
    database = cgx.utilities.AtomliteDatabase(database_path)
    properties = database.get_entry(key=name).properties
    final_molecule = database.get_molecule(name)

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

    g_measure = cgx.analysis.GeomMeasure.from_forcefield(forcefield)
    bond_data = g_measure.calculate_bonds(final_molecule)
    bond_data = {str("_".join(i)): bond_data[i] for i in bond_data}
    angle_data = g_measure.calculate_angles(final_molecule)
    angle_data = {str("_".join(i)): angle_data[i] for i in angle_data}
    dihedral_data = g_measure.calculate_torsions(
        molecule=final_molecule,
        absolute=True,
    )
    database.add_properties(
        key=name,
        property_dict={
            "bond_data": bond_data,
            "angle_data": angle_data,
            "dihedral_data": dihedral_data,
        },
    )

    ligands = stko.molecule_analysis.DecomposeMOC().decompose(
        molecule=final_molecule,
        metal_atom_nos=(46,),
    )

    # Get the bg angles.
    c_binder_binder_angles = []
    d_binder_binder_angles = []
    for lig in ligands:
        if lig.get_num_atoms() == 8:  # noqa: PLR2004
            as_building_block = stk.BuildingBlock.init_from_molecule(
                lig,
                stk.SmartsFunctionalGroupFactory(
                    smarts="[Pb]~[Ga]", bonders=(0,), deleters=(1,)
                ),
            )
            converging = True
        elif lig.get_num_atoms() == 5:  # noqa: PLR2004
            as_building_block = stk.BuildingBlock.init_from_molecule(
                lig,
                stk.SmartsFunctionalGroupFactory(
                    smarts="[Pb]~[Ba]", bonders=(0,), deleters=(1,)
                ),
            )
            converging = False

        if as_building_block.get_num_functional_groups() != 2:  # noqa: PLR2004
            raise RuntimeError

        vectors = [
            as_building_block.get_centroid(atom_ids=fg.get_bonder_ids())
            - as_building_block.get_centroid(atom_ids=fg.get_deleter_ids())
            for fg in as_building_block.get_functional_groups()
        ]
        normed = [i / np.linalg.norm(i) for i in vectors]
        angle = np.degrees(
            stko.vector_angle(vector1=normed[0], vector2=normed[1])
        )
        if converging:
            c_binder_binder_angles.append(angle)
        else:
            d_binder_binder_angles.append(angle)

    database.add_properties(
        key=name,
        property_dict={
            "converging_binder_binder_angles": c_binder_binder_angles,
            "diverging_binder_binder_angles": d_binder_binder_angles,
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
        xlbl = r"$ac$  [$\mathrm{\AA}$]"
        ylbl = "$bac$  [$^\\circ$]"
        ax.axhline(y=90, c="k", ls="--", alpha=0.5)
        ax.axhline(y=110, c="k", ls="--", alpha=0.5)
        ax.axhline(y=120, c="k", ls="--", alpha=0.5)
        ax.axvline(x=3.4 / (2 * cg_scale), c="k", ls="--", alpha=0.5)
        ax.axvline(x=5.0 / (2 * cg_scale), c="k", ls="--", alpha=0.5)
        xtarget = (
            3.4 / (2 * cg_scale),
            3.9 / (2 * cg_scale),
            5.0 / (2 * cg_scale),
        )
        ytarget = (90, 110, 120)

    elif filename == "scan_4.png":
        xoption = "d_d_e"
        xoption2 = None
        yoption = "b_a_c"
        red_x = [170] * 3
        red_y = [90, 110, 120]
        xlbl = "$dde$  [$^\\circ$]"
        ylbl = "$bac$  [$^\\circ$]"
        ax.axhline(y=90, c="k", ls="--", alpha=0.5)
        ax.axhline(y=110, c="k", ls="--", alpha=0.5)
        ax.axhline(y=120, c="k", ls="--", alpha=0.5)
        ax.axvline(x=170, c="k", ls="--", alpha=0.5)
        xtarget = (170,)
        ytarget = (90, 110, 120)

    elif filename == "scan_7.png":
        xoption = "d_d_e"
        xoption2 = None
        yoption = "d_d"
        red_x = [170]
        red_y = [7 / cg_scale]
        xlbl = "$dde$  [$^\\circ$]"
        ylbl = r"$dd$  [$\mathrm{\AA}$]"
        ax.axhline(y=7 / cg_scale, c="k", ls="--", alpha=0.5)
        ax.axvline(x=170, c="k", ls="--", alpha=0.5)
        xtarget = (170,)
        ytarget = (7 / cg_scale,)

    elif filename == "scan_13.png":
        xoption2 = None
        xoption = "d_d"
        yoption = "b_a_c"
        red_x = [7 / cg_scale] * 3
        red_y = [90, 110, 120]
        xlbl = r"$dd$  [$\mathrm{\AA}$]"
        ylbl = "$bac$  [$^\\circ$]"
        ax.axhline(y=90, c="k", ls="--", alpha=0.5)
        ax.axhline(y=110, c="k", ls="--", alpha=0.5)
        ax.axhline(y=120, c="k", ls="--", alpha=0.5)
        ax.axvline(x=7 / cg_scale, c="k", ls="--", alpha=0.5)

        xtarget = (7 / cg_scale,)
        ytarget = (90, 110, 120)

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


def make_energy_plot(
    database_path: pathlib.Path,
    figure_dir: pathlib.Path,
    filename: str,
) -> None:
    """Visualise energies."""
    fig, axs = plt.subplots(
        ncols=4, nrows=2, sharex=True, sharey=True, figsize=(16, 10)
    )

    combos = ("bac-aa", "bac-dde", "dd-dde", "bac-dd")
    row_plot = (
        {
            "ffx": "b_a_c",
            "aay": "diverging_binder_binder_angles",
            "xlim": (0, 120),
            "ylim": (None, None),
            "ylbl": eb_str(),
            "xlbl": "observed rigid angle  [$^\\circ$]",
            "obs_source": "bba",
        },
        {
            "ffx": "d_d_e",
            "aay": "converging_binder_binder_angles",
            "xlim": (0, 120),
            "ylim": (None, None),
            "ylbl": eb_str(),
            "xlbl": "observed twistable angle  [$^\\circ$]",
            "obs_source": "bba",
        },
    )
    for axrow, rowd in zip(axs, row_plot, strict=True):
        for combo, ax in zip(combos, axrow, strict=True):
            xs = []
            ys = []

            for entry in cgx.utilities.AtomliteDatabase(
                database_path
            ).get_entries():
                if combo != entry.key.split("_")[1]:
                    continue

                ys.append(float(entry.properties["energy_per_bb"]))
                if rowd["obs_source"] == "ff":
                    xs.append(entry.properties["angle_data"][rowd["aay"]])
                elif rowd["obs_source"] == "bba":
                    xs.append(np.mean(entry.properties[rowd["aay"]]))

            ax.scatter(
                xs,
                ys,
                c="tab:blue",
                alpha=1.0,
                edgecolor="k",
                s=60,
                zorder=2,
            )

            ax.tick_params(axis="both", which="major", labelsize=16)
            ax.set_title(f"${combo}$", fontsize=16)
            ax.set_xlabel(rowd["xlbl"], fontsize=16)
            ax.set_ylabel(rowd["ylbl"], fontsize=16)
            ax.set_xlim(rowd["xlim"])
            ax.set_ylim(rowd["ylim"])

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


def make_geom_plot(  # noqa: C901
    database_path: pathlib.Path,
    figure_dir: pathlib.Path,
    filename: str,
) -> None:
    """Visualise energies."""
    fig, axs = plt.subplots(ncols=3, nrows=4, figsize=(16, 16))
    combos = ("bac-aa", "bac-dde", "dd-dde")
    row_plot = (
        {
            "ffx": "b_a_c",
            "aay": "Pb_Ba_Ag",
            "xlim": (90, 150),
            "ylim": (90, 150),
            "targets": [90, 110, 120],
            "xlbl": "target $bac$  [$^\\circ$]",
            "ylbl": "observed $bac$  [$^\\circ$]",
            "obs_source": "ff",
        },
        {
            "ffx": "d_d_e",
            "aay": "Fe_Ni_Ni",
            "xlim": (120, 175),
            "ylim": (120, 175),
            "targets": [170],
            "xlbl": "target $dde$  [$^\\circ$]",
            "ylbl": "observed $dde$  [$^\\circ$]",
            "obs_source": "ff",
        },
        {
            "ffx": "b_a_c",
            "aay": "diverging_binder_binder_angles",
            "xlim": (90, 150),
            "ylim": (30, 100),
            "targets": [90, 110, 120],
            "xlbl": "target $bac$  [$^\\circ$]",
            "ylbl": "observed diverging angle  [$^\\circ$]",
            "obs_source": "bba",
        },
        {
            "ffx": "d_d_e",
            "aay": "converging_binder_binder_angles",
            "xlim": (120, 175),
            "ylim": (15, 60),
            "targets": [170],
            "xlbl": "target $dde$  [$^\\circ$]",
            "ylbl": "observed converging angle  [$^\\circ$]",
            "obs_source": "bba",
        },
    )
    for axrow, rowd in zip(axs, row_plot, strict=True):
        for combo, ax in zip(combos, axrow, strict=True):
            xs = []
            ys = []

            for entry in cgx.utilities.AtomliteDatabase(
                database_path
            ).get_entries():
                if combo != entry.key.split("_")[1]:
                    continue

                xs.append(
                    float(
                        entry.properties["forcefield_dict"]["v_dict"][
                            rowd["ffx"]
                        ]
                    )
                )
                if rowd["obs_source"] == "ff":
                    ys.append(entry.properties["angle_data"][rowd["aay"]])
                elif rowd["obs_source"] == "bba":
                    ys.append(entry.properties[rowd["aay"]])

            comp_values = {i: [] for i in sorted(set(xs))}
            for i, j in zip(xs, ys, strict=True):
                comp_values[i].extend(j)

            ax.scatter(
                list(comp_values),
                [np.mean(y) for y in comp_values.values()],
                c="tab:blue",
                alpha=1.0,
                edgecolor="k",
                s=60,
                zorder=2,
            )
            ax.fill_between(
                list(comp_values),
                y1=[np.min(comp_values[i]) for i in comp_values],
                y2=[np.max(comp_values[i]) for i in comp_values],
                alpha=0.6,
                color="tab:blue",
                edgecolor=(0, 0, 0, 2.0),
                lw=0,
            )

            ax.tick_params(axis="both", which="major", labelsize=16)
            ax.set_title(f"${combo}$", fontsize=16)
            ax.set_xlabel(rowd["xlbl"], fontsize=16)
            ax.set_ylabel(rowd["ylbl"], fontsize=16)
            ax.set_xlim(rowd["xlim"])
            ax.set_ylim(rowd["ylim"])
            if rowd["obs_source"] == "ff":
                ax.plot(rowd["xlim"], rowd["ylim"], c="k", zorder=-1)
                for t in rowd["targets"]:
                    ax.axhline(t, c="k", ls="--", zorder=-2, alpha=0.4)

            for t in rowd["targets"]:
                ax.axvline(t, c="k", ls="--", zorder=-2, alpha=0.4)

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


def make_geom_grid(
    database_path: pathlib.Path,
    figure_dir: pathlib.Path,
    filename: str,
) -> None:
    """Visualise energies."""
    fig, axs = plt.subplots(ncols=4, nrows=1, figsize=(16, 5))
    combos = ("bac-aa", "bac-dde", "dd-dde", "bac-dd")

    vmin = 0
    vmax = 0.6
    row_plot = (
        {
            "ffx": "diverging_binder_binder_angles",
            "aay": "converging_binder_binder_angles",
            "xlim": (0, 130),
            "ylim": (15, 50),
            "xlbl": "observed rigid angle  [$^\\circ$]",
            "ylbl": "observed twistable angle  [$^\\circ$]",
        },
    )
    for axrow, rowd in zip([axs], row_plot, strict=True):
        for combo, ax in zip(combos, axrow, strict=True):
            min_stable_x = float("inf")
            max_stable_x = 0
            min_stable_y = float("inf")
            max_stable_y = 0
            for entry in cgx.utilities.AtomliteDatabase(
                database_path
            ).get_entries():
                if combo != entry.key.split("_")[1]:
                    continue

                xs = entry.properties[rowd["ffx"]]
                ys = entry.properties[rowd["aay"]]
                c = float(entry.properties["energy_per_bb"])
                if c < 0.1:  # noqa: PLR2004
                    zorder = 2
                    min_stable_x = min((min(xs), min_stable_x))
                    max_stable_x = max((max(xs), max_stable_x))
                    min_stable_y = min((min(ys), min_stable_y))
                    max_stable_y = max((max(ys), max_stable_y))

                elif c < 0.3:  # noqa: PLR2004
                    zorder = 1
                else:
                    zorder = 0

                ax.scatter(
                    np.mean(xs),
                    np.mean(ys),
                    c=c,
                    alpha=1.0,
                    edgecolor="k",
                    s=80,
                    zorder=zorder,
                    vmin=vmin,
                    vmax=vmax,
                    cmap="Blues_r",
                )

            ax.tick_params(axis="both", which="major", labelsize=16)
            ax.set_title(f"${combo}$", fontsize=16)
            ax.set_xlabel(rowd["xlbl"], fontsize=16)
            ax.set_ylabel(rowd["ylbl"], fontsize=16)
            ax.set_xlim(rowd["xlim"])
            ax.set_ylim(rowd["ylim"])
            ax.axhspan(
                ymin=min_stable_y,
                ymax=max_stable_y,
                facecolor="k",
                alpha=0.2,
            )
            ax.axvspan(
                xmin=min_stable_x,
                xmax=max_stable_x,
                facecolor="k",
                alpha=0.2,
            )

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


def make_main_paper_geom_grid(  # noqa: PLR0915
    database_path: pathlib.Path,
    figure_dir: pathlib.Path,
    filename: str,
) -> None:
    """Visualise energies."""
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 5))
    combos = ("bac-dde",)
    xoption = "d_d_e"
    yoption = "b_a_c"
    pairs = {
        (120, 145): ("tab:red", "1"),
        (130, 100): ("tab:orange", "2"),
        (145, 125): ("tab:green", "3"),
        (160, 125): ("tab:purple", "4"),
        (170, 120): ("tab:cyan", "5-predicted"),
    }

    vmin = 0
    vmax = 0.6
    rowd = {
        "ffx": "diverging_binder_binder_angles",
        "aay": "converging_binder_binder_angles",
        "xlim": (0, 130),
        "ylim": (15, 50),
        "xlbl": "observed L1-like angle  [$^\\circ$]",
        "ylbl": "observed L2-like angle  [$^\\circ$]",
    }

    min_stable_x = float("inf")
    max_stable_x = 0
    min_stable_y = float("inf")
    max_stable_y = 0
    for entry in cgx.utilities.AtomliteDatabase(database_path).get_entries():
        if combos[0] != entry.key.split("_")[1]:
            continue

        x = float(entry.properties["forcefield_dict"]["v_dict"][xoption])
        y = float(entry.properties["forcefield_dict"]["v_dict"][yoption])

        xs = entry.properties[rowd["ffx"]]
        ys = entry.properties[rowd["aay"]]
        c = float(entry.properties["energy_per_bb"])
        if c < 0.1:  # noqa: PLR2004
            zorder = 1
            min_stable_x = min((min(xs), min_stable_x))
            max_stable_x = max((max(xs), max_stable_x))
            min_stable_y = min((min(ys), min_stable_y))
            max_stable_y = max((max(ys), max_stable_y))

        elif c < 0.3:  # noqa: PLR2004
            zorder = 0
        else:
            zorder = -1

        if (x, y) in pairs:
            logging.info("see: %s, with x: %s and y: %s", entry.key, x, y)
            ax.scatter(
                np.mean(xs),
                np.mean(ys),
                c="none",
                alpha=1.0,
                edgecolor=pairs[(x, y)][0],
                s=200,
                linewidth=2,
                zorder=2,
                label=pairs[(x, y)][1],
            )

        ax.scatter(
            np.mean(xs),
            np.mean(ys),
            c=c,
            alpha=1.0,
            edgecolor="k",
            s=60,
            zorder=zorder,
            vmin=vmin,
            vmax=vmax,
            cmap="Blues_r",
        )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel(rowd["xlbl"], fontsize=16)
    ax.set_ylabel(rowd["ylbl"], fontsize=16)
    ax.set_xlim(rowd["xlim"])
    ax.set_ylim(rowd["ylim"])
    ax.axhspan(
        ymin=min_stable_y,
        ymax=max_stable_y,
        facecolor="k",
        alpha=0.2,
        zorder=-2,
    )
    ax.axvspan(
        xmin=min_stable_x,
        xmax=max_stable_x,
        facecolor="k",
        alpha=0.2,
        zorder=-2,
    )

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
    ax.legend(fontsize=16)
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
    fig, ax = plt.subplots(figsize=(6, 3))
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
        xlbl = r"$ac$  [$\mathrm{\AA}$]"
        ylbl = "$bac$  [$^\\circ$]"
        ax.axhline(y=90, c="k", ls="--", alpha=0.5)
        ax.axhline(y=110, c="k", ls="--", alpha=0.5)
        ax.axhline(y=120, c="k", ls="--", alpha=0.5)
        ax.axvline(x=3.4 / (2 * cg_scale), c="k", ls="--", alpha=0.5)
        ax.axvline(x=5.0 / (2 * cg_scale), c="k", ls="--", alpha=0.5)

    elif filename in ("scan_4.png", "scan_4c.png"):
        xoption = "d_d_e"
        xoption2 = None
        yoption = "b_a_c"
        red_x = [170] * 3
        red_y = [90, 110, 120]
        xlbl = "$dde$  [$^\\circ$]"
        ylbl = "$bac$  [$^\\circ$]"
        ax.axhline(y=90, c="k", ls="--", alpha=0.5)
        ax.axhline(y=110, c="k", ls="--", alpha=0.5)
        ax.axhline(y=120, c="k", ls="--", alpha=0.5)
        ax.axvline(x=170, c="k", ls="--", alpha=0.5)

    elif filename in ("scan_7.png", "scan_7c.png"):
        xoption = "d_d_e"
        xoption2 = None
        yoption = "d_d"
        red_x = [170]
        red_y = [7 / cg_scale]
        xlbl = "$dde$  [$^\\circ$]"
        ylbl = r"$dd$  [$\mathrm{\AA}$]"
        ax.axhline(y=7 / cg_scale, c="k", ls="--", alpha=0.5)
        ax.axvline(x=170, c="k", ls="--", alpha=0.5)

    elif filename in ("scan_13c.png", "scan_13.png"):
        xoption = "d_d"
        xoption2 = None
        yoption = "b_a_c"
        red_x = [7 / cg_scale] * 3
        red_y = [90, 110, 120]
        xlbl = r"$dd$  [$\mathrm{\AA}$]"
        ylbl = "$bac$  [$^\\circ$]"
        ax.axvline(x=7 / cg_scale, c="k", ls="--", alpha=0.5)
        ax.axhline(y=90, c="k", ls="--", alpha=0.5)
        ax.axhline(y=110, c="k", ls="--", alpha=0.5)
        ax.axhline(y=120, c="k", ls="--", alpha=0.5)

    else:
        raise NotImplementedError

    if xoption2 is None:
        cname = f"{yoption.replace('_', '')}-{xoption.replace('_', '')}_"
    else:
        cname = f"{yoption.replace('_', '')}-{xoption2.replace('_', '')}_"

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
        xs,
        ys,
        zs,
        levels=[0.0, 0.1, 0.3, 0.6, 1.0],
        cmap="Blues_r",
        alpha=0.8,
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


def names_to_viz(
    database_path: pathlib.Path,
    cname: str,
) -> None:
    """Visualise energies."""
    xoption = "d_d_e"
    yoption = "b_a_c"

    pairs = ((120, 145), (130, 100), (145, 125), (160, 125), (170, 120))

    for entry in cgx.utilities.AtomliteDatabase(database_path).get_entries():
        if cname not in entry.key:
            continue
        x = float(entry.properties["forcefield_dict"]["v_dict"][xoption])
        y = float(entry.properties["forcefield_dict"]["v_dict"][yoption])

        if (x, y) in pairs:
            logging.info("see: %s, with x: %s and y: %s", entry.key, x, y)


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
        "bac-dd": {"yr": bac_range, "xr": dd_range, "yl": "st5", "xl": "la"},
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

    save_to_chemiscope(
        database_path=database_path, figure_dir=figure_dir, cname="bac-dde"
    )
    make_main_paper_geom_grid(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="scan_14.png",
    )
    make_plot(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="scan_13.png",
    )
    make_contour_plot(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="scan_13c.png",
    )

    make_geom_grid(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="scan_12.png",
    )
    make_energy_plot(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="scan_11.png",
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

    make_geom_plot(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="scan_10.png",
    )

    names_to_viz(database_path=database_path, cname="bac-dde")


if __name__ == "__main__":
    main()
