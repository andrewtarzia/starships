"""Script to generate and optimise CG models."""

import argparse
import itertools as it
import logging
import pathlib
from collections import defaultdict

import cgexplore as cgx
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import stk
import stko
from morfeus import Sterimol, read_xyz
from rdkit import RDLogger

from .utilities import (
    abead_c,
    abead_d,
    binder_bead,
    cbead_c,
    cbead_d,
    eb_str,
    ebead_c,
    inner_bead,
    precursors_to_forcefield,
    steric_bead,
    tetra_bead,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
RDLogger.DisableLog("rdApp.*")

cmap = {
    "3P6": "xkcd:dark blue",
    # "3P6s": "xkcd:blue",  # noqa: ERA001
    "3P6si": "xkcd:light blue",
    "4P8": "xkcd:dark orange",
    # "4P8s": "xkcd:orange",# noqa: ERA001
    "4P8si": "xkcd:light orange",
    "4P82": "xkcd:forest green",
    # "4P82s": "xkcd:kelly green",# noqa: ERA001
    "4P82si": "xkcd:chartreuse",
    "6P12": "xkcd:royal purple",
    # "6P12s": "xkcd:violet",# noqa: ERA001
    "6P12si": "xkcd:lilac",
}

dmap = {
    "3P6": ("xkcd:dark blue", "M$_3$L$_4$L'$_2$"),
    "3P6s": ("xkcd:blue", "M$_3$L$_4$L'$_2$"),
    "3P6si": ("xkcd:blue", "M$_3$L$_4$L'$_2$"),
    "4P8": ("xkcd:dark orange", "M$_4$L$_4$L'$_4$-sq"),
    "4P8s": ("xkcd:orange", "M$_4$L$_4$L'$_4$-sq"),
    "4P8si": ("xkcd:orange", "M$_4$L$_4$L'$_4$-sq"),
    "4P82": ("xkcd:forest green", "M$_4$L$_4$L'$_4$-td"),
    "4P82s": ("xkcd:kelly green", "M$_4$L$_4$L'$_4$-td"),
    "4P82si": ("xkcd:kelly green", "M$_4$L$_4$L'$_4$-td"),
    "6P12": ("xkcd:royal purple", "M$_6$L$_12$"),
    "6P12s": ("xkcd:violet", "M$_6$L$_12$"),
    "6P12si": ("xkcd:violet", "M$_6$L$_12$"),
}


ls = {
    "3P6": "-",
    "3P6s": "--",
    "3P6si": "-",
    "4P8": "-",
    "4P8s": "--",
    "4P8si": "-",
    "4P82": "-",
    "4P82s": "--",
    "4P82si": "-",
    "6P12": "-",
    "6P12s": "--",
    "6P12si": "-",
}

final_conformer = 9


def get_steric_measures(figure_dir: pathlib.Path) -> dict[str, Sterimol]:
    """Calculate and add to plot."""
    figure_dir.mkdir(exist_ok=True)

    molecules = {
        "L2": stko.UFF().optimize(stk.BuildingBlock("C1=CC(=O)C=C1")),
        "L3": stko.UFF().optimize(stk.BuildingBlock("C1=COC=C1")),
        "L4": stko.UFF().optimize(stk.BuildingBlock("CC1=COC=C1")),
        "L5": stko.UFF().optimize(stk.BuildingBlock("CC1(C=CC=C1)C")),
        "L6": stko.UFF().optimize(
            stk.BuildingBlock("CCCCCCC1(C=CC=C1)CCCCCC")
        ),
        "L7": stko.UFF().optimize(stk.BuildingBlock("C1=CC=C(C=C1)C2=COC=C2")),
    }
    # Build molecules and then modify them
    for molname, molecule in molecules.items():
        molecule.write(figure_dir / f"{molname}.xyz")

    results = {}
    for molname in molecules:
        man_file = figure_dir / f"{molname}_manual.xyz"
        if not man_file.exists():
            raise RuntimeError
        elements, coordinates = read_xyz(man_file)
        sterimol = Sterimol(elements, coordinates, 1, 2)

        logging.info(
            "%s: L: %s, B_1: %s, B_5: %s.",
            molname,
            round(sterimol.L_value, 2),
            round(sterimol.B_1_value, 2),
            round(sterimol.B_5_value, 2),
        )

        results[molname] = sterimol
    return results


def convert_coordinates(
    molecule: stk.ConstructedMolecule,
    nx_positions: np.ndarray,
) -> stk.ConstructedMolecule:
    """Convert networkx coordinates to molecule position matrix."""
    # We allow these to independantly fail because the nx graphs can
    # be ridiculous, just get the first that passes.
    for scaler in (3, 5, 10, 15):
        pos_mat = np.array([nx_positions[i] for i in nx_positions])
        new_mol = molecule.with_position_matrix(pos_mat * scaler)
        break
    return new_mol.with_centroid(np.array((0.0, 0.0, 0.0)))


def get_from_0_coordinares(
    molecule: stk.ConstructedMolecule,
    tstr: str,
    structure_dir: pathlib.Path,
) -> stk.ConstructedMolecule:
    """Translate to coordinates with a previous structure with less atoms."""
    previous = (
        structure_dir / f"ts_{tstr.replace('s', '').replace('i', '')}_0_0_"
        f"{final_conformer}_optc.mol"
    )

    if not previous.exists():
        msg = f"{previous} does not exist"
        raise RuntimeError(msg)
    previous_mol = stk.BuildingBlock.init_from_file(previous)
    new_position_matrix = list(previous_mol.get_position_matrix())

    bonds = list(molecule.get_bonds())

    # Insert 0s at missing ids.
    missing_ids = tuple(
        i.get_id()
        for i in molecule.get_atoms()
        if i.get_atomic_number() in (16, 77)
    )
    for mid in missing_ids:
        new_position_matrix.insert(mid, np.array((0.0, 0.0, 0.0)))

    # Update positions.
    new_mol = molecule.with_position_matrix(np.asarray(new_position_matrix))

    # Now place new atoms in between their neighbours if Ni, else toward
    # centroid if S.
    ni_ids = [
        i.get_id()
        for i in molecule.get_atoms()
        if i.get_atomic_number() == 77  # noqa: PLR2004
    ]
    for ni in ni_ids:
        neighbours = [
            i.get_atom1().get_id()
            if i.get_atom2().get_id() == ni
            else i.get_atom2().get_id()
            for i in bonds
            if ni in (i.get_atom1().get_id(), i.get_atom2().get_id())
            # Make sure not to include the steric atom.
            and 16  # noqa: PLR2004
            not in (
                i.get_atom1().get_atomic_number(),
                i.get_atom2().get_atomic_number(),
            )
        ]
        new_position_matrix[ni] = new_mol.get_centroid(atom_ids=neighbours)

    # Update positions.
    new_mol = molecule.with_position_matrix(np.asarray(new_position_matrix))

    s_ids = [
        i.get_id()
        for i in molecule.get_atoms()
        if i.get_atomic_number() == 16  # noqa: PLR2004
    ]
    for ni in s_ids:
        neighbours = [
            i.get_atom1().get_id()
            if i.get_atom2().get_id() == ni
            else i.get_atom2().get_id()
            for i in bonds
            if ni in (i.get_atom1().get_id(), i.get_atom2().get_id())
        ]

        cage_centroid = new_mol.get_centroid()
        neigh_centroid = new_mol.get_centroid(atom_ids=neighbours)
        vector = cage_centroid - neigh_centroid
        norm_vector = vector / np.linalg.norm(vector)

        new_position_matrix[ni] = neigh_centroid + norm_vector

    # Update positions.
    return new_mol.with_position_matrix(np.asarray(new_position_matrix))


def geom_distributions(  # noqa: C901
    database_path: pathlib.Path,
    figure_dir: pathlib.Path,
    fileprefix: str,
) -> None:
    """Plot geometry distributions."""
    targets = {
        "Pb_Ga_Fe": 120,
        "Pb_Ba_Ag": 120,
        "Ga_Fe_Ni": 180,
        "Fe_Ni_Ni": 170,
        "Ba_Ag_Ba": 180,
        "Fe_Ni_Ir": 170,
        "Ni_Ir_Ni": 180,
        "Ni_Ir_S": 90,
        "Pb_Ba_Ag_Ba_Pb": 0,
        "Fe_Ni_Ni_Fe": 0,
        "Fe_Ni_Ir_S": 0,
        "Fe_Ni_Ir_Ni_Fe": 0,
    }

    geom_dict: dict[str, dict[str, list[float]]] = {
        i: defaultdict(list) for i in targets
    }

    database = cgx.utilities.AtomliteDatabase(database_path)
    for entry in database.get_entries():
        if entry.properties["attempt"] != f"{final_conformer}":
            continue
        for label, gd_data in entry.properties["bond_data"].items():
            if label not in targets:
                continue
            geom_dict[label][entry.properties["tstr"]].extend(gd_data)
        for label, gd_data in entry.properties["angle_data"].items():
            if label not in targets:
                continue
            geom_dict[label][entry.properties["tstr"]].extend(gd_data)
        for label, gd_data in entry.properties["dihedral_data"].items():
            if label not in targets:
                continue
            geom_dict[label][entry.properties["tstr"]].extend(gd_data)

    fig, axs = plt.subplots(ncols=4, nrows=3, figsize=(16, 10))
    flat_axs = axs.flatten()
    for ax, (label, tstr_dict) in zip(
        flat_axs, geom_dict.items(), strict=True
    ):
        target = targets[label]
        for tstr, col in cmap.items():
            xdata = tstr_dict[tstr]
            if len(xdata) == 0:
                continue

            xwidth = 2
            xmin = -40
            xmax = 40

            xbins = np.arange(xmin - xwidth, xmax + xwidth, xwidth)

            ax.hist(
                x=[i - target for i in xdata],
                bins=list(xbins),
                density=True,
                histtype="stepfilled",
                stacked=True,
                facecolor=col,
                linewidth=1.0,
                edgecolor="none",
                label=dmap[tstr][1],
                alpha=1.0,
            )

        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.set_title(label, fontsize=16)
        ax.set_xlabel("value-target", fontsize=16)
        ax.set_ylabel("frequency", fontsize=16)
        ax.set_yticks([])
        if label == "Ni_Ir_S":
            ax.legend(ncol=1, fontsize=16)

    fig.tight_layout()
    fig.savefig(
        figure_dir / f"{fileprefix}.png",
        dpi=360,
        bbox_inches="tight",
    )
    fig.savefig(
        figure_dir / f"{fileprefix}.pdf",
        dpi=360,
        bbox_inches="tight",
    )
    plt.close()


def analyse_cage(
    database_path: pathlib.Path,
    name: str,
    forcefield: cgx.forcefields.ForceField,
    num_building_blocks: int,
    rescan: bool,  # noqa: FBT001
) -> None:
    """Analyse a toy model cage."""
    database = cgx.utilities.AtomliteDatabase(database_path)
    properties = database.get_entry(key=name).properties
    final_molecule = database.get_molecule(name)

    if "converging_binder_binder_angles" not in properties or rescan:
        _, tstr, _, _, attempt = name.split("_")

        database.add_properties(
            key=name,
            property_dict={
                "forcefield_dict": forcefield.get_forcefield_dictionary(),
                "energy_per_bb": cgx.utilities.get_energy_per_bb(
                    energy_decomposition=properties["energy_decomposition"],
                    number_building_blocks=num_building_blocks,
                ),
                "tstr": tstr,
                "attempt": attempt,
            },
        )

        g_measure = cgx.analysis.GeomMeasure.from_forcefield(forcefield)
        bond_data = g_measure.calculate_bonds(final_molecule)
        bond_data = {"_".join(i): bond_data[i] for i in bond_data}
        angle_data = g_measure.calculate_angles(final_molecule)
        angle_data = {"_".join(i): angle_data[i] for i in angle_data}
        dihedral_data = g_measure.calculate_torsions(
            molecule=final_molecule,
            absolute=False,
            as_search_string=True,
        )

        database.add_properties(
            key=name,
            property_dict={
                "bond_data": bond_data,
                "angle_data": angle_data,
                "dihedral_data": dihedral_data,
            },
        )

        ii_dists = (
            stko.molecule_analysis.GeometryAnalyser().get_metal_distances(
                molecule=final_molecule,
                metal_atom_nos=(77,),
            )
        )
        if "s" not in tstr:
            min_ii_value = -1.0
            max_ii_value = -1.0
        else:
            min_ii_value = min(ii_dists.values())
            max_ii_value = max(ii_dists.values())

        ss_dists = (
            stko.molecule_analysis.GeometryAnalyser().get_metal_distances(
                molecule=final_molecule,
                metal_atom_nos=(16,),
            )
        )
        if "i" not in tstr:
            min_ss_value = -1.0
            max_ss_value = -1.0
        else:
            min_ss_value = min(ss_dists.values())
            max_ss_value = max(ss_dists.values())

        database.add_properties(
            key=name,
            property_dict={
                "min_ii_dist": min_ii_value,
                "max_ii_dist": max_ii_value,
                "min_ss_dist": min_ss_value,
                "max_ss_dist": max_ss_value,
            },
        )

        # Get the bg angles.
        ligands = stko.molecule_analysis.DecomposeMOC().decompose(
            molecule=final_molecule,
            metal_atom_nos=(46,),
        )

        c_binder_binder_angles = []
        d_binder_binder_angles = []
        for lig in ligands:
            if lig.get_num_atoms() in (8, 9, 10):
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


def make_plot(
    database_path: pathlib.Path,
    figure_dir: pathlib.Path,
    filename: str,
) -> None:
    """Visualise energies."""
    fig, ax = plt.subplots(figsize=(8, 5))

    datas: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for entry in cgx.utilities.AtomliteDatabase(database_path).get_entries():
        tstr = entry.properties["tstr"]

        if tstr[-1] == "s":
            x = entry.properties["forcefield_dict"]["v_dict"]["i"]

        elif tstr[-1] == "i":
            x = entry.properties["forcefield_dict"]["v_dict"]["s"]

        else:
            x = -0.02

        y = entry.properties["energy_per_bb"]

        datas[tstr][x].append(y)

    for tstr, col in cmap.items():
        ax.plot(
            list(datas[tstr]),
            [min(datas[tstr][i]) for i in datas[tstr]],
            alpha=1.0,
            marker="o",
            markerfacecolor=col,
            mec="k",
            markersize=10,
            ls=ls[tstr],
            label=dmap[tstr][1],
            c="k" if "s" in tstr else "w",
        )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel(r"$\sigma_{s}$  [$\mathrm{\AA}$]", fontsize=16)
    ax.set_ylabel(eb_str(), fontsize=16)

    ax.legend(ncol=4, fontsize=16)
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


def make_distinct_plot(
    database_path: pathlib.Path,
    figure_dir: pathlib.Path,
    filename: str,
) -> None:
    """Visualise energies."""
    steric_dict = get_steric_measures(figure_dir=figure_dir / "sterimol")

    fig, ax = plt.subplots(
        ncols=1,
        sharey=True,
        sharex=True,
        figsize=(8, 5),
    )

    datas: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for entry in cgx.utilities.AtomliteDatabase(database_path).get_entries():
        tstr = entry.properties["tstr"]

        if tstr[-1] == "s":
            x = entry.properties["forcefield_dict"]["v_dict"]["i"]

        elif tstr[-1] == "i":
            x = entry.properties["forcefield_dict"]["v_dict"]["s"]

        else:
            x = -0.02

        y = entry.properties["energy_per_bb"]

        datas[tstr][x].append(y)

    for tstr, (col, lbl) in dmap.items():
        if tstr in ("3P6", "4P8", "4P82"):
            continue
        if tstr in ("3P6s", "4P8s", "4P82s", "6P12s"):
            continue

        if tstr in ("3P6si", "4P8si", "4P82si", "6P12si"):
            ax.plot(
                list(datas[tstr]),
                [min(datas[tstr][i]) for i in datas[tstr]],
                alpha=1.0,
                marker="o",
                markerfacecolor=col,
                mec="k",
                markersize=10,
                ls="-",
                label=lbl,
                c="k" if "s" in tstr else "w",
            )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel(r"$\sigma_{s}$  [$\mathrm{\AA}$]", fontsize=16)
    ax.set_ylabel(eb_str(), fontsize=16)
    ax.set_yscale("log")
    cg_scale = 2
    for mname, smol in steric_dict.items():
        ax.axvline(x=smol.L_value / cg_scale, c="k")
        ax.text(x=smol.L_value / cg_scale, y=0.1, s=mname, fontsize=16)

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


def ii_plot(  # noqa: C901, PLR0912
    database_path: pathlib.Path,
    figure_dir: pathlib.Path,
    filename: str,
    target: str,
) -> None:
    """Visualise energies."""
    fig, ax = plt.subplots(figsize=(8, 5))

    datas: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(tuple)
    )
    for entry in cgx.utilities.AtomliteDatabase(database_path).get_entries():
        tstr = entry.properties["tstr"]

        if tstr[-1] == "s":
            x = entry.properties["forcefield_dict"]["v_dict"]["i"]
        elif tstr[-1] == "i":
            x = entry.properties["forcefield_dict"]["v_dict"]["s"]
        else:
            continue

        if target == "i":
            if "min_ii_dist" not in entry.properties:
                continue
            y = entry.properties["min_ii_dist"]

        elif target == "s":
            if "min_ss_dist" not in entry.properties:
                continue
            y = entry.properties["min_ss_dist"]

        try:
            if entry.properties["energy_per_bb"] < datas[tstr][x][1]:
                datas[tstr][x] = (y, entry.properties["energy_per_bb"])
        except IndexError:
            datas[tstr][x] = (y, entry.properties["energy_per_bb"])

    if target == "i":
        xlbl = r"min. $r_{i-i}$ [$\AA$]"
        str_skip = "s"

    elif target == "s":
        xlbl = r"min. $r_{s-s}$ [$\AA$]"
        str_skip = "si"

    for tstr in cmap:
        if str_skip not in tstr:
            continue

        ax.plot(
            list(datas[tstr]),
            [datas[tstr][i][0] for i in datas[tstr]],
            alpha=1.0,
            marker="o",
            markerfacecolor=dmap[tstr][0],
            mec="k",
            markersize=10,
            ls=ls[tstr],
            label=dmap[tstr][1],
            c="k",
        )
    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel(r"$\sigma_{s}$  [$\AA$]", fontsize=16)
    ax.set_ylabel(xlbl, fontsize=16)
    ax.legend(ncol=1, fontsize=16)
    ax.set_ylim(0, None)

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


def binder_vector_angles_plot(
    database_path: pathlib.Path,
    figure_dir: pathlib.Path,
    filename: str,
) -> None:
    """Visualise energies."""
    fig, (ax, ax1) = plt.subplots(
        ncols=2,
        sharey=True,
        sharex=True,
        figsize=(16, 5),
    )

    datas_l7: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(tuple)
    )
    datas_l1: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(tuple)
    )
    for entry in cgx.utilities.AtomliteDatabase(database_path).get_entries():
        tstr = entry.properties["tstr"]

        if tstr[-1] == "i":
            x = entry.properties["forcefield_dict"]["v_dict"]["s"]

        else:
            continue

        yl7 = entry.properties["converging_binder_binder_angles"]
        yl1 = entry.properties["diverging_binder_binder_angles"]

        try:
            if entry.properties["energy_per_bb"] < datas_l7[tstr][x][1]:
                datas_l7[tstr][x] = (yl7, entry.properties["energy_per_bb"])
        except IndexError:
            datas_l7[tstr][x] = (yl7, entry.properties["energy_per_bb"])

        try:
            if entry.properties["energy_per_bb"] < datas_l1[tstr][x][1]:
                datas_l1[tstr][x] = (yl1, entry.properties["energy_per_bb"])
        except IndexError:
            datas_l1[tstr][x] = (yl1, entry.properties["energy_per_bb"])

    for tstr in cmap:
        if tstr not in datas_l1:
            continue

        if tstr not in ("6P12", "6P12s", "6P12si"):
            ax.plot(
                list(datas_l1[tstr]),
                [np.mean(datas_l1[tstr][i][0]) for i in datas_l1[tstr]],
                alpha=1.0,
                marker="o",
                markerfacecolor=dmap[tstr][0],
                mec="k",
                markersize=10,
                ls=ls[tstr],
                c="k",
                zorder=2,
            )
            ax.fill_between(
                list(datas_l1[tstr]),
                y1=[np.min(datas_l1[tstr][i][0]) for i in datas_l1[tstr]],
                y2=[np.max(datas_l1[tstr][i][0]) for i in datas_l1[tstr]],
                alpha=0.6,
                color=dmap[tstr][0],
                edgecolor=(0, 0, 0, 2.0),
                lw=0,
            )
        ax1.plot(
            list(datas_l7[tstr]),
            [np.mean(datas_l7[tstr][i][0]) for i in datas_l7[tstr]],
            alpha=1.0,
            marker="o",
            markerfacecolor=dmap[tstr][0],
            mec="k",
            markersize=10,
            ls=ls[tstr],
            c="k",
            zorder=2,
            label=dmap[tstr][1],
        )
        ax1.fill_between(
            list(datas_l7[tstr]),
            y1=[np.min(datas_l7[tstr][i][0]) for i in datas_l7[tstr]],
            y2=[np.max(datas_l7[tstr][i][0]) for i in datas_l7[tstr]],
            alpha=0.6,
            color=dmap[tstr][0],
            edgecolor=(0, 0, 0, 2.0),
            lw=0,
        )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel(r"$\sigma_{s}$  [$\AA$]", fontsize=16)
    ax.set_ylabel("mean rigid binder angle [$^\\circ$]", fontsize=16)
    ax.set_ylim(0, None)

    ax1.tick_params(axis="both", which="major", labelsize=16)
    ax1.set_xlabel(r"$\sigma_{s}$  [$\AA$]", fontsize=16)
    ax1.set_ylabel("mean twistable binder angle [$^\\circ$]", fontsize=16)
    ax1.legend(ncol=1, fontsize=16)

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


def main() -> None:  # noqa: PLR0915, C901, PLR0912
    """Run script."""
    args = _parse_args()

    wd = pathlib.Path("/home/atarzia/workingspace/starships/")
    calculation_dir = wd / "tsteric_calculations"
    calculation_dir.mkdir(exist_ok=True)
    structure_dir = wd / "tsteric_structures"
    structure_dir.mkdir(exist_ok=True)
    ligand_dir = wd / "tsteric_ligands"
    ligand_dir.mkdir(exist_ok=True)
    data_dir = wd / "tsteric_data"
    data_dir.mkdir(exist_ok=True)
    figure_dir = wd / "figures"
    figure_dir.mkdir(exist_ok=True)

    database_path = data_dir / "tsteric.db"

    pair = "la_st5"
    convergings = cgx.molecular.StericSixBead(
        bead=cbead_c,
        abead1=abead_c,
        abead2=ebead_c,
        ibead=inner_bead,
    )
    convergingsi = cgx.molecular.StericSevenBead(
        bead=cbead_c,
        abead1=abead_c,
        abead2=ebead_c,
        ibead=inner_bead,
        sbead=steric_bead,
    )
    converging = cgx.molecular.SixBead(
        bead=cbead_c,
        abead1=abead_c,
        abead2=ebead_c,
    )
    converging_name = "la"
    diverging_name = "st5"
    diverging = cgx.molecular.TwoC1Arm(bead=cbead_d, abead1=abead_d)
    tetra = cgx.molecular.FourC1Arm(bead=tetra_bead, abead1=binder_bead)

    new_definer_dict = {
        # Bonds.
        "mb": ("bond", 1.0, 1e5),
        "is": ("bond", 1.0, 1e5),
        # Angles.
        "bmb": ("pyramid", 90, 1e2),
        "mba": ("angle", 180, 1e2),
        "mbg": ("angle", 180, 1e2),
        "aca": ("angle", 180, 1e2),
        "egb": ("angle", 120, 1e2),
        "deg": ("angle", 180, 1e2),
        # Torsions.
        # Nonbondeds.
        "m": ("nb", 10.0, 1.0),
        "d": ("nb", 10.0, 1.0),
        "e": ("nb", 10.0, 1.0),
        "a": ("nb", 10.0, 1.0),
        "b": ("nb", 10.0, 1.0),
        "c": ("nb", 10.0, 1.0),
        "g": ("nb", 10.0, 1.0),
        "i": ("nb", 10.0, 1.0),
        "s": ("nb", 10.0, 1.0),
    }

    topologies = (
        ("3P6", stk.cage.M3L6),
        ("4P8", cgx.topologies.CGM4L8),
        ("4P82", cgx.topologies.M4L82),
        ("6P12", stk.cage.M6L12Cube),
        # ("3P6s", stk.cage.M3L6),  # noqa: ERA001
        # ("4P8s", cgx.topologies.CGM4L8),  # noqa: ERA001
        # ("4P82s", cgx.topologies.M4L82),  # noqa: ERA001
        # ("6P12s", stk.cage.M6L12Cube),  # noqa: ERA001
        ("6P12si", stk.cage.M6L12Cube),
        ("3P6si", stk.cage.M3L6),
        ("4P82si", cgx.topologies.M4L82),
        ("4P8si", cgx.topologies.CGM4L8),
    )
    st5_values = {"ba": 2.8, "aa": 5.0, "bac": 120, "bacab": 180}
    la_values = {
        "dd": 7.0,
        "de": 1.5,
        "ide": 170,
        "eg": 1.4,
        "gb": 1.4,
        "dde": 170,
    }

    if args.run:
        for tstr, tfunction in topologies:
            if tstr[-1] == "s":
                e_range = [10]
                s_range = np.linspace(0.0, 4, 10)
                cc = convergings

            elif tstr[-2:] == "si":
                e_range = [10]
                s_range = np.linspace(0.0, 4, 10)
                cc = convergingsi

            else:
                e_range = [10]
                s_range = [0.0]
                cc = converging

            for (i, vdw_e_value), (j, vdw_s_value) in it.product(
                enumerate(e_range), enumerate(s_range)
            ):
                if tstr[-1] == "s":
                    ligand_measures = {
                        "la": {
                            **la_values,
                            "ivdw_e": vdw_e_value,
                            "ivdw_s": vdw_s_value,
                        },
                        "st5": st5_values,
                    }

                elif tstr[-2:] == "si":
                    ligand_measures = {
                        "la": {
                            **la_values,
                            "svdw_e": vdw_e_value,
                            "svdw_s": vdw_s_value,
                        },
                        "st5": st5_values,
                    }

                else:
                    ligand_measures = {"la": la_values, "st5": st5_values}

                forcefield = precursors_to_forcefield(
                    pair=pair,
                    diverging=diverging,
                    converging=cc,
                    conv_meas=ligand_measures["la"],
                    dive_meas=ligand_measures["st5"],
                    new_definer_dict=new_definer_dict,
                    vdw_bond_cutoff=3,
                )

                converging_name = (
                    f"{cc.get_name()}_f{forcefield.get_identifier()}"
                )
                converging_bb = cgx.utilities.optimise_ligand(
                    molecule=cc.get_building_block(),
                    name=converging_name,
                    output_dir=calculation_dir,
                    forcefield=forcefield,
                    platform=None,
                )
                converging_bb.write(
                    str(ligand_dir / f"{converging_name}_optl.mol")
                )
                converging_bb = converging_bb.clone()

                tetra_name = (
                    f"{tetra.get_name()}_f{forcefield.get_identifier()}"
                )
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
                diverging_bb.write(
                    str(ligand_dir / f"{diverging_name}_optl.mol")
                )
                diverging_bb = diverging_bb.clone()

                for attempt, scale in enumerate(
                    (
                        "from_0",
                        "spring",
                        "kamada",
                        1.1,
                        1.0,
                        0.9,
                        0.8,
                        0.7,
                        0.6,
                        0.5,
                    )
                ):
                    actual_scale = 1 if not isinstance(scale, float) else scale

                    name = f"ts_{tstr}_{i}_{j}_{attempt}"
                    # Some skips.
                    if attempt in (3, 4, 5, 6, 7, 8) and tstr in (
                        "3P6s",
                        "3P6si",
                        "4P8s",
                        "4P8si",
                        "4P82s",
                        "4P82si",
                        "6P12s",
                        "6P12si",
                    ):
                        logging.info("skipping %s", name)
                        continue

                    logging.info("building %s", name)

                    if tstr in ("3P6", "3P6s", "3P6si"):
                        cage = stk.ConstructedMolecule(
                            tfunction(
                                building_blocks={
                                    tetra_bb: (0, 1, 2),
                                    converging_bb: (3, 4, 5, 6),
                                    diverging_bb: (7, 8),
                                },
                                vertex_positions=None,
                                scale_multiplier=actual_scale,
                            )
                        )
                        num_bbs = 9

                    elif tstr in ("4P8", "4P8s", "4P8si"):
                        cage = stk.ConstructedMolecule(
                            tfunction(
                                building_blocks={
                                    tetra_bb: (0, 1, 2, 3),
                                    converging_bb: (4, 6, 8, 10),
                                    diverging_bb: (5, 7, 9, 11),
                                },
                                vertex_positions=None,
                                scale_multiplier=actual_scale,
                            )
                        )
                        num_bbs = 12

                    elif tstr in ("4P82", "4P82s", "4P82si"):
                        cage = stk.ConstructedMolecule(
                            tfunction(
                                building_blocks={
                                    tetra_bb: (0, 1, 2, 3),
                                    converging_bb: (5, 6, 7, 8),
                                    diverging_bb: (4, 9, 10, 11),
                                },
                                vertex_positions=None,
                                scale_multiplier=actual_scale,
                            )
                        )
                        num_bbs = 12

                    elif tstr in ("6P12", "6P12s", "6P12si"):
                        cage = stk.ConstructedMolecule(
                            tfunction(
                                building_blocks=(tetra_bb, converging_bb),
                                vertex_positions=None,
                                scale_multiplier=actual_scale,
                            )
                        )
                        num_bbs = 18

                    else:
                        raise NotImplementedError

                    if scale == "from_0":
                        if "s" not in tstr:
                            continue
                        cage = get_from_0_coordinares(
                            molecule=cage,
                            tstr=tstr,
                            structure_dir=structure_dir,
                        )

                    if scale == "spring":
                        stko_graph = stko.Network.init_from_molecule(cage)
                        nx_positions = nx.spring_layout(
                            stko_graph.get_graph(), dim=3
                        )
                        cage = convert_coordinates(cage, nx_positions)

                    if scale == "kamada":
                        stko_graph = stko.Network.init_from_molecule(cage)
                        nx_positions = nx.kamada_kawai_layout(
                            stko_graph.get_graph(), dim=3
                        )
                        cage = convert_coordinates(cage, nx_positions)

                    cage.write(structure_dir / f"{name}_unopt.mol")

                    potential_names = []
                    for num in range(10):
                        potential_names.extend(
                            [
                                f"ts_{tstr}_{i - 3}_{j - 3}_{num}",
                                f"ts_{tstr}_{i - 3}_{j}_{num}",
                                f"ts_{tstr}_{i}_{j - 3}_{num}",
                                f"ts_{tstr}_{i - 2}_{j - 2}_{num}",
                                f"ts_{tstr}_{i - 2}_{j}_{num}",
                                f"ts_{tstr}_{i}_{j - 2}_{num}",
                                f"ts_{tstr}_{i - 1}_{j - 1}_{num}",
                                f"ts_{tstr}_{i - 1}_{j}_{num}",
                                f"ts_{tstr}_{i}_{j - 1}_{num}",
                                f"ts_{tstr}_{i}_{j}_{num}",
                            ]
                        )

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
                        conformer.molecule.with_centroid((0, 0, 0)).write(
                            str(structure_dir / f"{name}_optc.mol")
                        )

                    analyse_cage(
                        database_path=database_path,
                        name=name,
                        forcefield=forcefield,
                        num_building_blocks=num_bbs,
                        rescan=False,
                    )

            if tstr in ("3P6", "4P8", "4P82", "6P12"):
                continue
            # Rescan over the surface for improved energies.
            for (i, vdw_e_value), (j, vdw_s_value) in it.product(
                enumerate(e_range), enumerate(s_range)
            ):
                if tstr[-1] == "s":
                    ligand_measures = {
                        "la": {
                            **la_values,
                            "ivdw_e": vdw_e_value,
                            "ivdw_s": vdw_s_value,
                        },
                        "st5": st5_values,
                    }

                elif tstr[-2:] == "si":
                    ligand_measures = {
                        "la": {
                            **la_values,
                            "svdw_e": vdw_e_value,
                            "svdw_s": vdw_s_value,
                        },
                        "st5": st5_values,
                    }

                else:
                    ligand_measures = {"la": la_values, "st5": st5_values}

                forcefield = precursors_to_forcefield(
                    pair=pair,
                    diverging=diverging,
                    converging=cc,
                    conv_meas=ligand_measures["la"],
                    dive_meas=ligand_measures["st5"],
                    new_definer_dict=new_definer_dict,
                    vdw_bond_cutoff=3,
                )

                name = f"ts_{tstr}_{i}_{j}_{attempt}"
                logging.info("rescanning %s", name)

                current_cage = stk.BuildingBlock.init_from_file(
                    structure_dir / f"{name}_optc.mol"
                )

                if tstr in ("3P6s", "3P6si"):
                    num_bbs = 9
                elif tstr in ("4P8s", "4P8si", "4P82s", "4P82si"):
                    num_bbs = 12
                elif tstr in ("6P12s", "6P12si"):
                    num_bbs = 18
                else:
                    raise NotImplementedError

                potential_names = []
                for ix, jx in it.product(
                    range(len(e_range)), range(len(s_range))
                ):
                    potential_names.append(f"ts_{tstr}_{ix}_{jx}_{attempt}")

                conformer = cgx.scram.optimise_from_files(
                    molecule=current_cage,
                    name=name,
                    output_dir=calculation_dir,
                    forcefield=forcefield,
                    platform=None,
                    database_path=database_path,
                    potential_names=potential_names,
                )

                conformer.molecule.with_centroid((0, 0, 0)).write(
                    str(structure_dir / f"{name}_optc.mol")
                )

                analyse_cage(
                    database_path=database_path,
                    name=name,
                    forcefield=forcefield,
                    num_building_blocks=num_bbs,
                    rescan=True,
                )

    make_distinct_plot(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="sterics_1d.png",
    )
    make_plot(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="sterics_1.png",
    )

    ii_plot(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="sterics_2.png",
        target="i",
    )
    ii_plot(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="sterics_4.png",
        target="s",
    )

    geom_distributions(
        database_path=database_path,
        figure_dir=figure_dir,
        fileprefix="sterics_3",
    )

    binder_vector_angles_plot(
        database_path=database_path,
        figure_dir=figure_dir,
        filename="sterics_5.png",
    )


if __name__ == "__main__":
    main()
