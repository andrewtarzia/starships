"""Script to generate and optimise CG models."""

import logging
import pathlib
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import stk
import stko
from rdkit import RDLogger

from cage_construct._internal.scripts.utilities import name_parser

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
RDLogger.DisableLog("rdApp.*")


def main() -> None:
    """Run script."""
    wd = pathlib.Path("/home/atarzia/workingspace/starships/")
    structure_dir = wd / "experimental_aa_structures"
    figure_dir = wd / "figures"
    figure_dir.mkdir(exist_ok=True)

    files = {
        "xray_l1b_l2": structure_dir / "eb712bb_sq_2_res_withh.mol",
        "dft_l1_l7": structure_dir / "L7ST5_Td_conv.mol",
        "dft_l1_l2": structure_dir / "eb712bb_sq_2_res_withh.mol",
        "xray_l7": structure_dir / "Pd_6L7_12_cleaned.mol",
    }

    lmaps = {
        40: "la",
        42: "c1",
        50: "las",
        44: "st5",
    }

    manual_measures = {
        "xray_l1b_l2": {
            40: {
                "dde": [171.8, 165.1, 168.3, 169.4, 167.8, 166.5, 168.6, 169.8]
            }
        },
        "dft_l1_l7": {50: {"dde": []}},
    }
    logging.info(
        "%s: avg. %s dde: %s",
        "xray_l1b_l2",
        name_parser(lmaps[40]),
        np.mean([manual_measures["xray_l1b_l2"][40]["dde"]]),
    )

    datas = {}
    for sname, sfile in files.items():
        if sname == "dft_l1_l2":
            logging.warning("skips")
            continue
        molecule = stk.BuildingBlock.init_from_file(sfile)

        ligands = stko.molecule_analysis.DecomposeMOC().decompose(
            molecule=molecule,
            metal_atom_nos=(46,),
        )

        ligand_dict = defaultdict(dict)
        for id_, lig in enumerate(ligands):
            calc = stko.molecule_analysis.DitopicThreeSiteAnalyser()
            # Define as building block with
            as_building_block = stk.BuildingBlock.init_from_molecule(
                lig,
                stko.functional_groups.ThreeSiteFactory(
                    smarts="[#6]~[#7X2]~[#6]"
                ),
            )
            ldata = {
                "binder_angles": calc.get_binder_angles(as_building_block),
                "binder_distance": calc.get_binder_distance(as_building_block),
                "binder_adjacent_torsion": calc.get_binder_adjacent_torsion(
                    as_building_block
                ),
            }
            ligand_dict[lig.get_num_atoms()][id_] = ldata
        datas[sname] = ligand_dict

        logging.info(
            "%s has %s ligands with sizes %s (%s)",
            sname,
            [len(i) for i in ligand_dict.values()],
            list(ligand_dict.keys()),
            [name_parser(lmaps[i]) for i in ligand_dict],
        )

    fig, ax = plt.subplots(figsize=(8, 5))
    for sname, ligand_dict in datas.items():
        for num, ndict in ligand_dict.items():
            logging.info(
                "%s: avg. %s binder-angle: %s",
                sname,
                name_parser(lmaps[num]),
                np.mean(
                    [ndict[i]["binder_angles"][0] for i in ndict]
                    + [ndict[i]["binder_angles"][1] for i in ndict]
                ),
            )

            xs = [ndict[i]["binder_angles"][0] for i in ndict] + [
                ndict[i]["binder_angles"][1] for i in ndict
            ]
            ys = [ndict[i]["binder_distance"] for i in ndict] + [
                ndict[i]["binder_distance"] for i in ndict
            ]

            ax.scatter(
                xs,
                ys,
                edgecolor="k",
                s=40,
                label=f"{sname}_{num}",
            )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel("binder angle [deg]", fontsize=16)
    ax.set_ylabel("N-N distance [AA]", fontsize=16)
    ax.legend(fontsize=16)

    fig.tight_layout()
    fig.savefig(figure_dir / "aa_cf_1.png", dpi=360, bbox_inches="tight")
    fig.savefig(figure_dir / "aa_cf_1.pdf", dpi=360, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
