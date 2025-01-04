"""Utilities for complex processing."""

import bbprep
import stk


def ensure_nccnbr_torsion(
    building_block: stk.BuildingBlock,
) -> stk.BuildingBlock:
    """Ensure the orientation."""
    ensemble = bbprep.generators.ETKDG(num_confs=100).generate_conformers(
        building_block
    )

    # Select the C-C-N-Br torsion.
    process1 = bbprep.TargetTorsion(
        ensemble=ensemble,
        selector=bbprep.selectors.BySmartsSelector(
            smarts="[#6]~[#6]~[#7]~[#35]",
            selected_indices=(0, 1, 2, 3),
        ),
        target_value=180,
    )
    p1_by_id = process1.get_all_scores_by_id()

    # Select the N-C-C-N torsion.
    process2 = bbprep.TargetTorsion(
        ensemble=ensemble,
        selector=bbprep.selectors.BySmartsSelector(
            smarts="[#7]~[#6]~[#6]~[#7]",
            selected_indices=(0, 1, 2, 3),
        ),
        target_value=0,
    )
    p2_by_id = process2.get_all_scores_by_id()

    best_score = float("inf")
    best_conformer = bbprep.Conformer(
        molecule=ensemble.get_base_molecule().clone(),
        conformer_id=-1,
        source=None,
        permutation=None,
    )
    for conformer in ensemble.yield_conformers():
        p1score = p1_by_id[conformer.conformer_id]
        p2score = p2_by_id[conformer.conformer_id]
        sum_score = p1score + p2score
        if sum_score < best_score:
            best_conformer = bbprep.Conformer(
                molecule=conformer.molecule.clone(),
                conformer_id=conformer.conformer_id,
                source=conformer.source,
                permutation=conformer.permutation,
            )
            best_score = sum_score

    # Get the best conformer as an stk.BuildingBlock.
    return best_conformer.molecule
