"""Utilities for ligand processing."""

import bbprep
import stk
import stko


def get_lowest_energy_conformer(
    building_block: stk.BuildingBlock,
) -> tuple[stk.BuildingBlock, float]:
    """Get the lowest energy conformer.

    Directly from bbprep recipes.
    """
    # This uses the rdkit conformer generation and calculations.
    calculator = bbprep.EnergyCalculator(
        name="MMFFEnergy",
        function=stko.MMFFEnergy().get_energy,
    )

    optimiser = bbprep.Optimiser(
        name="MMFF",
        function=stko.MMFF().optimize,
    )

    ensemble = bbprep.generators.ETKDG(num_confs=100).generate_conformers(
        building_block
    )
    # Optimise ensemble.
    opt_ensemble = ensemble.optimise_conformers(optimiser=optimiser)

    # Get lowest energy conformer.
    lowest_energy_conformer = opt_ensemble.get_lowest_energy_conformer(
        calculator=calculator
    )

    return lowest_energy_conformer.molecule, calculator.function(
        lowest_energy_conformer.molecule
    )
