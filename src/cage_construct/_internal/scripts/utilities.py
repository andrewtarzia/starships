"""Utilities module."""

import itertools
import logging
from collections import abc
from copy import deepcopy

import cgexplore as cgx
import numpy as np
import stk

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def simple_beeswarm2(
    y: np.ndarray | list,
    nbins: int | None = None,
    width: float = 1.0,
) -> np.ndarray:
    """Returns beeswarm for y."""
    # Convert y to a numpy array to ensure it is compatible with numpy
    # functions
    y = np.asarray(y)

    # If nbins is not provided, calculate a suitable number of bins based on
    # data length
    if nbins is None:
        nbins = np.ceil(len(y) / 6).astype(int)

    # Get the histogram of y and the corresponding bin edges
    nn, ybins = np.histogram(y, bins=nbins)

    # Find the maximum count in any bin to be used in calculating the x
    # positions
    nmax = nn.max()

    # Create an array of zeros with the same length as y, to store x-
    # coordinates
    x = np.zeros(len(y))

    # Divide indices of y-values into corresponding bins
    ibs = []
    for ymin, ymax in itertools.pairwise(ybins):
        # Find the indices where y falls within the current bin
        i = np.nonzero((y > ymin) * (y <= ymax))[0]
        ibs.append(i)

    # Assign x-coordinates to the points in each bin
    dx = width / (nmax // 2)

    for i in ibs:
        yy = y[i]
        if len(i) > 1:
            # Determine the starting index (j) based on the number of elements
            # in the bin
            j = len(i) % 2

            # Sort the indices based on their corresponding y-values
            i = i[np.argsort(yy)]  # noqa: PLW2901

            # Separate the indices into two halves (a and b) for arranging the
            # spoints
            a = i[j::2]
            b = i[j + 1 :: 2]

            # Assign x-coordinates to points in each half of the bin
            x[a] = (0.5 + j / 3 + np.arange(len(b))) * dx
            x[b] = (0.5 + j / 3 + np.arange(len(b))) * -dx

    return x


def name_parser(name: str) -> str:
    """Convert computational names to paper names."""
    dictionary = {
        "st5": "L1",
        "c1": "L1b",
        "la": "L2",
        "las": "L7",
    }
    return dictionary[name]


class Scrambler:
    """Iterate over topology graphs.

    This is an old version of this code, which I do not recommend using over
    the `TopologyIterator`.

    TODO: Clean-up and remove this class.

    """

    def __init__(
        self,
        tetra_bb: stk.BuildingBlock,
        converging_bb: stk.BuildingBlock,
        diverging_bb: stk.BuildingBlock,
        multiplier: int,
        stoichiometry: tuple[int, int, int],
    ) -> None:
        """Initialize."""
        logging.warning(
            "This is a hard-coded, old solution. Look for new solutions in "
            "cgx.scram."
        )
        self._building_blocks: dict[stk.BuildingBlock, abc.Sequence[int]]
        self._underlying_topology: type[stk.cage.Cage]

        if stoichiometry == (1, 1, 1):
            if multiplier == 1:
                self._building_blocks = {
                    tetra_bb: (0,),
                    converging_bb: (1,),
                    diverging_bb: (2,),
                }
                self._underlying_topology = cgx.topologies.UnalignedM1L2
                self._scale_multiplier = 2
                self._skip_initial = True

            elif multiplier == 2:  # noqa: PLR2004
                self._building_blocks = {
                    tetra_bb: (0, 1),
                    converging_bb: (2, 3),
                    diverging_bb: (4, 5),
                }
                self._underlying_topology = stk.cage.M2L4Lantern
                self._scale_multiplier = 2
                self._skip_initial = False

            elif multiplier == 3:  # noqa: PLR2004
                self._building_blocks = {
                    tetra_bb: (0, 1, 2),
                    converging_bb: (3, 4, 5),
                    diverging_bb: (6, 7, 8),
                }
                self._underlying_topology = stk.cage.M3L6
                self._scale_multiplier = 2
                self._skip_initial = False

            elif multiplier == 4:  # noqa: PLR2004
                self._building_blocks = {
                    tetra_bb: (0, 1, 2, 3),
                    converging_bb: (4, 5, 6, 7),
                    diverging_bb: (8, 9, 10, 11),
                }
                self._underlying_topology = cgx.topologies.CGM4L8
                self._scale_multiplier = 2
                self._skip_initial = False

        if stoichiometry == (4, 2, 3):
            if multiplier == 1:
                self._building_blocks = {
                    tetra_bb: (0, 1, 2),
                    converging_bb: (3, 4, 5, 6),
                    diverging_bb: (7, 8),
                }
                self._underlying_topology = stk.cage.M3L6
                self._scale_multiplier = 2
                self._skip_initial = False

            elif multiplier == 2:  # noqa: PLR2004
                self._building_blocks = {
                    tetra_bb: (0, 1, 2, 3, 4, 5),
                    converging_bb: (6, 7, 8, 9, 10, 11, 12, 13),
                    diverging_bb: (14, 15, 16, 17),
                }
                self._underlying_topology = stk.cage.M6L12Cube
                self._scale_multiplier = 5
                self._skip_initial = False

            elif multiplier == 4:  # noqa: PLR2004
                self._building_blocks = {
                    tetra_bb: range(12),
                    converging_bb: range(12, 28),
                    diverging_bb: range(28, 36),
                }
                self._underlying_topology = cgx.topologies.CGM12L24
                self._scale_multiplier = 5
                self._skip_initial = False

        self._init_vertex_prototypes = deepcopy(
            self._underlying_topology._vertex_prototypes  # noqa: SLF001
        )
        self._init_edge_prototypes = deepcopy(
            self._underlying_topology._edge_prototypes  # noqa: SLF001
        )
        self._vertices = tuple(
            stk.cage.UnaligningVertex(
                id=i.get_id(),
                position=i.get_position(),
                aligner_edge=i.get_aligner_edge(),
                use_neighbor_placement=i.use_neighbor_placement(),
            )
            for i in self._underlying_topology._vertex_prototypes  # noqa: SLF001
        )
        self._edges = tuple(
            stk.Edge(
                id=i.get_id(),
                vertex1=self._vertices[i.get_vertex1_id()],
                vertex2=self._vertices[i.get_vertex2_id()],
            )
            for i in self._underlying_topology._edge_prototypes  # noqa: SLF001
        )
        self._num_scrambles = 200
        self._num_mashes = 2

        self._define_underlying()
        self._beta = 10

    def _define_underlying(self) -> None:
        self._vertex_connections: dict[int, int] = {}
        for edge in self._init_edge_prototypes:
            if edge.get_vertex1_id() not in self._vertex_connections:
                self._vertex_connections[edge.get_vertex1_id()] = 0
            self._vertex_connections[edge.get_vertex1_id()] += 1

            if edge.get_vertex2_id() not in self._vertex_connections:
                self._vertex_connections[edge.get_vertex2_id()] = 0
            self._vertex_connections[edge.get_vertex2_id()] += 1

        self._type1 = [
            i
            for i in self._vertex_connections
            if self._vertex_connections[i] == 4  # noqa: PLR2004
        ]
        self._type2 = [
            i
            for i in self._vertex_connections
            if self._vertex_connections[i] == 2  # noqa: PLR2004
        ]

        combination = [
            tuple(sorted((i.get_vertex1_id(), i.get_vertex2_id())))
            for i in self._init_edge_prototypes
        ]
        self._initial_topology_code = cgx.scram.TopologyCode(
            vertex_map=combination,
            as_string=cgx.scram.vmap_to_str(combination),
        )

    def get_num_building_blocks(self) -> int:
        """Get number of building blocks."""
        return len(self._init_vertex_prototypes)

    def get_num_scrambles(self) -> int:
        """Get num. scrambles algorithm."""
        return self._num_scrambles

    def get_num_mashes(self) -> int:
        """Get num. mashes algorithm."""
        return self._num_mashes

    def get_constructed_molecules(  # noqa: C901, PLR0912, PLR0915
        self,
    ) -> abc.Generator[cgx.scram.Constructed]:
        """Get constructed molecules from iteration."""
        combinations_tested = set()
        rng = np.random.default_rng(seed=100)
        count = 0

        if not self._skip_initial:
            try:
                constructed = stk.ConstructedMolecule(
                    self._underlying_topology(
                        building_blocks=self._building_blocks,
                        vertex_positions=None,
                    )
                )

                yield cgx.scram.Constructed(
                    constructed_molecule=constructed,
                    idx=0,
                    topology_code=self._initial_topology_code,
                )
            except ValueError:
                pass
            combinations_tested.add(self._initial_topology_code.as_string)

            # Scramble the vertex positions.
            for _ in range(self._num_mashes):
                coordinates = rng.random(size=(len(self._vertices), 3))
                new_vertex_positions = {
                    j: coordinates[j] * 10
                    for j, i in enumerate(self._vertices)
                }

                count += 1
                try:
                    # Try with aligning vertices.
                    constructed = stk.ConstructedMolecule(
                        self._underlying_topology(
                            building_blocks=self._building_blocks,
                            vertex_positions=None,
                        )
                    )
                    yield cgx.scram.Constructed(
                        constructed_molecule=constructed,
                        idx=count,
                        topology_code=self._initial_topology_code,
                    )
                except ValueError:
                    # Try with unaligning.
                    try:
                        constructed = stk.ConstructedMolecule(
                            self._underlying_topology(
                                building_blocks=self._building_blocks,
                                vertex_positions=None,
                            )
                        )
                        yield cgx.scram.Constructed(
                            constructed_molecule=constructed,
                            idx=count,
                            topology_code=self._initial_topology_code,
                        )
                    except ValueError:
                        pass

        for _ in range(self._num_scrambles):
            # Scramble the edges.
            remaining_connections = deepcopy(self._vertex_connections)
            available_type1s = deepcopy(self._type1)
            available_type2s = deepcopy(self._type2)

            new_edges: list[stk.Edge] = []
            combination = []
            for _ in range(len(self._init_edge_prototypes)):
                try:
                    vertex1 = rng.choice(available_type1s)
                    vertex2 = rng.choice(available_type2s)
                except ValueError:
                    if len(remaining_connections) == 1:
                        vertex1 = next(iter(remaining_connections.keys()))
                        vertex2 = next(iter(remaining_connections.keys()))

                new_edge = stk.Edge(
                    id=len(new_edges),
                    vertex1=self._vertices[vertex1],
                    vertex2=self._vertices[vertex2],
                )
                new_edges.append(new_edge)

                remaining_connections[vertex1] += -1
                remaining_connections[vertex2] += -1

                remaining_connections = {
                    i: remaining_connections[i]
                    for i in remaining_connections
                    if remaining_connections[i] != 0
                }

                available_type1s = [
                    i for i in self._type1 if i in remaining_connections
                ]
                available_type2s = [
                    i for i in self._type2 if i in remaining_connections
                ]
                combination.append(tuple(sorted((vertex1, vertex2))))

            topology_code = cgx.scram.TopologyCode(
                vertex_map=combination,
                as_string=cgx.scram.vmap_to_str(combination),
            )

            # If you broke early, do not try to build.
            if len(new_edges) != len(self._edges):
                continue

            if topology_code.as_string in combinations_tested:
                continue

            combinations_tested.add(topology_code.as_string)

            count += 1
            try:
                # Try with aligning vertices.
                constructed = stk.ConstructedMolecule(
                    cgx.topologies.CustomTopology(  # type:ignore[arg-type]
                        building_blocks=self._building_blocks,
                        vertex_prototypes=self._init_vertex_prototypes,
                        edge_prototypes=new_edges,
                        vertex_alignments=None,
                        vertex_positions=None,
                        scale_multiplier=self._scale_multiplier,
                    )
                )
                yield cgx.scram.Constructed(
                    constructed_molecule=constructed,
                    idx=count,
                    topology_code=topology_code,
                )
            except ValueError:
                # Try with unaligning.
                try:
                    constructed = stk.ConstructedMolecule(
                        cgx.topologies.CustomTopology(  # type:ignore[arg-type]
                            building_blocks=self._building_blocks,
                            vertex_prototypes=self._vertices,
                            edge_prototypes=new_edges,
                            vertex_alignments=None,
                            vertex_positions=None,
                            scale_multiplier=self._scale_multiplier,
                        )
                    )
                    yield cgx.scram.Constructed(
                        constructed_molecule=constructed,
                        idx=count,
                        topology_code=topology_code,
                    )
                except ValueError:
                    pass

            # Scramble the vertex positions.
            for _ in range(self._num_mashes):
                coordinates = rng.random(size=(len(self._vertices), 3))
                new_vertex_positions = {
                    j: coordinates[j] * 10
                    for j, i in enumerate(self._vertices)
                }

                count += 1
                try:
                    # Try with aligning vertices.
                    constructed = stk.ConstructedMolecule(
                        cgx.topologies.CustomTopology(  # type:ignore[arg-type]
                            building_blocks=self._building_blocks,
                            vertex_prototypes=self._init_vertex_prototypes,
                            edge_prototypes=new_edges,
                            vertex_alignments=None,
                            vertex_positions=new_vertex_positions,
                            scale_multiplier=self._scale_multiplier,
                        )
                    )
                    yield cgx.scram.Constructed(
                        constructed_molecule=constructed,
                        idx=count,
                        topology_code=topology_code,
                    )
                except ValueError:
                    # Try with unaligning.
                    try:
                        constructed = stk.ConstructedMolecule(
                            cgx.topologies.CustomTopology(  # type:ignore[arg-type]
                                building_blocks=self._building_blocks,
                                vertex_prototypes=self._vertices,
                                edge_prototypes=new_edges,
                                vertex_alignments=None,
                                vertex_positions=new_vertex_positions,
                                scale_multiplier=self._scale_multiplier,
                            )
                        )
                        yield cgx.scram.Constructed(
                            constructed_molecule=constructed,
                            idx=count,
                            topology_code=topology_code,
                        )
                    except ValueError:
                        pass

    def _get_random_topology_code(
        self, generator: np.random.Generator
    ) -> cgx.scram.TopologyCode:
        remaining_connections = deepcopy(self._vertex_connections)
        available_type1s = deepcopy(self._type1)
        available_type2s = deepcopy(self._type2)

        vertex_map = []
        for _ in range(len(self._init_edge_prototypes)):
            try:
                vertex1 = generator.choice(available_type1s)
                vertex2 = generator.choice(available_type2s)
            except ValueError:
                if len(remaining_connections) == 1:
                    vertex1 = next(iter(remaining_connections.keys()))
                    vertex2 = next(iter(remaining_connections.keys()))

            vertex_map.append(tuple(sorted((vertex1, vertex2))))

            remaining_connections[vertex1] += -1
            remaining_connections[vertex2] += -1
            remaining_connections = {
                i: remaining_connections[i]
                for i in remaining_connections
                if remaining_connections[i] != 0
            }
            available_type1s = [
                i for i in self._type1 if i in remaining_connections
            ]
            available_type2s = [
                i for i in self._type2 if i in remaining_connections
            ]

        return cgx.scram.TopologyCode(
            vertex_map=vertex_map, as_string=cgx.scram.vmap_to_str(vertex_map)
        )

    def _shuffle_topology_code(
        self,
        topology_code: cgx.scram.TopologyCode,
        generator: np.random.Generator,
    ) -> cgx.scram.TopologyCode:
        old_vertex_map = topology_code.vertex_map

        size = (
            generator.integers(
                low=1, high=int(len(old_vertex_map) / 2), size=1
            )
            * 2
        )

        swaps = list(
            generator.choice(
                range(len(old_vertex_map)),
                size=int(size[0]),
                replace=False,
            )
        )

        new_vertex_map = []
        already_done = set()
        for vmap_idx in range(len(old_vertex_map)):
            if vmap_idx in already_done:
                continue
            if vmap_idx in swaps:
                possible_ids = [i for i in swaps if i != vmap_idx]
                other_idx = generator.choice(possible_ids, size=1)[0]

                # Swap connections.
                old1 = old_vertex_map[vmap_idx]
                old2 = old_vertex_map[other_idx]

                new1 = (old1[0], old2[1])
                new2 = (old2[0], old1[1])

                new_vertex_map.append(new1)
                new_vertex_map.append(new2)
                swaps = [i for i in swaps if i not in (vmap_idx, other_idx)]

                already_done.add(other_idx)
            else:
                new_vertex_map.append(old_vertex_map[vmap_idx])

        return cgx.scram.TopologyCode(
            vertex_map=new_vertex_map,
            as_string=cgx.scram.vmap_to_str(new_vertex_map),
        )

    def get_topology(
        self,
        input_topology_code: cgx.scram.TopologyCode | None,
        generator: np.random.Generator,
    ) -> cgx.scram.Constructed | None:
        """Get a topology."""
        if input_topology_code is None:
            topology_code = self._get_random_topology_code(generator=generator)
        else:
            topology_code = self._shuffle_topology_code(
                topology_code=input_topology_code,
                generator=generator,
            )

        try:
            # Try with aligning vertices.
            constructed = stk.ConstructedMolecule(
                cgx.topologies.CustomTopology(  # type:ignore[arg-type]
                    building_blocks=self._building_blocks,
                    vertex_prototypes=self._init_vertex_prototypes,
                    edge_prototypes=tuple(
                        stk.Edge(
                            id=i,
                            vertex1=self._init_vertex_prototypes[vmap[0]],
                            vertex2=self._init_vertex_prototypes[vmap[1]],
                        )
                        for i, vmap in enumerate(topology_code.vertex_map)
                    ),
                    vertex_alignments=None,
                    vertex_positions=None,
                    scale_multiplier=self._scale_multiplier,
                )
            )
            return cgx.scram.Constructed(
                constructed_molecule=constructed,
                idx=None,
                topology_code=topology_code,
            )
        except ValueError:
            # Try with unaligning.
            try:
                constructed = stk.ConstructedMolecule(
                    cgx.topologies.CustomTopology(  # type:ignore[arg-type]
                        building_blocks=self._building_blocks,
                        vertex_prototypes=self._vertices,
                        edge_prototypes=tuple(
                            stk.Edge(
                                id=i,
                                vertex1=self._vertices[vmap[0]],
                                vertex2=self._vertices[vmap[1]],
                            )
                            for i, vmap in enumerate(topology_code.vertex_map)
                        ),
                        vertex_alignments=None,
                        vertex_positions=None,
                        scale_multiplier=self._scale_multiplier,
                    )
                )
                return cgx.scram.Constructed(
                    constructed_molecule=constructed,
                    idx=None,
                    topology_code=topology_code,
                )
            except ValueError:
                return None

    def get_mashed_topology(
        self,
        topology_code: cgx.scram.TopologyCode,
        generator: np.random.Generator,
    ) -> cgx.scram.Constructed | None:
        """Get a mashed topology, where vertex coordinates are changed."""
        coordinates = generator.random(size=(len(self._vertices), 3))
        new_vertex_positions = {
            j: coordinates[j] * 10 for j, i in enumerate(self._vertices)
        }

        try:
            # Try with aligning vertices.
            constructed = stk.ConstructedMolecule(
                cgx.topologies.CustomTopology(  # type:ignore[arg-type]
                    building_blocks=self._building_blocks,
                    vertex_prototypes=self._init_vertex_prototypes,
                    edge_prototypes=tuple(
                        stk.Edge(
                            id=i,
                            vertex1=self._init_vertex_prototypes[vmap[0]],
                            vertex2=self._init_vertex_prototypes[vmap[1]],
                        )
                        for i, vmap in enumerate(topology_code.vertex_map)
                    ),
                    vertex_alignments=None,
                    vertex_positions=new_vertex_positions,
                    scale_multiplier=self._scale_multiplier,
                )
            )
            return cgx.scram.Constructed(
                constructed_molecule=constructed,
                idx=None,
                topology_code=topology_code,
            )
        except ValueError:
            # Try with unaligning.
            try:
                constructed = stk.ConstructedMolecule(
                    cgx.topologies.CustomTopology(  # type:ignore[arg-type]
                        building_blocks=self._building_blocks,
                        vertex_prototypes=self._vertices,
                        edge_prototypes=tuple(
                            stk.Edge(
                                id=i,
                                vertex1=self._vertices[vmap[0]],
                                vertex2=self._vertices[vmap[1]],
                            )
                            for i, vmap in enumerate(topology_code.vertex_map)
                        ),
                        vertex_alignments=None,
                        vertex_positions=new_vertex_positions,
                        scale_multiplier=self._scale_multiplier,
                    )
                )
                return cgx.scram.Constructed(
                    constructed_molecule=constructed,
                    idx=None,
                    topology_code=topology_code,
                )
            except ValueError:
                return None


def eb_str(*, no_unit: bool = False) -> str:
    """Get variable string."""
    if no_unit:
        return r"$E_{\mathrm{b}}$"

    return r"$E_{\mathrm{b}}$ [kJmol$^{-1}$]"


def isomer_energy() -> float:
    """Get constant."""
    return 0.3


# Diverging ligands.
cbead_d = cgx.molecular.CgBead(
    element_string="Ag",
    bead_class="c",
    bead_type="c",
    coordination=2,
)
abead_d = cgx.molecular.CgBead(
    element_string="Ba",
    bead_class="a",
    bead_type="a",
    coordination=2,
)
ebead_d = cgx.molecular.CgBead(
    element_string="Mn",
    bead_class="f",
    bead_type="f",
    coordination=2,
)

# Converging ligands.
cbead_c = cgx.molecular.CgBead(
    element_string="Ni",
    bead_class="d",
    bead_type="d",
    coordination=2,
)
abead_c = cgx.molecular.CgBead(
    element_string="Fe",
    bead_class="e",
    bead_type="e",
    coordination=2,
)
ebead_c = cgx.molecular.CgBead(
    element_string="Ga",
    bead_class="g",
    bead_type="g",
    coordination=2,
)

# Constant.
binder_bead = cgx.molecular.CgBead(
    element_string="Pb",
    bead_class="b",
    bead_type="b",
    coordination=2,
)
tetra_bead = cgx.molecular.CgBead(
    element_string="Pd",
    bead_class="m",
    bead_type="m",
    coordination=4,
)
steric_bead = cgx.molecular.CgBead(
    element_string="S",
    bead_class="s",
    bead_type="s",
    coordination=1,
)
inner_bead = cgx.molecular.CgBead(
    element_string="Ir",
    bead_class="i",
    bead_type="i",
    coordination=2,
)

constant_definer_dict = {
    # Bonds.
    "mb": ("bond", 1.0, 1e5),
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
}


def precursors_to_forcefield(  # noqa: C901, PLR0912, PLR0913, PLR0915
    pair: str,
    diverging: cgx.molecular.Precursor,
    converging: cgx.molecular.Precursor,
    conv_meas: dict[str, float],
    dive_meas: dict[str, float],
    new_definer_dict: dict[str, tuple] | None = None,  # type: ignore[type-arg]
    vdw_bond_cutoff: int | None = None,
) -> cgx.forcefields.ForceField:
    """Get a forcefield from precursor definitions."""
    if vdw_bond_cutoff is None:
        vdw_bond_cutoff = 2

    # Define bead libraries.
    present_beads = (
        cbead_d,
        abead_d,
        cbead_c,
        abead_c,
        ebead_c,
        ebead_d,
        binder_bead,
        tetra_bead,
        steric_bead,
        inner_bead,
    )
    cgx.molecular.BeadLibrary(present_beads)

    if new_definer_dict is None:
        definer_dict = deepcopy(constant_definer_dict)
    else:
        definer_dict = deepcopy(new_definer_dict)

    cg_scale = 2

    if isinstance(converging, cgx.molecular.SixBead):
        beads = converging.get_bead_set()
        if "d" not in beads or "e" not in beads or "g" not in beads:
            raise RuntimeError
        if "d" in conv_meas:
            definer_dict["d"] = ("nb", 10.0, conv_meas["d"])
        definer_dict["dd"] = ("bond", conv_meas["dd"] / cg_scale, 1e5)
        definer_dict["de"] = ("bond", conv_meas["de"] / cg_scale, 1e5)
        definer_dict["eg"] = ("bond", conv_meas["eg"] / cg_scale, 1e5)
        definer_dict["gb"] = ("bond", conv_meas["gb"] / cg_scale, 1e5)
        definer_dict["dde"] = ("angle", conv_meas["dde"], 1e2)
        definer_dict["edde"] = ("tors", "0123", 180.0, 50.0, 1)  # type: ignore[assignment]
        definer_dict["mbge"] = ("tors", "0123", 180.0, 50.0, 1)  # type: ignore[assignment]

    elif isinstance(converging, cgx.molecular.StericSixBead):
        beads = converging.get_bead_set()

        if (
            "d" not in beads
            or "e" not in beads
            or "g" not in beads
            or "i" not in beads
        ):
            raise RuntimeError
        definer_dict["di"] = ("bond", conv_meas["dd"] / cg_scale / 2, 1e5)
        definer_dict["de"] = ("bond", conv_meas["de"] / cg_scale, 1e5)
        definer_dict["eg"] = ("bond", conv_meas["eg"] / cg_scale, 1e5)
        definer_dict["gb"] = ("bond", conv_meas["gb"] / cg_scale, 1e5)
        definer_dict["ide"] = ("angle", conv_meas["ide"], 1e2)
        definer_dict["did"] = ("angle", 180, 1e2)
        definer_dict["edide"] = ("tors", "0134", 180, 50, 1)  # type: ignore[assignment]
        definer_dict["mbge"] = ("tors", "0123", 180, 50, 1)  # type: ignore[assignment]
        definer_dict["i"] = ("nb", conv_meas["ivdw_e"], conv_meas["ivdw_s"])

    elif isinstance(converging, cgx.molecular.StericSevenBead):
        beads = converging.get_bead_set()

        if (
            "d" not in beads
            or "e" not in beads
            or "g" not in beads
            or "i" not in beads
            or "s" not in beads
        ):
            raise RuntimeError

        definer_dict["di"] = ("bond", conv_meas["dd"] / cg_scale / 2, 1e5)
        definer_dict["de"] = ("bond", conv_meas["de"] / cg_scale, 1e5)
        definer_dict["eg"] = ("bond", conv_meas["eg"] / cg_scale, 1e5)
        definer_dict["gb"] = ("bond", conv_meas["gb"] / cg_scale, 1e5)
        definer_dict["ide"] = ("angle", conv_meas["ide"], 1e2)
        definer_dict["did"] = ("angle", 180, 1e2)
        definer_dict["dis"] = ("angle", 90, 1e2)
        definer_dict["edide"] = ("tors", "0134", 180, 50, 1)  # type: ignore[assignment]
        definer_dict["mbge"] = ("tors", "0123", 180, 50, 1)  # type: ignore[assignment]
        definer_dict["edis"] = ("tors", "0123", 180, 50, 1)  # type: ignore[assignment]
        definer_dict["s"] = ("nb", conv_meas["svdw_e"], conv_meas["svdw_s"])

    if isinstance(diverging, cgx.molecular.TwoC1Arm):
        beads = diverging.get_bead_set()
        if "a" not in beads or "c" not in beads:
            raise RuntimeError
        definer_dict["ba"] = ("bond", dive_meas["ba"] / cg_scale, 1e5)
        ac = dive_meas["aa"] / 2
        definer_dict["ac"] = ("bond", ac / cg_scale, 1e5)
        definer_dict["bac"] = ("angle", dive_meas["bac"], 1e2)
        definer_dict["bacab"] = ("tors", "0134", dive_meas["bacab"], 50, 1)  # type: ignore[assignment]
    else:
        raise NotImplementedError

    return cgx.systems_optimisation.get_forcefield_from_dict(
        identifier=f"{pair}ff",
        prefix=f"{pair}ff",
        vdw_bond_cutoff=vdw_bond_cutoff,
        present_beads=present_beads,
        definer_dict=definer_dict,
    )
