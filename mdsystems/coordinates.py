"""Functions for loading, storing, and manipulating coordinates."""

from dataclasses import dataclass
from typing import Generator

import mdtraj as mdt
import numpy as np


@dataclass
class System:
    """
    Container for trajectory paths, topology, and frame processing utilities.

    Parameters
    ----------
    traj : str
        Path to trajectory file readable by MDTraj.
    top : str | None, optional
        Path to topology file. Required for formats like XTC or TRR that
        do not store atom information.
    chunk : int, optional
        Number of frames to load per chunk using `mdtraj.iterload`.
        This controls memory usage but not the logical behavior.
    stride : int, optional
        Frame stride to use when loading the trajectory. A stride of 1 loads
        every frame; a stride of N loads every Nth frame.

    Notes
    -----
    The trajectory is not loaded into memory; all access is chunked via
    `mdtraj.iterload` for memory safety.
    """

    traj: str
    top: str | None = None

    top_filter: str | None = None
    xyz_filter: str = "x >= -100.0"

    chunk: int = 100
    stride: int = 0

    @property
    def trajectory(self) -> mdt.Trajectory:
        """
        Return an iterator over trajectory chunks using `mdtraj.iterload`.

        Returns
        -------
        TrajectoryIterator
            An iterator that yields MDTraj Trajectory objects of size `chunk`.

        Notes
        -----
        This does *not* return a full trajectory in memory. Each iteration
        yields a new Trajectory chunk, allowing processing of arbitrarily
        large files.
        """
        return mdt.iterload(self.traj, top=self.top, chunk=self.chunk, stride=self.stride)

    @property
    def topology(self) -> mdt.Topology:
        """
        Return the topology for the trajectory.

        Returns
        -------
        mdt.Topology
            The MDTraj topology object taken from the first frame of the
            first loaded chunk.

        Notes
        -----
        This forces the loading of the first chunk of the trajectory. Only
        the topology is taken; coordinates are discarded.
        """
        first_chunk = next(
            mdt.iterload(self.traj, top=self.top, chunk=self.chunk, stride=self.stride)
        )
        first_frame = first_chunk[0]
        return first_frame.topology

    def load_frame(
        self,
        com: bool = False,
        vecs: tuple[str, str] | None = None,
        topology_filter: str | None = None,
        xyz_filter: str = "x > -100.0",
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """
        Yield filtered frame data from the trajectory.

        Parameters
        ----------
        com : bool, optional
            If True, compute centers of mass for each residue.
        vecs : tuple(str, str) | None, optional
            If provided, compute residue vectors A1 → A2 using atom names.
            Example: vecs=("CA", "CB").
        topology_filter : str | None
            MDTraj DSL atom selector. Limits which residues/atoms are included.
            Overrides the class default if provided.
        xyz_filter : str | None
            Boolean expression applied to per-residue coordinates/vectors.
            Overrides the class default if provided.

        Yields
        ------
        (first_frame_filtered, frame_filtered)
            Each with shape (M, 3), where M is number of residues/atoms passing filters.
        """
        if com and vecs is not None:
            raise ValueError("com=True and vecs!=None are mutually exclusive.")

        topology_filter = (
            topology_filter
            if topology_filter is not None
            else getattr(self, "topology_filter", None)
        )
        xyz_filter = (
            xyz_filter if xyz_filter is not None else getattr(self, "xyz_filter", "x > -100.0")
        )

        if topology_filter is not None:
            selected_atoms = self.topology.select(self.topology_filter)
        else:
            selected_atoms = np.arange(self.topology.n_atoms)

        if com:
            residue_indices, residue_masses = parse_residues(self.topology, selected_atoms)

        if vecs is not None:
            atom1, atom2 = vecs
            residue_pairs = parse_vector_pairs(self.topology, atom1, atom2, selected_atoms)

        for chunk in self.trajectory:
            frame0 = chunk[0][0]
            box0 = frame0.unitcell_lengths[0]
            xyz0 = frame0.xyz[0]

            if com:
                first_frame = centers_of_masses(xyz0, residue_indices, residue_masses)
            elif vecs is not None:
                first_frame = residue_vectors(xyz0, residue_pairs, box0)
            else:
                first_frame = xyz0

            spatial_indices = spatial_mask(first_frame, xyz_filter)

            for frame in chunk:
                xyz = frame.xyz[0]
                box = frame.unitcell_lengths[0]

                print(f"Frame from: {frame.time[0]} time units")

                if com:
                    vals = centers_of_masses(xyz, residue_indices, residue_masses)
                elif vecs is not None:
                    vals = residue_vectors(xyz, residue_pairs, box)
                else:
                    vals = xyz

                yield first_frame[spatial_indices], vals[spatial_indices]


def parse_residues(
    topology: mdt.Topology,
    selected_atoms: np.ndarray | None = None,
):
    """
    Extract residue-level atom lists and atomic masses from an MDTraj topology.

    Parameters
    ----------
    topology : mdt.Topology
        The MDTraj topology describing atom connectivity and residue structure.
    selected_atoms : array-like or None, optional
        Indices of atoms to include. If None, all atoms in the topology are used.

    Returns
    -------
    residue_indices : list of np.ndarray
        For each residue, a NumPy array of atom indices belonging to that residue
        and included in `selected_atoms`.
    residue_masses : list of np.ndarray
        For each residue, the atomic masses corresponding to the indices in
        `residue_indices`.

    Notes
    -----
    Residues with no selected atoms are skipped. The output lists therefore
    include only residues that contain at least one selected atom.
    """
    if selected_atoms is None:
        selected_atoms = np.arange(topology.n_atoms)
    else:
        selected_atoms = np.asarray(selected_atoms, dtype=int)

    selected_atoms_set = set(selected_atoms)

    residue_indices = []
    residue_masses = []

    for r in topology.residues:
        atoms = np.array(
            [a.index for a in r.atoms if a.index in selected_atoms_set], dtype=np.int32
        )

        if atoms.size == 0:
            continue

        masses = np.array(
            [a.element.mass for a in r.atoms if a.index in selected_atoms_set], dtype=np.float32
        )

        residue_indices.append(atoms)
        residue_masses.append(masses)

    return residue_indices, residue_masses


def centers_of_masses(
    xyz: np.ndarray, residue_indices: list[np.ndarray], residue_masses: list[np.ndarray]
) -> np.ndarray:
    """
    Compute residue-level centers of mass for a single frame.

    Parameters
    ----------
    xyz : np.ndarray, shape (N, 3)
        Coordinates for all atoms in the frame.
    residue_indices : list of np.ndarray
        Atom indices belonging to each residue.
    residue_masses : list of np.ndarray
        Atomic masses corresponding to each residue.

    Returns
    -------
    np.ndarray, shape (R, 3)
        Center of mass of each residue (R residues). Residues that contain
        zero atoms or have total mass of zero are assigned NaN.

    Notes
    -----
    This operates on a single frame (not chunked).
    """
    coms = np.empty((len(residue_indices), 3), dtype=np.float32)

    for i in range(len(residue_indices)):
        atoms = residue_indices[i]
        masses = residue_masses[i]

        if len(atoms) == 0:
            coms[i] = np.nan
            continue

        coords = xyz[atoms]
        total_mass = masses.sum()

        if total_mass == 0:
            coms[i] = np.nan
            continue

        coms[i] = np.sum(coords * masses[:, None], axis=0) / total_mass

    return coms


def spatial_mask(xyz: np.ndarray, spatial_filter: str):
    """
    Create a boolean mask selecting coordinates based on a spatial filter.

    Parameters
    ----------
    xyz : np.ndarray, shape (N, 3)
        Coordinates to be filtered.
    spatial_filter : str
        Boolean expression evaluated using variables `x`, `y`, `z`.
        Example expressions:
            "x > 0"
            "np.sqrt(x**2 + y**2) < 5"
            "(z > 1.0) & (x < 2.0)"

    Returns
    -------
    np.ndarray, dtype=bool
        A boolean mask of shape (N,) indicating which coordinates satisfy
        the spatial filter.
    """
    x, y, z = xyz.T
    mask = eval(spatial_filter, {"np": np}, {"x": x, "y": y, "z": z})
    return mask


def parse_vector_pairs(
    topology: mdt.Topology,
    atom1: str,
    atom2: str,
    selected_atoms: np.ndarray | None = None,
) -> list[tuple[int, int]]:
    """
    Return residue-level atom index pairs (A1, A2).

    Only residues containing both atoms and passing selected_atoms are included.
    """
    if selected_atoms is not None:
        selected_atoms = set(selected_atoms)

    pairs = []

    for res in topology.residues:
        a1 = next((a.index for a in res.atoms if a.name == atom1), None)
        a2 = next((a.index for a in res.atoms if a.name == atom2), None)

        if a1 is None or a2 is None:
            continue

        if selected_atoms is not None and (a1 not in selected_atoms or a2 not in selected_atoms):
            continue

        pairs.append((a1, a2))

    return pairs


def residue_vectors(
    xyz: np.ndarray,
    residue_pairs: list[tuple[int, int]],
    box: np.ndarray | None,
) -> np.ndarray:
    """
    Compute PBC-corrected residue-level vectors A1 → A2 for a single frame.

    Parameters
    ----------
    xyz : np.ndarray, shape (N, 3)
        Atomic coordinates for the frame.
    residue_pairs : list of tuple(int, int)
        For each residue, (a1, a2) atom indices.
    box : np.ndarray or None
        Simulation box lengths, shape (3,), typically from frame.unitcell_lengths.
        If None, PBC is ignored.

    Returns
    -------
    np.ndarray, shape (R, 3)
        Minimum-image-corrected vectors for each residue.
    """
    vectors = np.empty((len(residue_pairs), 3), dtype=np.float32)

    for i, (a1, a2) in enumerate(residue_pairs):
        if a1 is None or a2 is None:
            vectors[i] = np.nan
            continue

        v = xyz[a2] - xyz[a1]

        if box is not None:
            v -= box * np.round(v / box)

        vectors[i] = v

    return vectors
