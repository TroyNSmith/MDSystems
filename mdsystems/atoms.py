"""Modules for handling trajectory and topology data."""

from functools import cached_property
from pathlib import Path

import mdtraj as mdt
import numpy as np


class System:
    """Container for relevant trajectory and topology data."""

    def __init__(
        self,
        traj: str | Path,
        top: str | Path,
        stride: int = 1,
        top_filter: str | None = None,
    ):
        self._mdtraj = mdt.load(traj, top=top, stride=stride)
        self.top_filter = top_filter

    @cached_property
    def trajectory(self) -> mdt.Trajectory:
        """Return trajectory from MDTraj."""
        return self._mdtraj

    @cached_property
    def topology(self) -> mdt.Topology:
        """Return topology from MDTraj."""
        return self._mdtraj.topology

    @property
    def centers_of_mass(self) -> np.ndarray:
        """Compute center of mass coordinates with topological filtering."""
        xyz = self.coordinates
        n_frames = xyz.shape[0]
        n_residues = len(self.residue_indices)
        coms = np.empty((n_frames, n_residues, 3), dtype=np.float64)

        selected_mask_map = {atom: i for i, atom in enumerate(self.selected_atoms)}

        for i, atoms in enumerate(self.residue_indices):
            atoms_in_mask = np.array([selected_mask_map[a] for a in atoms], dtype=np.intp)
            masses = self.residue_masses[i]
            total_mass = masses.sum()

            if len(atoms_in_mask) == 0 or total_mass == 0.0:
                coms[:, i, :] = np.nan
                continue

            coms[:, i, :] = (
                np.sum(xyz[:, atoms_in_mask, :] * masses[None, :, None], axis=1) / total_mass
            )

        return coms

    @property
    def coordinates(self) -> np.ndarray:
        """Return coordinates from MDTrajectory object."""
        mask = self.selected_atoms
        return self._mdtraj.xyz[:, mask, :]

    @cached_property
    def num_atoms(self) -> int:
        """Return number of atoms in topology."""
        return self.trajectory.n_atoms

    @property
    def num_frames(self) -> int:
        """Return number of frames in trajectory."""
        return self.trajectory.n_frames

    @cached_property
    def residue_indices(self) -> np.ndarray:
        """Return list of atom indices for each residue after topological filtering."""
        selected_atoms = set(self.selected_atoms)
        residue_indices = []

        for r in self.topology.residues:
            atoms = [a.index for a in r.atoms if a.index in selected_atoms]
            if atoms:
                residue_indices.append(np.array(atoms, dtype=np.int32))

        return residue_indices

    @cached_property
    def residue_masses(self) -> np.ndarray:
        """Return list of atomic masses per residue for selected atoms."""
        masses_per_residue = []
        for r_indices in self.residue_indices:
            masses_per_residue.append(
                np.array([self.topology.atom(i).element.mass for i in r_indices], dtype=np.float32)
            )

        return masses_per_residue

    @property
    def selected_atoms(self) -> np.ndarray:
        """Return indices for atoms / residues matching topology filter."""
        if self.top_filter is not None:
            return self.topology.select(self.top_filter)
        else:
            return np.arange(self.topology.n_atoms)

    @property
    def times(self) -> np.ndarray:
        """Return timestamps for each frame in trajectory."""
        return self.trajectory.time

    @property
    def unitcell(self) -> np.ndarray:
        """Return unitcell vectors from each frame in trajectory."""
        return self.trajectory.unitcell_vectors

    def volume(self, average: bool = True) -> mdt.Topology:
        """Return unitcell volumes from MDTraj.

        Parameters
        ----------
        average : bool
            If True, return the average volume over the length of the entire trajectory.

        Returns
        -------
        volume : np.ndarray | np.float32

        """
        volumes = (
            np.average(self.trajectory.unitcell_volumes)
            if average
            else self.trajectory.unitcell_volumes
        )

        return volumes


class SubSystem:
    """Subsystem defined by spatial xyz_filter evaluated over frame interval."""

    def __init__(
        self,
        parent: System,
        start_frame: int = 0,
        end_frame: int | None = None,
        xyz_filter: str = "x > -100.0",
        strict: bool = False,
    ):
        self.parent = parent
        self.start_frame = start_frame
        self.end_frame = end_frame

        self._init_mask(xyz_filter, strict)

    def _init_mask(self, xyz_filter, strict):
        xyzs = self.parent.coordinates(self.start_frame, self.end_frame)

        if not strict:
            x, y, z = xyzs[0].T

            mask = eval(xyz_filter, {"np": np}, {"x": x, "y": y, "z": z})

        else:
            x = xyzs[:, :, 0]
            y = xyzs[:, :, 1]
            z = xyzs[:, :, 2]

            cond = eval(xyz_filter, {"np": np}, {"x": x, "y": y, "z": z})

            mask = np.all(cond, axis=0)

            raise NotImplementedError(
                "Strict (every frame rather than first frame) spatial filtering not implemented."
            )

        self.atom_mask = mask
        self.selected_atoms = self.parent.selected_atoms[mask]
