"""Functions for loading and sorting trajectories/topologies."""

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Generator

import mdtraj as mdt
import numpy as np


@dataclass
class System:
    """Container for components of MD system."""

    # File inputs
    traj: str | Path
    top: str | Path | None = None
    # Atom filters
    top_filter: str | None = None
    xyz_filter: str = "x > -100.0"
    # Coordinate loading options
    chunk_size: int = 100
    stride_len: int = 0
    tile_size: int = 5000

    @property
    def trajectory(self) -> Generator:
        """Return system trajectory with chunk_size frames."""
        return mdt.iterload(self.traj, top=self.top, chunk=self.chunk_size, stride=self.stride_len)

    @cached_property
    def topology(self) -> mdt.Topology:
        """Return system topology information."""
        first_chunk = next(mdt.iterload(self.traj, top=self.top, chunk=1))
        first_frame = first_chunk[0]
        return first_frame.topology

    @cached_property
    def selected_atoms(self) -> np.ndarray:
        """Return indices of atoms matching topology selection (System.top_filter)."""
        if self.top_filter is not None:
            return self.topology.select(self.top_filter)
        else:
            return np.arange(self.topology.n_atoms)

    @cached_property
    def times(self) -> np.ndarray:
        """Return time stamps for every frame in trajectory."""
        times = []
        for chunk in mdt.iterload(self.traj, top=self.top):
            t = [frame.time[0] for frame in chunk]
            times.append(t)
        return np.concatenate(times)

    @cached_property
    def unitcell(self) -> np.ndarray:
        """Return system unitcell vectors."""
        first_chunk = next(mdt.iterload(self.traj, top=self.top, chunk=1))
        first_frame = first_chunk[0]
        return first_frame.unitcell_vectors[0]

    @cached_property
    def volume(self) -> np.ndarray:
        """Return system volume."""
        first_chunk = next(mdt.iterload(self.traj, top=self.top, chunk=1))
        first_frame = first_chunk[0]
        return first_frame.unitcell_volumes[0]

    def load_cartesian(
        self,
        com: bool = False,
        filter_xyz: bool = True,
        mask: np.ndarray = None,
        return_mask: bool = False,
    ) -> Generator[np.ndarray, None, None]:
        """Yield one set of Cartesian coordinates at a time from system trajectory."""
        if com:
            self._parse_residues()

        for chunk in self.trajectory:
            for frame in chunk:
                # Filter out by topological selections
                xyz = frame.xyz[0][self.selected_atoms]

                # Compute centers of masses
                if com:
                    xyz = centers_of_masses(xyz, self.residue_indices, self.residue_masses)

                # Evaluate mask for positional selections
                if filter_xyz and mask is None:
                    x, y, z = xyz.T
                    mask = eval(self.xyz_filter, {"np": np}, {"x": x, "y": y, "z": z})

                else:  # All True or external mask
                    mask = np.full(len(xyz), True, dtype=np.bool) if mask is None else mask

                # Return mask with xyz (to use same filter for additional steps)
                if return_mask:
                    yield xyz, mask

                else:
                    yield xyz[mask]

    def _parse_residues(self):
        """Extract residue-level atom lists and atomic masses after topology filtering."""
        selected_atoms = self.selected_atoms
        selected_atoms_set = set(selected_atoms)

        # original atom index -> filtered index
        filtered_index_map = {orig: i for i, orig in enumerate(selected_atoms)}

        self.residue_indices = []
        self.residue_masses = []

        for r in self.topology.residues:
            # Filter atoms that survive the topology filter
            atoms = [a for a in r.atoms if a.index in selected_atoms_set]
            if not atoms:
                continue

            filtered_atom_indices = np.array(
                [filtered_index_map[a.index] for a in atoms], dtype=np.int32
            )

            masses = np.array([a.element.mass for a in atoms], dtype=np.float32)

            self.residue_indices.append(filtered_atom_indices)
            self.residue_masses.append(masses)


def centers_of_masses(
    xyz: np.ndarray, residue_indices: list[np.ndarray], residue_masses: list[np.ndarray]
) -> np.ndarray:
    """Compute residue-level centers of mass for a single frame."""
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
