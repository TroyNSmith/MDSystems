"""Functions for loading, storing, and manipulating coordinates."""

from typing import Any, Generator

import mdtraj
import numpy as np
from numba import njit
from pydantic import BaseModel, Field

try:
    import cupy as cp

    xp = cp
    print("Successfully enabled cuda acceleration.")
except Exception:
    xp = np


class System(BaseModel):  # noqa: D101
    trajectory: str
    xyz_filter: str | None = None
    topology: str | None = None
    topology_filter: str | None = None

    stride: int | None = Field(default=0, ge=0)
    chunk_size: int = Field(default=0, ge=0)

    def iter_coords(self) -> Generator[Any, Any, None]:
        """Return mdtrajectory Trajectory generator."""
        return mdtraj.iterload(
            self.trajectory, top=self.topology, chunk=self.chunk_size, skip=self.stride
        )


# Numba-accelerated per-residue COM computation
@njit
def __residue_coms(xyz_frame, residue_atom_indices, residue_masses, coms):
    for i in range(len(residue_atom_indices)):
        atoms = residue_atom_indices[i]
        masses = residue_masses[i]
        if len(atoms) == 0:
            coms[i] = np.nan
            continue
        coords = xyz_frame[atoms]
        total_mass = masses.sum()
        if total_mass == 0:
            coms[i] = np.nan
            continue
        coms[i] = np.sum(coords * masses[:, None], axis=0) / total_mass


def com_coordinates(
    system: Any,  # your System type
    top_filter: str | None = None,
    xyz_filter: str | None = None,
) -> Generator[np.ndarray, None, None]:
    """Yield center of mass coordinates for residues in the system per frame."""
    top_filter = top_filter or system.topology_filter
    xyz_filter = xyz_filter or system.xyz_filter

    for chunk in system.iter_coords():
        top = chunk.topology

        if top_filter:
            atom_selection = top.select(top_filter)
        else:
            atom_selection = np.arange(top.n_atoms)

        # Precompute per-residue atom indices and masses
        residue_atom_indices = []
        residue_masses = []
        for r in top.residues:
            atoms = np.intersect1d(top.select(f"resid {r.index}"), atom_selection)
            residue_atom_indices.append(atoms.astype(np.int32))
            residue_masses.append(
                np.array([top.atom(int(i)).element.mass for i in atoms], dtype=np.float32)
            )

        coms = np.empty((top.n_residues, 3), dtype=np.float32)

        # Iterate frames
        for frame in chunk:
            xyz = frame.xyz[0].astype(np.float32)
            __residue_coms(xyz, residue_atom_indices, residue_masses, coms)

            if xyz_filter:
                x, y, z = coms.T
                mask = eval(xyz_filter, {"np": np}, {"x": x, "y": y, "z": z})
                yield coms[mask]
            else:
                yield coms

def raw_coordinates(
    system: System,
    top_filter: str | None = None,
    xyz_filter: str | None = None,
) -> Generator[np.ndarray, None, None]:
    """Yield raw atomic coordinates from the System with optional filtering.

    :param system: System object with trajectory and topology.
    :param top_filter: MDTraj atom selection string (e.g., "resid 10 to 50") to select subset of atoms.
    :param xyz_filter: Coordinate filter applied to x, y, z of atoms.
        Example: "(x > 1.0) & (y < 1.0)".

    Yields
    ------
    np.ndarray
        Array of shape (n_atoms_filtered, 3) for each frame.
    """
    top_filter = top_filter or system.topology_filter
    xyz_filter = xyz_filter or system.xyz_filter

    for chunk in system.iter_coords():
        top = chunk.topology

        if top_filter:
            atom_selection = top.select(top_filter)
        else:
            atom_selection = np.arange(top.n_atoms)

        for frame in chunk:
            xyz = frame.xyz[0, atom_selection, :]

            if xyz_filter:
                x, y, z = xyz.T
                mask = eval(xyz_filter, {"np": np}, {"x": x, "y": y, "z": z})
                xyz = xyz[mask]

            if xyz.size > 0:
                yield xyz


def unit_cells(system: System) -> Generator[np.ndarray, None, None]:
    """Return unit cell matrices for each frame in the trajectory."""
    for chunk in system.iter_coords():
        for uc in chunk.unitcell_vectors:
            yield uc
