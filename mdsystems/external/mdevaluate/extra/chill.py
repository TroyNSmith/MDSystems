from typing import Tuple, Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray
import pandas as pd
from scipy import sparse
from scipy.spatial import KDTree
from scipy.special import sph_harm

from ..coordinates import CoordinateFrame, Coordinates
from ..pbc import pbc_points


def calc_aij(atoms: ArrayLike, N: int = 4, l: int = 3) -> tuple[NDArray, NDArray]:
    tree = KDTree(atoms)

    dist, indices = tree.query(atoms, N + 1)
    indices = indices[:, 1:]

    vecs = atoms[:, np.newaxis, :] - atoms[indices]
    vecs /= np.linalg.norm(vecs, axis=-1)[..., np.newaxis]

    theta = np.arctan2(vecs[..., 1], vecs[..., 0]) + np.pi
    phi = np.arccos(vecs[..., 2])
    qijlm = sph_harm(
        np.arange(-l, l + 1)[np.newaxis, np.newaxis, :],
        l,
        theta[..., np.newaxis],
        phi[..., np.newaxis],
    )
    qilm = np.average(qijlm, axis=1)
    qil = np.sum(qilm * np.conj(qilm), axis=-1) ** 0.5
    aij = (
        np.sum(qilm[:, np.newaxis, :] * np.conj(qilm[indices]), axis=-1)
        / qil[:, np.newaxis]
        / qil[indices]
    )
    return np.real(aij), indices


def classify_ice(
    aij: NDArray, indices: NDArray, neighbors: NDArray, indexSOL: NDArray
) -> NDArray:
    staggerdBonds = np.sum(aij <= -0.8, axis=1)
    eclipsedBonds = np.sum((aij >= -0.35) & (aij <= 0.25), axis=1)

    iceTypes = np.full(len(aij), 5)
    for i in indexSOL:
        if neighbors[i] != 4:
            continue
        elif staggerdBonds[i] == 4:
            iceTypes[i] = 0
        elif staggerdBonds[i] == 3 and eclipsedBonds[i] == 1:
            iceTypes[i] = 1
        elif staggerdBonds[i] == 3:
            for j in indices[i]:
                if staggerdBonds[j] >= 2:
                    iceTypes[i] = 2
                    break
        elif staggerdBonds[i] == 2:
            for j in indices[i]:
                if staggerdBonds[j] >= 3:
                    iceTypes[i] = 2
                    break
        elif eclipsedBonds[i] == 4:
            iceTypes[i] = 3
        elif eclipsedBonds[i] == 3:
            iceTypes[i] = 4
    iceTypes = iceTypes[indexSOL]

    return iceTypes


def count_ice_types(iceTypes: NDArray) -> NDArray:
    cubic = len(iceTypes[iceTypes == 0])
    hexagonal = len(iceTypes[iceTypes == 1])
    interface = len(iceTypes[iceTypes == 2])
    clathrate = len(iceTypes[iceTypes == 3])
    clathrate_interface = len(iceTypes[iceTypes == 4])
    liquid = len(iceTypes[iceTypes == 5])
    return np.array(
        [cubic, hexagonal, interface, clathrate, clathrate_interface, liquid]
    )


def selector_ice(
    oxygen_atoms_water: CoordinateFrame,
    chosen_ice_types: ArrayLike,
    combined: bool = True,
    next_neighbor_distance: float = 0.35,
) -> NDArray:
    atoms = oxygen_atoms_water
    atoms_PBC = pbc_points(atoms, thickness=next_neighbor_distance * 2.2)
    aij, indices = calc_aij(atoms_PBC)
    tree = KDTree(atoms_PBC)
    neighbors = tree.query_ball_point(
        atoms_PBC, next_neighbor_distance, return_length=True
    ) - 1
    index_SOL = atoms_PBC.tolist().index(atoms[0].tolist())
    index_SOL = np.arange(index_SOL, index_SOL + len(atoms))
    ice_Types = classify_ice(aij, indices, neighbors, index_SOL)
    index = []
    if combined is True:
        for i, ice_Type in enumerate(ice_Types):
            if ice_Type in chosen_ice_types:
                index.append(i)
    else:
        for entry in chosen_ice_types:
            index_entry = []
            for i, ice_Type in enumerate(ice_Types):
                if ice_Type == entry:
                    index_entry.append(i)
            index.append(np.array(index_entry))
    return np.array(index)


def ice_types(trajectory: Coordinates, segments: int = 10000) -> pd.DataFrame:
    def ice_types_distribution(frame: CoordinateFrame, selector: Callable) -> NDArray:
        atoms_PBC = pbc_points(frame, thickness=1)
        aij, indices = calc_aij(atoms_PBC)
        tree = KDTree(atoms_PBC)
        neighbors = tree.query_ball_point(atoms_PBC, 0.35, return_length=True) - 1
        index = selector(frame, atoms_PBC)
        ice_types_data = classify_ice(aij, indices, neighbors, index)
        ice_parts_data = count_ice_types(ice_types_data)
        return ice_parts_data

    def selector(frame: CoordinateFrame, atoms_PBC: ArrayLike) -> NDArray:
        atoms_SOL = traj.subset(residue_name="SOL")[frame.step]
        index = atoms_PBC.tolist().index(atoms_SOL[0].tolist())
        index = np.arange(index, index + len(atoms_SOL))
        return np.array(index)

    traj = trajectory.subset(atom_name="OW")

    frame_indices = np.unique(np.int_(np.linspace(0, len(traj) - 1, num=segments)))

    result = np.array(
        [
            [
                traj[frame_index].time,
                *ice_types_distribution(traj[frame_index], selector),
            ]
            for frame_index in frame_indices
        ]
    )

    return pd.DataFrame(
        {
            "time": result[:, 0],
            "cubic": result[:, 1],
            "hexagonal": result[:, 2],
            "ice_interface": result[:, 3],
            "clathrate": result[:, 4],
            "clathrate_interface": result[:, 5],
            "liquid": result[:, 6],
        }
    )


def ice_clusters_traj(
    traj: Coordinates, segments: int = 10000, skip: float = 0.1
) -> list:
    def ice_clusters(frame: CoordinateFrame) -> Tuple[float, list]:
        selection = selector_ice(frame, [0, 1, 2])
        if len(selection) == 0:
            return frame.time, []
        else:
            ice = frame[selection]
            ice_PBC, indices_PBC = pbc_points(
                ice, box=frame.box, thickness=0.5, index=True
            )
            ice_tree = KDTree(ice_PBC)
            ice_matrix = ice_tree.sparse_distance_matrix(
                ice_tree, 0.35, output_type="ndarray"
            )
            new_ice_matrix = np.zeros((len(ice), len(ice)))
            for entry in ice_matrix:
                if entry[2] > 0:
                    new_ice_matrix[indices_PBC[entry[0]], indices_PBC[entry[1]]] = 1
            n_components, labels = sparse.csgraph.connected_components(
                new_ice_matrix, directed=False
            )
            clusters = []
            selection = np.array(selection)
            for i in range(0, np.max(labels) + 1):
                if len(ice[labels == i]) > 1:
                    clusters.append(
                        list(zip(selection[labels == i], ice[labels == i].tolist()))
                    )
            return frame.time, clusters

    frame_indices = np.unique(
        np.int_(np.linspace(len(traj) * skip, len(traj) - 1, num=segments))
    )
    all_clusters = [
        ice_clusters(traj[frame_index]) for frame_index in frame_indices
    ]
    return all_clusters
