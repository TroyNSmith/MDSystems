from functools import partial
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray
from numpy.polynomial.polynomial import Polynomial as Poly
from scipy.spatial import KDTree
import pandas as pd
import multiprocessing as mp

from ..coordinates import Coordinates


def _pbc_points_reduced(
    coordinates: ArrayLike,
    pore_geometry: str,
    box: Optional[NDArray] = None,
    thickness: Optional[float] = None,
) -> tuple[NDArray, NDArray]:
    if box is None:
        box = coordinates.box
    if pore_geometry == "cylindrical":
        grid = np.array([[i, j, k] for k in [-1, 0, 1] for j in [0] for i in [0]])
        indices = np.tile(np.arange(len(coordinates)), 3)
    elif pore_geometry == "slit":
        grid = np.array(
            [[i, j, k] for k in [0] for j in [1, 0, -1] for i in [-1, 0, 1]]
        )
        indices = np.tile(np.arange(len(coordinates)), 9)
    else:
        raise ValueError(
            f"pore_geometry is {pore_geometry}, should either be "
            f"'cylindrical' or 'slit'"
        )
    coordinates_pbc = np.concatenate([coordinates + v @ box for v in grid], axis=0)
    size = np.diag(box)

    if thickness is not None:
        mask = np.all(coordinates_pbc > -thickness, axis=1)
        coordinates_pbc = coordinates_pbc[mask]
        indices = indices[mask]
        mask = np.all(coordinates_pbc < size + thickness, axis=1)
        coordinates_pbc = coordinates_pbc[mask]
        indices = indices[mask]

    return coordinates_pbc, indices


def _build_tree(points, box, r_max, pore_geometry):
    if np.all(np.diag(np.diag(box)) == box):
        tree = KDTree(points % box, boxsize=box)
        points_pbc_index = None
    else:
        points_pbc, points_pbc_index = _pbc_points_reduced(
            points,
            pore_geometry,
            box,
            thickness=r_max + 0.01,
        )
        tree = KDTree(points_pbc)
    return tree, points_pbc_index


def occupation_matrix(
    trajectory: Coordinates,
    edge_length: float = 0.05,
    segments: int = 1000,
    skip: float = 0.1,
    nodes: int = 8,
) -> pd.DataFrame:
    frame_indices = np.unique(
        np.int_(np.linspace(len(trajectory) * skip, len(trajectory) - 1, num=segments))
    )

    box = trajectory[0].box
    x_bins = np.arange(0, box[0][0] + edge_length, edge_length)
    y_bins = np.arange(0, box[1][1] + edge_length, edge_length)
    z_bins = np.arange(0, box[2][2] + edge_length, edge_length)
    bins = [x_bins, y_bins, z_bins]
    # Trajectory is split for parallel computing
    indices = np.array_split(frame_indices, nodes)
    pool = mp.Pool(nodes)
    results = pool.map(
        partial(_calc_histogram, trajectory=trajectory, bins=bins), indices
    )
    pool.close()
    matbin = np.sum(results, axis=0)
    x = (x_bins[:-1] + x_bins[1:]) / 2
    y = (y_bins[:-1] + y_bins[1:]) / 2
    z = (z_bins[:-1] + z_bins[1:]) / 2

    coords = np.array(np.meshgrid(x, y, z, indexing="ij"))
    coords = np.array([x.flatten() for x in coords])
    matbin_new = matbin.flatten()
    occupation_df = pd.DataFrame(
        {"x": coords[0], "y": coords[1], "z": coords[2], "occupation": matbin_new}
    )
    occupation_df = occupation_df.query("occupation != 0").reset_index(drop=True)
    return occupation_df


def _calc_histogram(
    indices: ArrayLike, trajectory: Coordinates, bins: ArrayLike
) -> NDArray:
    matbin = None
    for index in range(0, len(indices), 1000):
        try:
            current_indices = indices[index : index + 1000]
        except IndexError:
            current_indices = indices[index:]
        frames = np.concatenate(np.array([trajectory.pbc[i] for i in current_indices]))
        hist, _ = np.histogramdd(frames, bins=bins)
        if matbin is None:
            matbin = hist
        else:
            matbin += hist
    return matbin


def find_maxima(
    occupation_df: pd.DataFrame, box: ArrayLike, radius: float, pore_geometry: str
) -> pd.DataFrame:
    maxima_df = occupation_df.copy()
    maxima_df["maxima"] = None
    points = np.array(maxima_df[["x", "y", "z"]])
    tree, points_pbc_index = _build_tree(points, box, radius, pore_geometry)
    for i in range(len(maxima_df)):
        if maxima_df.loc[i, "maxima"] is not None:
            continue
        maxima_pos = maxima_df.loc[i, ["x", "y", "z"]]
        neighbors = np.array(tree.query_ball_point(maxima_pos, radius))
        if points_pbc_index is not None:
            neighbors = points_pbc_index[neighbors]
        neighbors = neighbors[neighbors != i]
        if len(neighbors) == 0:
            maxima_df.loc[i, "maxima"] = True
        elif (
            maxima_df.loc[neighbors, "occupation"].max()
            < maxima_df.loc[i, "occupation"]
        ):
            maxima_df.loc[neighbors, "maxima"] = False
            maxima_df.loc[i, "maxima"] = True
        else:
            maxima_df.loc[i, "maxima"] = False
    return maxima_df


def _calc_energies(
    maxima_indices: ArrayLike,
    maxima_df: pd.DataFrame,
    bins: ArrayLike,
    box: NDArray,
    pore_geometry: str,
    T: float,
    nodes: int = 8,
) -> NDArray:
    points = np.array(maxima_df[["x", "y", "z"]])
    tree, points_pbc_index = _build_tree(points, box, bins[-1], pore_geometry)
    maxima = maxima_df.loc[maxima_indices, ["x", "y", "z"]]
    maxima_occupations = np.array(maxima_df.loc[maxima_indices, "occupation"])
    num_of_neighbors = np.max(
        tree.query_ball_point(maxima, bins[-1], return_length=True)
    )
    split_maxima = []
    for i in range(0, len(maxima), 1000):
        split_maxima.append(maxima[i : i + 1000])

    distances = []
    indices = []
    for maxima in split_maxima:
        distances_step, indices_step = tree.query(
            maxima, k=num_of_neighbors, distance_upper_bound=bins[-1], workers=nodes
        )
        distances.append(distances_step)
        indices.append(indices_step)
    distances = np.concatenate(distances)
    indices = np.concatenate(indices)
    all_energy_hist = []
    all_occupied_bins_hist = []
    if distances.ndim == 1:
        current_distances = distances[1:][distances[1:] <= bins[-1]]
        if points_pbc_index is None:
            current_indices = indices[1:][distances[1:] <= bins[-1]]
        else:
            current_indices = points_pbc_index[indices[1:][distances[1:] <= bins[-1]]]
        energy = (
            -np.log(maxima_df.loc[current_indices, "occupation"] / maxima_occupations)
            * T
        )
        energy_hist = np.histogram(current_distances, bins=bins, weights=energy)[0]
        occupied_bins_hist = np.histogram(current_distances, bins=bins)[0]
        result = energy_hist / occupied_bins_hist
        return result
    for i, maxima_occupation in enumerate(maxima_occupations):
        current_distances = distances[i, 1:][distances[i, 1:] <= bins[-1]]
        if points_pbc_index is None:
            current_indices = indices[i, 1:][distances[i, 1:] <= bins[-1]]
        else:
            current_indices = points_pbc_index[
                indices[i, 1:][distances[i, 1:] <= bins[-1]]
            ]
        energy = (
            -np.log(maxima_df.loc[current_indices, "occupation"] / maxima_occupation)
            * T
        )
        energy_hist = np.histogram(current_distances, bins=bins, weights=energy)[0]
        occupied_bins_hist = np.histogram(current_distances, bins=bins)[0]
        all_energy_hist.append(energy_hist)
        all_occupied_bins_hist.append(occupied_bins_hist)
    result = np.sum(all_energy_hist, axis=0) / np.sum(all_occupied_bins_hist, axis=0)
    return result


def add_distances(
    occupation_df: pd.DataFrame, pore_geometry: str, origin: ArrayLike
) -> pd.DataFrame:
    distance_df = occupation_df.copy()
    if pore_geometry == "cylindrical":
        distance_df["distance"] = (
            (distance_df["x"] - origin[0]) ** 2 + (distance_df["y"] - origin[1]) ** 2
        ) ** (1 / 2)
    elif pore_geometry == "slit":
        distance_df["distance"] = np.abs(distance_df["z"] - origin[2])
    else:
        raise ValueError(
            f"pore_geometry is {pore_geometry}, should either be "
            f"'cylindrical' or 'slit'"
        )
    return distance_df


def distance_resolved_energies(
    maxima_df: pd.DataFrame,
    distance_bins: ArrayLike,
    r_bins: ArrayLike,
    box: NDArray,
    pore_geometry: str,
    temperature: float,
    nodes: int = 8,
) -> pd.DataFrame:
    results = []
    distances = []
    for i in range(len(distance_bins) - 1):
        maxima_indices = np.array(
            maxima_df.index[
                (maxima_df["distance"] >= distance_bins[i])
                * (maxima_df["distance"] < distance_bins[i + 1])
                * (maxima_df["maxima"] == True)
            ]
        )
        try:
            results.append(
                _calc_energies(
                    maxima_indices,
                    maxima_df,
                    r_bins,
                    box,
                    pore_geometry,
                    temperature,
                    nodes,
                )
            )
            distances.append((distance_bins[i] + distance_bins[i + 1]) / 2)
        except ValueError:
            pass

    radii = (r_bins[:-1] + r_bins[1:]) / 2
    d = np.array([d for d in distances for r in radii])
    r = np.array([r for d in distances for r in radii])
    result = np.array(results).flatten()
    return pd.DataFrame({"d": d, "r": r, "energy": result})


def find_energy_maxima(
    energy_df: pd.DataFrame,
    r_min: float,
    r_max: float,
    r_eval: float = None,
    degree: int = 2,
) -> pd.DataFrame:
    distances = []
    energies = []
    for d, data_d in energy_df.groupby("d"):
        distances.append(d)
        x = np.array(data_d["r"])
        y = np.array(data_d["energy"])
        mask = (x >= r_min) * (x <= r_max)
        p3 = Poly.fit(x[mask], y[mask], deg=degree)
        if r_eval is None:
            energies.append(np.max(p3(np.linspace(r_min, r_max, 1000))))
        else:
            energies.append(p3(r_eval))
    return pd.DataFrame({"d": distances, "energy": energies})
