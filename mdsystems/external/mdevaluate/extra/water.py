from functools import partial
from typing import Tuple, Callable, Optional

import numpy as np
from numpy.typing import NDArray, ArrayLike
import pandas as pd
from scipy.spatial import KDTree

from ..distribution import hbonds
from ..pbc import pbc_points
from ..correlation import shifted_correlation, overlap
from ..coordinates import Coordinates, CoordinateFrame


def tanaka_zeta(
    trajectory: Coordinates, angle: float = 30, segments: int = 100, skip: float = 0.1
) -> pd.DataFrame:
    frame_indices = np.unique(
        np.int_(np.linspace(len(trajectory) * skip, len(trajectory) - 1, num=segments))
    )
    sel = trajectory.atom_subset.selection
    A = np.where(
        trajectory.subset(atom_name="OW", residue_name="SOL").atom_subset.selection[sel]
    )[0]
    D = np.vstack([A] * 2).T.reshape((-1,))
    H = np.where(
        trajectory.subset(atom_name="HW.", residue_name="SOL").atom_subset.selection[
            sel
        ]
    )[0]

    zeta_dist = []
    zeta_cg_dist = []
    for frame_index in frame_indices:
        D_frame = trajectory[frame_index][D]
        H_frame = trajectory[frame_index][H]
        A_frame = trajectory[frame_index][A]
        box = trajectory[frame_index].box
        pairs = hbonds(
            D_frame, H_frame, A_frame, box, min_cos=np.cos(angle / 180 * np.pi)
        )
        pairs[:, 0] = np.int_((pairs[:, 0] / 2))
        pairs = np.sort(pairs, axis=1)
        pairs = np.unique(pairs, axis=0)
        pairs = pairs.tolist()

        A_PBC, A_index = pbc_points(A_frame, box, thickness=0.7, index=True)
        A_tree = KDTree(A_PBC)
        dist, dist_index = A_tree.query(A_frame, 16, distance_upper_bound=0.7)

        dist_index = A_index[dist_index]
        zeta = []
        for i, indices in enumerate(dist_index):
            dist_hbond = []
            dist_non_hbond = []
            for j, index in enumerate(indices):
                if j != 0:
                    if np.sort([indices[0], index]).tolist() in pairs:
                        dist_hbond.append(dist[i, j])
                    else:
                        dist_non_hbond.append(dist[i, j])
            try:
                zeta.append(np.min(dist_non_hbond) - np.max(dist_hbond))
            except ValueError:
                zeta.append(0)

        zeta = np.array(zeta)

        dist, dist_index = A_tree.query(A_frame, 16, distance_upper_bound=0.7)
        dist_index = A_index[dist_index]
        dist_index = np.array(
            [indices[dist[i] <= 0.35] for i, indices in enumerate(dist_index)]
        )
        zeta_cg = np.array([np.mean(zeta[indices]) for indices in dist_index])

        bins = np.linspace(-0.1, 0.2, 301)
        zeta_dist.append(np.histogram(zeta, bins=bins)[0])
        zeta_cg_dist.append(np.histogram(zeta_cg, bins=bins)[0])
        z = bins[1:] - (bins[1] - bins[0]) / 2

    zeta_dist = np.mean(zeta_dist, axis=0)
    zeta_dist = zeta_dist / np.mean(zeta_dist)

    zeta_cg_dist = np.mean(zeta_cg_dist, axis=0)
    zeta_cg_dist = zeta_cg_dist / np.mean(zeta_cg_dist)

    return pd.DataFrame({"zeta": z, "result": zeta_dist, "result_cg": zeta_cg_dist})


def chi_four_trans(
    trajectory: Coordinates, skip: float = 0.1, segments: int = 10000
) -> pd.DataFrame:
    traj = trajectory.nojump
    N = len(trajectory[0])
    t, S = shifted_correlation(
        partial(overlap, radius=0.1), traj, skip=skip, segments=segments, average=False
    )
    chi = 1 / N * S.var(axis=0)[1:]
    return pd.DataFrame({"time": t[1:], "chi": chi})


def tanaka_correlation_map(
    trajectory: Coordinates,
    data_chi_four_trans: pd.DataFrame,
    angle: float = 30,
    segments: int = 100,
    skip: float = 0.1,
) -> pd.DataFrame:
    def tanaka_zeta_cg(
        trajectory: Coordinates,
        angle: float = 30,
        segments: int = 1000,
        skip: float = 0.1,
    ) -> Tuple[NDArray, NDArray]:
        frame_indices = np.unique(
            np.int_(
                np.linspace(len(trajectory) * skip, len(trajectory) - 1, num=segments)
            )
        )
        sel = trajectory.atom_subset.selection
        A = np.where(
            trajectory.subset(atom_name="OW", residue_name="SOL").atom_subset.selection[
                sel
            ]
        )[0]
        D = np.vstack([A] * 2).T.reshape((-1,))
        H = np.where(
            trajectory.subset(
                atom_name="HW.", residue_name="SOL"
            ).atom_subset.selection[sel]
        )[0]

        zeta_cg = []
        times = []
        for frame_index in frame_indices:
            D_frame = trajectory[frame_index][D]
            H_frame = trajectory[frame_index][H]
            A_frame = trajectory[frame_index][A]
            box = trajectory[frame_index].box
            pairs = hbonds(
                D_frame, H_frame, A_frame, box, min_cos=np.cos(angle / 180 * np.pi)
            )
            pairs[:, 0] = np.int_((pairs[:, 0] / 2))
            pairs = np.sort(pairs, axis=1)
            pairs = np.unique(pairs, axis=0)
            pairs = pairs.tolist()

            A_PBC, A_index = pbc_points(A_frame, box, thickness=0.7, index=True)
            A_tree = KDTree(A_PBC)
            dist, dist_index = A_tree.query(A_frame, 16, distance_upper_bound=0.7)

            dist_index = A_index[dist_index]
            zeta = []
            for i, indices in enumerate(dist_index):
                dist_hbond = []
                dist_non_hbond = []
                for j, index in enumerate(indices):
                    if j != 0:
                        if np.sort([indices[0], index]).tolist() in pairs:
                            dist_hbond.append(dist[i, j])
                        else:
                            dist_non_hbond.append(dist[i, j])
                try:
                    zeta.append(np.min(dist_non_hbond) - np.max(dist_hbond))
                except ValueError:
                    zeta.append(0)
            zeta = np.array(zeta)
            dist_index = np.array(
                [indices[dist[i] <= 0.35] for i, indices in enumerate(dist_index)]
            )
            zeta_cg.append(np.array([np.mean(zeta[indices]) for indices in dist_index]))
            times.append(trajectory[frame_index].time)
        return np.array(times), np.array(zeta_cg)

    def delta_r_max(
        trajectory: Coordinates, frame: CoordinateFrame, tau_4: float
    ) -> NDArray:
        dt = trajectory[1].time - trajectory[0].time
        index_start = frame.step
        index_end = index_start + int(tau_4 / dt) + 1
        frame_indices = np.arange(index_start, index_end + 1)
        end_cords = np.array([trajectory[frame_index] for frame_index in frame_indices])
        vectors = trajectory[index_start] - end_cords

        delta_r = np.linalg.norm(vectors, axis=-1)
        delta_r = np.max(delta_r, axis=0)
        return delta_r

    d = np.array(data_chi_four_trans[["time", "chi"]])
    mask = d[:, 1] >= 0.7 * np.max(d[:, 1])
    fit = np.polyfit(d[mask, 0], d[mask, 1], 4)
    p = np.poly1d(fit)
    x_inter = np.linspace(d[mask, 0][0], d[mask, 0][-1], 1e6)
    y_inter = p(x_inter)
    tau_4 = x_inter[y_inter == np.max(y_inter)]

    oxygens = trajectory.nojump.subset(atom_name="OW")
    window = tau_4 / trajectory[-1].time
    start_frames = np.unique(
        np.linspace(
            len(trajectory) * skip,
            len(trajectory) * (1 - window),
            num=segments,
            endpoint=False,
            dtype=int,
        )
    )

    times, zeta_cg = tanaka_zeta_cg(trajectory, angle=angle)

    zeta_cg_mean = np.array(
        [
            np.mean(
                zeta_cg[
                    (times >= trajectory[start_frame].time)
                    * (times <= (trajectory[start_frame].time + tau_4))
                ],
                axis=0,
            )
            for start_frame in start_frames
        ]
    ).flatten()
    delta_r = np.array(
        [
            delta_r_max(oxygens, oxygens[start_frame], tau_4)
            for start_frame in start_frames
        ]
    ).flatten()
    return pd.DataFrame({"zeta_cg": zeta_cg_mean, "delta_r": delta_r})


def LSI_atom(distances: ArrayLike) -> NDArray:
    r_j = distances[distances <= 0.37]
    r_j = r_j.tolist()
    r_j.append(distances[len(r_j)])
    delta_ji = [r_j[i + 1] - r_j[i] for i in range(0, len(r_j) - 1)]
    mean_delta_i = np.mean(delta_ji)
    I = 1 / len(delta_ji) * np.sum((mean_delta_i - delta_ji) ** 2)
    return I


def LSI(
    trajectory: Coordinates, segments: int = 10000, skip: float = 0
) -> pd.DataFrame:
    def LSI_distribution(
        frame: CoordinateFrame, bins: NDArray, selector: Optional[Callable] = None
    ) -> NDArray:
        atoms_PBC = pbc_points(frame, frame.box, thickness=0.7)
        atoms_tree = KDTree(atoms_PBC)
        if selector:
            index = selector(frame)
        else:
            index = np.arange(len(frame))
        dist, _ = atoms_tree.query(frame[index], 50, distance_upper_bound=0.6)
        distances = dist[:, 1:]
        LSI_values = np.array([LSI_atom(distance) for distance in distances])
        dist = np.histogram(LSI_values, bins=bins, density=True)[0]
        return dist

    bins = np.linspace(0, 0.007, 201)
    I = bins[1:] - (bins[1] - bins[0]) / 2

    frame_indices = np.unique(
        np.int_(np.linspace(len(trajectory) * skip, len(trajectory) - 1, num=segments))
    )
    distributions = np.array(
        [
            LSI_distribution(trajectory[frame_index], bins, selector=None)
            for frame_index in frame_indices
        ]
    )
    P = np.mean(distributions, axis=0)
    return pd.DataFrame({"I": I, "P": P})


def HDL_LDL_positions(
    frame: CoordinateFrame, selector: Optional[Callable] = None
) -> Tuple[NDArray, NDArray]:
    atoms_PBC = pbc_points(frame, frame.box, thickness=0.7)
    atoms_tree = KDTree(atoms_PBC)
    if selector:
        index = selector(frame)
    else:
        index = range(len(frame))
    dist = atoms_tree.query(frame[index], 50, distance_upper_bound=0.6)[0]
    distances = dist[:, 1:]
    LSI_values = np.array([LSI_atom(distance) for distance in distances])
    LDL = LSI_values >= 0.0013
    HDL = LSI_values < 0.0013
    pos_HDL = frame[index][HDL]
    pos_LDL = frame[index][LDL]
    return pos_HDL, pos_LDL


def HDL_LDL_gr(
    trajectory: Coordinates, segments: int = 10000, skip: float = 0.1
) -> pd.DataFrame:
    def gr_frame(
        frame: CoordinateFrame, trajectory: Coordinates, bins: ArrayLike
    ) -> NDArray:
        atoms_ALL = frame
        atoms_HDL, atoms_LDL = HDL_LDL_positions(frame, trajectory)

        atoms_PBC_ALL = pbc_points(atoms_ALL, frame.box)
        atoms_PBC_LDL = pbc_points(atoms_LDL, frame.box)
        atoms_PBC_HDL = pbc_points(atoms_HDL, frame.box)

        tree_ALL = KDTree(atoms_PBC_ALL)
        tree_LDL = KDTree(atoms_PBC_LDL)
        tree_HDL = KDTree(atoms_PBC_HDL)

        dist_ALL_ALL, _ = tree_ALL.query(
            atoms_ALL, len(frame) // 2, distance_upper_bound=bins[-1] + 0.1
        )
        dist_HDL_HDL, _ = tree_HDL.query(
            atoms_HDL, len(frame) // 2, distance_upper_bound=bins[-1] + 0.1
        )
        dist_LDL_LDL, _ = tree_LDL.query(
            atoms_LDL, len(frame) // 2, distance_upper_bound=bins[-1] + 0.1
        )
        dist_HDL_LDL, _ = tree_LDL.query(
            atoms_HDL, len(frame) // 2, distance_upper_bound=bins[-1] + 0.1
        )

        dist_ALL_ALL = dist_ALL_ALL[:, 1:].flatten()
        dist_HDL_HDL = dist_HDL_HDL[:, 1:].flatten()
        dist_LDL_LDL = dist_LDL_LDL[:, 1:].flatten()
        dist_HDL_LDL = dist_HDL_LDL.flatten()

        hist_ALL_ALL = np.histogram(
            dist_ALL_ALL, bins=bins, range=(0, bins[-1]), density=False
        )[0]
        hist_HDL_HDL = np.histogram(
            dist_HDL_HDL, bins=bins, range=(0, bins[-1]), density=False
        )[0]
        hist_LDL_LDL = np.histogram(
            dist_LDL_LDL, bins=bins, range=(0, bins[-1]), density=False
        )[0]
        hist_HDL_LDL = np.histogram(
            dist_HDL_LDL, bins=bins, range=(0, bins[-1]), density=False
        )[0]

        return np.array(
            [
                hist_ALL_ALL / len(atoms_ALL),
                hist_HDL_HDL / len(atoms_HDL),
                hist_LDL_LDL / len(atoms_LDL),
                hist_HDL_LDL / len(atoms_HDL),
            ]
        )

    start_frame = trajectory[int(len(trajectory) * skip)]
    upper_bound = round(np.min(np.diag(start_frame.box)) / 2 - 0.05, 1)
    bins = np.linspace(0, upper_bound, upper_bound * 500 + 1)
    frame_indices = np.unique(
        np.int_(np.linspace(len(trajectory) * skip, len(trajectory) - 1, num=segments))
    )

    gr = []
    for frame_index in frame_indices:
        hists = gr_frame(trajectory[frame_index], trajectory, bins)
        gr.append(hists)

    gr = np.mean(gr, axis=0)
    gr = gr / (4 / 3 * np.pi * bins[1:] ** 3 - 4 / 3 * np.pi * bins[:-1] ** 3)
    r = bins[1:] - (bins[1] - bins[0]) / 2

    return pd.DataFrame(
        {"r": r, "gr_ALL": [0], "gr_HDL": gr[1], "gr_LDL": gr[2], "gr_MIX": gr[3]}
    )


def HDL_LDL_concentration(
    trajectory: Coordinates, segments: int = 10000, skip: float = 0.1
) -> pd.DataFrame:
    def HDL_LDL_concentration_frame(
        frame: CoordinateFrame, bins: ArrayLike
    ) -> Tuple[NDArray, NDArray]:
        atoms_HDL, atoms_LDL = HDL_LDL_positions(frame, trajectory)
        atoms_PBC_HDL = pbc_points(atoms_HDL, frame.box, thickness=0.61)
        atoms_PBC_LDL = pbc_points(atoms_LDL, frame.box, thickness=0.61)
        tree_LDL = KDTree(atoms_PBC_LDL)
        tree_HDL = KDTree(atoms_PBC_HDL)
        dist_HDL_HDL, _ = tree_HDL.query(atoms_HDL, 31, distance_upper_bound=0.6)
        dist_HDL_LDL, _ = tree_LDL.query(atoms_HDL, 30, distance_upper_bound=0.6)
        HDL_near_HDL = np.sum(
            dist_HDL_HDL <= 0.5, axis=-1
        )  # Ausgangsteilchen dazu zählen
        LDL_near_HDL = np.sum(dist_HDL_LDL <= 0.5, axis=-1)
        x_HDL = HDL_near_HDL / (HDL_near_HDL + LDL_near_HDL)
        x_HDL_dist = np.histogram(x_HDL, bins=bins, range=(0, bins[-1]), density=True)[
            0
        ]
        dist_LDL_LDL, _ = tree_LDL.query(atoms_LDL, 31, distance_upper_bound=0.6)
        dist_LDL_HDL, _ = tree_HDL.query(atoms_LDL, 30, distance_upper_bound=0.6)
        LDL_near_LDL = np.sum(
            dist_LDL_LDL <= 0.5, axis=-1
        )  # Ausgangsteilchen dazu zählen
        HDL_near_LDL = np.sum(dist_LDL_HDL <= 0.5, axis=-1)
        x_LDL = LDL_near_LDL / (LDL_near_LDL + HDL_near_LDL)
        x_LDL_dist = np.histogram(x_LDL, bins=bins, range=(0, bins[-1]), density=True)[
            0
        ]
        return x_HDL_dist, x_LDL_dist

    bins = np.linspace(0, 1, 21)
    x = bins[1:] - (bins[1] - bins[0]) / 2
    frame_indices = np.unique(
        np.int_(np.linspace(len(trajectory) * skip, len(trajectory) - 1, num=segments))
    )
    local_concentration_dist = np.array(
        [
            HDL_LDL_concentration_frame(trajectory[frame_index], trajectory, bins)
            for frame_index in frame_indices
        ]
    )
    x_HDL = np.mean(local_concentration_dist[:, 0], axis=0)
    x_LDL = np.mean(local_concentration_dist[:, 1], axis=0)
    return pd.DataFrame({"x": x, "x_HDL": x_HDL, "x_LDL": x_LDL})
