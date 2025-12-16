from typing import Callable, Optional, Union, Tuple, List

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import spatial
from scipy.spatial import KDTree
from scipy.sparse.csgraph import connected_components

from .coordinates import (
    rotate_axis,
    polar_coordinates,
    Coordinates,
    CoordinateFrame,
    next_neighbors,
    number_of_neighbors,
)
from .autosave import autosave_data
from .pbc import pbc_diff, pbc_points


@autosave_data(nargs=2, kwargs_keys=("coordinates_b",))
def time_average(
    function: Callable,
    coordinates: Coordinates,
    coordinates_b: Optional[Coordinates] = None,
    skip: float = 0.1,
    segments: int = 100,
) -> NDArray:
    """
    Compute the time average of a function.

    Args:
        function:
            The function that will be averaged, it has to accept exactly one argument
            which is the current atom set (or two if coordinates_b is provided)
        coordinates: The coordinates object of the simulation
        coordinates_b: Additional coordinates object of the simulation
        skip:
        segments:
    """
    frame_indices = np.unique(
        np.int_(
            np.linspace(len(coordinates) * skip, len(coordinates) - 1, num=segments)
        )
    )
    if coordinates_b is None:
        result = [function(coordinates[frame_index]) for frame_index in frame_indices]
    else:
        result = [
            function(coordinates[frame_index], coordinates_b[frame_index])
            for frame_index in frame_indices
        ]
    return np.mean(result, axis=0)


@autosave_data(nargs=2, kwargs_keys=("coordinates_b",))
def time_distribution(
    function: Callable,
    coordinates: Coordinates,
    coordinates_b: Optional[Coordinates] = None,
    skip: float = 0,
    segments: int = 100,
) -> Tuple[NDArray, List]:
    """
    Compute the time distribution of a function.

    Args:
        function:
            The function that will be averaged, it has to accept exactly one argument
            which is the current atom set (or two if coordinates_b is provided)
        coordinates: The coordinates object of the simulation
        coordinates_b: Additional coordinates object of the simulation
        skip:
        segments:
    """
    frame_indices = np.unique(
        np.int_(
            np.linspace(len(coordinates) * skip, len(coordinates) - 1, num=segments)
        )
    )
    times = np.array([coordinates[frame_index].time for frame_index in frame_indices])
    if coordinates_b is None:
        result = [function(coordinates[frame_index]) for frame_index in frame_indices]
    else:
        result = [
            function(coordinates[frame_index], coordinates_b[frame_index])
            for frame_index in frame_indices
        ]
    return times, result


def rdf(
    atoms_a: CoordinateFrame,
    atoms_b: Optional[CoordinateFrame] = None,
    bins: Optional[ArrayLike] = None,
    mode: str = 'total',
    **kwargs
) -> NDArray:
    r"""
    Compute the radial pair distribution of one or two sets of atoms.

    .. math::
       g_{AB}(r) = \frac{1}{\langle \rho_B\rangle N_A}\sum\limits_{i\in A}^{N_A}
       \sum\limits_{j\in B}^{N_B}\frac{\delta(r_{ij} -r)}{4\pi r^2}

    For use with :func:`time_average`, define bins through the use of
    :func:`~functools.partial`, the atom sets are passed to :func:`time_average`, if a
    second set of atoms should be used specify it as ``coordinates_b`` and it will be
    passed to this function.

    Args:
        atoms_a: First set of atoms, used internally
        atoms_b (opt.): Second set of atoms, used internal
        bins: Bins of the radial distribution function
        mode: 'Total' | 'Intra' | 'Inter'
    """
    analysis_mode = mode

    kwargs.pop("mode", None)

    distinct = True
    if atoms_b is None:
        atoms_b = atoms_a
        distinct = False
    elif np.array_equal(atoms_a, atoms_b):
        distinct = False
    if bins is None:
        bins = np.arange(0, 1, 0.01)

    particles_in_volume = int(
        np.max(number_of_neighbors(atoms_a, query_atoms=atoms_b, r_max=bins[-1])) * 1.1
    )
    distances, indices = next_neighbors(
        atoms_a,
        atoms_b,
        number_of_neighbors=particles_in_volume,
        distance_upper_bound=bins[-1],
        distinct=distinct,
        **kwargs
    )

    if analysis_mode == "inter":
        new_distances = []
        for entry in list(zip(atoms_a.residue_ids, distances, indices)):
            mask = entry[1] < np.inf
            new_distances.append(
                entry[1][mask][atoms_b.residue_ids[entry[2][mask]] != entry[0]]
            )
        distances = np.concatenate(new_distances)

    elif analysis_mode == "intra":
        new_distances = []
        for entry in zip(atoms_a.residue_ids, distances, indices):
            mask = entry[1] < np.inf
            atom_a_resid = entry[0]
            valid_indices = entry[2][mask]
            intra_mask = atoms_b.residue_ids[valid_indices] == atom_a_resid
            new_distances.append(entry[1][mask][intra_mask])
        distances = np.concatenate(new_distances)

    else:
        distances = [d for dist in distances for d in dist]

    hist, bins = np.histogram(distances, bins=bins, range=(0, bins[-1]), density=False)
    hist = hist / len(atoms_b)
    hist = hist / (4 / 3 * np.pi * bins[1:] ** 3 - 4 / 3 * np.pi * bins[:-1] ** 3)
    n = len(atoms_a) / np.prod(np.diag(atoms_a.box))
    hist = hist / n

    return hist


def distance_distribution(
    atoms: CoordinateFrame, bins: Union[int, ArrayLike]
) -> NDArray:
    connection_vectors = atoms[:-1, :] - atoms[1:, :]
    connection_lengths = (connection_vectors**2).sum(axis=1) ** 0.5
    return np.histogram(connection_lengths, bins)[0]


def tetrahedral_order(
    atoms: CoordinateFrame, reference_atoms: CoordinateFrame = None
) -> NDArray:
    if reference_atoms is None:
        reference_atoms = atoms
    indices = next_neighbors(
        reference_atoms,
        query_atoms=atoms,
        number_of_neighbors=4,
    )[1]
    neighbors = reference_atoms[indices]
    neighbors_1, neighbors_2, neighbors_3, neighbors_4 = (
        neighbors[:, 0, :],
        neighbors[:, 1, :],
        neighbors[:, 2, :],
        neighbors[:, 3, :],
    )

    # Connection vectors
    neighbors_1 -= atoms
    neighbors_2 -= atoms
    neighbors_3 -= atoms
    neighbors_4 -= atoms

    # Normed Connection vectors
    neighbors_1 /= np.linalg.norm(neighbors_1, axis=-1).reshape(-1, 1)
    neighbors_2 /= np.linalg.norm(neighbors_2, axis=-1).reshape(-1, 1)
    neighbors_3 /= np.linalg.norm(neighbors_3, axis=-1).reshape(-1, 1)
    neighbors_4 /= np.linalg.norm(neighbors_4, axis=-1).reshape(-1, 1)

    a_1_2 = ((neighbors_1 * neighbors_2).sum(axis=1) + 1 / 3) ** 2
    a_1_3 = ((neighbors_1 * neighbors_3).sum(axis=1) + 1 / 3) ** 2
    a_1_4 = ((neighbors_1 * neighbors_4).sum(axis=1) + 1 / 3) ** 2

    a_2_3 = ((neighbors_2 * neighbors_3).sum(axis=1) + 1 / 3) ** 2
    a_2_4 = ((neighbors_2 * neighbors_4).sum(axis=1) + 1 / 3) ** 2

    a_3_4 = ((neighbors_3 * neighbors_4).sum(axis=1) + 1 / 3) ** 2

    q = 1 - 3 / 8 * (a_1_2 + a_1_3 + a_1_4 + a_2_3 + a_2_4 + a_3_4)

    return q


def tetrahedral_order_distribution(
    atoms: CoordinateFrame,
    reference_atoms: Optional[CoordinateFrame] = None,
    bins: Optional[ArrayLike] = None,
) -> NDArray:
    assert bins is not None, "Bin edges of the distribution have to be specified."
    Q = tetrahedral_order(atoms, reference_atoms=reference_atoms)
    return np.histogram(Q, bins=bins)[0]


def radial_density(
    atoms: CoordinateFrame,
    bins: Optional[ArrayLike] = None,
    symmetry_axis: ArrayLike = (0, 0, 1),
    origin: Optional[ArrayLike] = None,
) -> NDArray:
    """
    Calculate the radial density distribution.

    This function is meant to be used with time_average.

    Args:
        atoms:
            Set of coordinates.
        bins (opt.):
            Bin specification that is passed to numpy.histogram. This needs to be
            a list of bin edges if the function is used within time_average.
        symmetry_axis (opt.):
            Vector of the symmetry axis, around which the radial density is calculated,
            default is z-axis.
        origin (opt.):
            Origin of the rotational symmetry, e.g. center of the pore.
    """
    if origin is None:
        origin = np.diag(atoms.box) / 2
    if bins is None:
        bins = np.arange(0, np.min(np.diag(atoms.box) / 2), 0.01)
    length = np.diag(atoms.box) * symmetry_axis
    cartesian = rotate_axis(atoms - origin, symmetry_axis)
    radius, _ = polar_coordinates(cartesian[:, 0], cartesian[:, 1])
    hist = np.histogram(radius, bins=bins)[0]
    volume = np.pi * (bins[1:] ** 2 - bins[:-1] ** 2) * np.linalg.norm(length)
    return hist / volume


def shell_density(
    atoms: CoordinateFrame,
    shell_radius: float,
    bins: ArrayLike,
    shell_thickness: float = 0.5,
    symmetry_axis: ArrayLike = (0, 0, 1),
    origin: Optional[ArrayLike] = None,
) -> NDArray:
    """
    Compute the density distribution on a cylindrical shell.

    Args:
        atoms: The coordinates of the atoms
        shell_radius: Inner radius of the shell
        bins: Histogramm bins, this has to be a two-dimensional list of bins: [angle, z]
        shell_thickness (opt.): Thicknes of the shell, default is 0.5
        symmetry_axis (opt.): The symmtery axis of the pore, the coordinates will be
            rotated such that this axis is the z-axis
        origin (opt.): Origin of the pore, the coordinates will be moved such that this
            is the new origin.

    Returns:
        Two-dimensional density distribution of the atoms in the defined shell.
    """
    if origin is None:
        origin = np.diag(atoms.box) / 2
    cartesian = rotate_axis(atoms - origin, symmetry_axis)
    radius, theta = polar_coordinates(cartesian[:, 0], cartesian[:, 1])
    shell_indices = (shell_radius <= radius) & (
        radius <= shell_radius + shell_thickness
    )
    hist = np.histogram2d(theta[shell_indices], cartesian[shell_indices, 2], bins)[0]

    return hist


def next_neighbor_distribution(
    atoms: CoordinateFrame,
    reference: Optional[CoordinateFrame] = None,
    number_of_neighbors: int = 4,
    bins: Optional[ArrayLike] = None,
    normed: bool = True,
) -> NDArray:
    """
    Compute the distribution of next neighbors with the same residue name.
    """
    assert bins is not None, "Bins have to be specified."
    if reference is None:
        reference = atoms
    nn = next_neighbors(
        reference, query_atoms=atoms, number_of_neighbors=number_of_neighbors
    )[1]
    resname_nn = reference.residue_names[nn]
    count_nn = (resname_nn == atoms.residue_names.reshape(-1, 1)).sum(axis=1)
    return np.histogram(count_nn, bins=bins, density=normed)[0]


def hbonds(
    atoms: CoordinateFrame,
    donator_indices: ArrayLike,
    hydrogen_indices: ArrayLike,
    acceptor_indices: ArrayLike,
    DA_lim: float = 0.35,
    HA_lim: float = 0.35,
    max_angle_deg: float = 30,
    full_output: bool = False,
) -> Union[NDArray, tuple[NDArray, NDArray, NDArray]]:
    """
    Compute h-bond pairs

    Args:
        atoms: Set of all coordinates for a frame.
        donator_indices: Set of indices for donators.
        hydrogen_indices: Set of indices for hydrogen atoms. Should have the same
            length as D.
        acceptor_indices:  Set of indices for acceptors.
        DA_lim (opt.): Minimum distance between donator and acceptor.
        HA_lim (opt.): Minimum distance between hydrogen and acceptor.
        max_angle_deg (opt.): Maximum angle in degree for the HDA angle.
        full_output (opt.): Returns additionally the cosine of the
        angles and the DA distances

    Return:
        List of (D,A)-pairs in hbonds.
    """

    def dist_DltA(
        D: CoordinateFrame, A: CoordinateFrame, max_dist: float = 0.35
    ) -> NDArray:
        ppoints, pind = pbc_points(D, thickness=max_dist + 0.1, index=True)
        Dtree = spatial.cKDTree(ppoints)
        Atree = spatial.cKDTree(A)
        pairs = Dtree.sparse_distance_matrix(Atree, max_dist, output_type="ndarray")
        pairs = np.asarray(pairs.tolist())
        pairs = np.int_(pairs[pairs[:, 2] > 0][:, :2])
        pairs[:, 0] = pind[pairs[:, 0]]
        return pairs

    def dist_AltD(
        D: CoordinateFrame, A: CoordinateFrame, max_dist: float = 0.35
    ) -> NDArray:
        ppoints, pind = pbc_points(A, thickness=max_dist + 0.1, index=True)
        Atree = spatial.cKDTree(ppoints)
        Dtree = spatial.cKDTree(D)
        pairs = Atree.sparse_distance_matrix(Dtree, max_dist, output_type="ndarray")
        pairs = np.asarray(pairs.tolist())
        pairs = np.int_(pairs[pairs[:, 2] > 0][:, :2])
        pairs = pairs[:, ::-1]
        pairs[:, 1] = pind[pairs[:, 1]]
        return pairs

    D = atoms[donator_indices]
    H = atoms[hydrogen_indices]
    A = atoms[acceptor_indices]
    min_cos = np.cos(max_angle_deg * np.pi / 180)
    box = D.box
    if len(D) <= len(A):
        pairs = dist_DltA(D, A, DA_lim)
    else:
        pairs = dist_AltD(D, A, DA_lim)

    vDH = pbc_diff(D[pairs[:, 0]], H[pairs[:, 0]], box)
    vDA = pbc_diff(D[pairs[:, 0]], A[pairs[:, 1]], box)
    vHA = pbc_diff(H[pairs[:, 0]], A[pairs[:, 1]], box)
    angles_cos = np.clip(
        np.einsum("ij,ij->i", vDH, vDA)
        / np.linalg.norm(vDH, axis=1)
        / np.linalg.norm(vDA, axis=1),
        -1,
        1,
    )
    is_bond = (
        (angles_cos >= min_cos)
        & (np.sum(vHA**2, axis=-1) <= HA_lim**2)
        & (np.sum(vDA**2, axis=-1) <= DA_lim**2)
    )
    if full_output:
        return (
            pairs[is_bond],
            angles_cos[is_bond],
            np.sum(vDA[is_bond] ** 2, axis=-1) ** 0.5,
        )
    else:
        return pairs[is_bond]


def calc_cluster_sizes(atoms: CoordinateFrame, r_max: float = 0.35) -> NDArray:
    frame_PBC, indices_PBC = pbc_points(atoms, thickness=r_max + 0.1, index=True)
    tree = KDTree(frame_PBC)
    matrix = tree.sparse_distance_matrix(tree, r_max, output_type="ndarray")
    new_matrix = np.zeros((len(atoms), len(atoms)))
    for entry in matrix:
        if entry[2] > 0:
            new_matrix[indices_PBC[entry[0]], indices_PBC[entry[1]]] = 1
    n_components, labels = connected_components(new_matrix, directed=False)
    cluster_sizes = []
    for i in range(0, np.max(labels) + 1):
        cluster_sizes.append(np.sum(labels == i))
    return np.array(cluster_sizes).flatten()


def gyration_radius(position: CoordinateFrame) -> NDArray:
    r"""
    Calculates a list of all radii of gyration of all molecules given in the coordinate
    frame, weighted with the masses of the individual atoms.

    Args:
        position: Coordinate frame object

    ..math::
        R_G = \left(\frac{\sum_{i=1}^{n} m_i |\vec{r_i}
              - \vec{r_{COM}}|^2 }{\sum_{i=1}^{n} m_i }
        \rigth)^{\frac{1}{2}}
    """
    gyration_radii = np.array([])

    for resid in np.unique(position.residue_ids):
        pos = position.whole[position.residue_ids == resid]
        mass = position.masses[position.residue_ids == resid][:, np.newaxis]
        COM = 1 / mass.sum() * (mass * position).sum(axis=0)
        r_sq = ((pbc_diff(pos, COM, pos.box.diagonal())) ** 2).sum(1)[:, np.newaxis]
        g_radius = ((r_sq * mass).sum() / mass.sum()) ** 0.5

        gyration_radii = np.append(gyration_radii, g_radius)

    return gyration_radii
