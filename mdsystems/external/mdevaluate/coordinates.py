from functools import partial, wraps
from copy import copy
from .logging import logger
from typing import Optional, Callable, List, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial import KDTree

from .atoms import AtomSubset
from .pbc import whole, nojump, pbc_diff, pbc_points
from .utils import singledispatchmethod
from .checksum import checksum


class UnknownCoordinatesMode(Exception):
    pass


class CoordinateFrame(NDArray):
    _known_modes = ("pbc", "whole", "nojump")

    @property
    def box(self):
        return np.array(self.coordinates.frames[self.step].triclinic_dimensions)

    @property
    def volume(self):
        return self.box.diagonal().cumprod()[-1]

    @property
    def time(self):
        return self.coordinates.frames[self.step].time

    @property
    def masses(self):
        return self.coordinates.atoms.masses[self.coordinates.atom_subset.selection]

    @property
    def charges(self):
        return self.coordinates.atoms.charges[self.coordinates.atom_subset.selection]

    @property
    def residue_ids(self):
        return self.coordinates.atom_subset.residue_ids

    @property
    def residue_names(self):
        return self.coordinates.atom_subset.residue_names

    @property
    def atom_names(self):
        return self.coordinates.atom_subset.atom_names

    @property
    def indices(self):
        return self.coordinates.atom_subset.indices

    @property
    def selection(self):
        return self.coordinates.atom_subset.selection

    @property
    def whole(self):
        frame = whole(self)
        frame.mode = "whole"
        return frame

    @property
    def pbc(self):
        frame = self % self.box.diagonal()
        frame.mode = "pbc"
        return frame

    @property
    def nojump(self):
        if self.mode != "nojump":
            if self.mode is not None:
                logger.warn(
                    "Combining Nojump with other Coordinate modes is not supported and "
                    "may cause unexpected results."
                )
            frame = nojump(self)
            frame.mode = "nojump"
            return frame
        else:
            return self

    def __new__(
        subtype,
        shape,
        dtype=float,
        buffer=None,
        offset=0,
        strides=None,
        order=None,
        coordinates=None,
        step=None,
        box=None,
        mode=None,
    ):
        obj = NDArray.__new__(subtype, shape, dtype, buffer, offset, strides)

        obj.coordinates = coordinates
        obj.step = step
        obj.mode = mode
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return

        self.coordinates = getattr(obj, "coordinates", None)
        self.step = getattr(obj, "step", None)
        self.mode = getattr(obj, "mode", None)
        if hasattr(obj, "reference"):
            self.reference = getattr(obj, "reference")


class Coordinates:
    """
    Coordinates represent trajectory data, which is used for evaluation functions.

    Atoms may be selected by specifying an atom_subset or an atom_filter.
    """

    def get_mode(self, mode):
        if self.atom_subset is not None:
            return Coordinates(
                frames=self.frames, atom_subset=self.atom_subset, mode=mode
            )[self._slice]
        else:
            return Coordinates(
                frames=self.frames, atom_filter=self.atom_filter, mode=mode
            )[self._slice]

    @property
    def pbc(self):
        return self.get_mode("pbc")

    @property
    def whole(self):
        return self.get_mode("whole")

    @property
    def nojump(self):
        return self.get_mode("nojump")

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, val):
        if val in CoordinateFrame._known_modes:
            logger.warn(
                "Changing the Coordinates mode directly is deprecated. "
                "Use Coordinates.%s instead, which returns a copy.",
                val,
            )
            self._mode = val
        else:
            raise UnknownCoordinatesMode("No such mode: {}".format(val))

    def __init__(
        self, frames, atom_filter=None, atom_subset: AtomSubset = None, mode=None
    ):
        """
        Args:
            frames: The trajectory reader
            atom_filter (opt.): A mask which selects a subset of the system
            atom_subset (opt.): A AtomSubset that selects a subset of the system
            mode (opt.): PBC mode of the Coordinates, can be pbc, whole or nojump.

        Note:
            The caching in Coordinates is deprecated, use the CachedReader or the function open
            from the reader module instead.
        """
        self._mode = mode
        self.frames = frames
        self._slice = slice(None)
        assert (
            atom_filter is None or atom_subset is None
        ), "Cannot use both: subset and filter"

        if atom_filter is not None:
            self.atom_filter = atom_filter
            self.atom_subset = None
        elif atom_subset is not None:
            self.atom_filter = atom_subset.selection
            self.atom_subset = atom_subset
            self.atoms = atom_subset.atoms
        else:
            self.atom_filter = np.ones(shape=(len(frames[0].coordinates),), dtype=bool)
            self.atom_subset = None

    def get_frame(self, fnr):
        """Returns the fnr-th frame."""
        try:
            if self.atom_filter is not None:
                frame = (
                    self.frames[fnr].positions[self.atom_filter].view(CoordinateFrame)
                )
            else:
                frame = self.frames.__getitem__(fnr).positions.view(CoordinateFrame)
            frame.coordinates = self
            frame.step = fnr
            if self.mode is not None:
                frame = getattr(frame, self.mode)
        except EOFError:
            raise IndexError

        return frame

    def clear_cache(self):
        """Clears the frame cache, if it is enabled."""
        if hasattr(self.get_frame, "clear_cache"):
            self.get_frame.clear_cache()

    def __iter__(self):
        for i in range(len(self.frames))[self._slice]:
            yield self[i]

    @singledispatchmethod
    def __getitem__(self, item):
        return self.get_frame(item)

    @__getitem__.register(slice)
    def _(self, item):
        sliced = copy(self)
        sliced._slice = item
        return sliced

    def __len__(self):
        return len(self.frames[self._slice])

    def __checksum__(self):
        return checksum(self.frames, self.atom_filter, self._slice, self.mode)

    def __repr__(self):
        return "Coordinates <{}>: {}".format(self.frames.filename, self.atom_subset)

    @wraps(AtomSubset.subset)
    def subset(self, **kwargs):
        return Coordinates(
            self.frames, atom_subset=self.atom_subset.subset(**kwargs), mode=self._mode
        )

    @property
    def description(self):
        return self.atom_subset.description

    @description.setter
    def description(self, desc):
        self.atom_subset.description = desc


class CoordinatesMap:
    def __init__(self, coordinates, function):
        self.coordinates = coordinates
        self.frames = self.coordinates.frames
        self.atom_subset = self.coordinates.atom_subset
        self.function = function
        self._slice = slice(None)
        if isinstance(function, partial):
            self._description = self.function.func.__name__
        else:
            self._description = self.function.__name__
        self._slice = slice(None)

    def __iter__(self):
        for frame in self.coordinates:
            step = frame.step
            frame = self.function(frame)
            if not isinstance(frame, CoordinateFrame):
                frame = frame.view(CoordinateFrame)
                frame.coordinates = self
                frame.step = step
            yield frame

    def __getitem__(self, item):
        if isinstance(item, slice):
            return self.__class__(self.coordinates[item], self.function)
        else:
            frame = self.function(self.coordinates.__getitem__(item))
            if not isinstance(frame, CoordinateFrame):
                frame = frame.view(CoordinateFrame)
            frame.coordinates = self
            frame.step = item
            return frame

    def __len__(self):
        return len(self.coordinates.frames)

    def __checksum__(self):
        return checksum(self.coordinates, self.function)

    @wraps(Coordinates.subset)
    def subset(self, **kwargs):
        return CoordinatesMap(self.coordinates.subset(**kwargs), self.function)

    @property
    def description(self):
        return "{}_{}".format(self._description, self.coordinates.description)

    @description.setter
    def description(self, desc):
        self._description = desc

    @property
    def nojump(self):
        return CoordinatesMap(self.coordinates.nojump, self.function)

    @property
    def whole(self):
        return CoordinatesMap(self.coordinates.whole, self.function)

    @property
    def pbc(self):
        return CoordinatesMap(self.coordinates.pbc, self.function)


def rotate_axis(coords: ArrayLike, axis: ArrayLike) -> NDArray:
    """
    Rotate a set of coordinates to a given axis.
    """
    axis = np.array(axis) / np.linalg.norm(axis)
    zaxis = np.array([0, 0, 1])
    if (axis == zaxis).sum() == 3:
        return coords
    rotation_axis = np.cross(axis, zaxis)
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)

    theta = np.arccos(axis @ zaxis / np.linalg.norm(axis))

    # return theta/pi, rotation_axis

    ux, uy, uz = rotation_axis
    cross_matrix = np.array([[0, -uz, uy], [uz, 0, -ux], [-uy, ux, 0]])
    rotation_matrix = (
        np.cos(theta) * np.identity(len(axis))
        + (1 - np.cos(theta))
        * rotation_axis.reshape(-1, 1)
        @ rotation_axis.reshape(1, -1)
        + np.sin(theta) * cross_matrix
    )

    if len(coords.shape) == 2:
        rotated = np.array([rotation_matrix @ xyz for xyz in coords])
    else:
        rotated = rotation_matrix @ coords
    return rotated


def spherical_radius(
    frame: CoordinateFrame, origin: Optional[ArrayLike] = None
) -> NDArray:
    """
    Transform a frame of cartesian coordinates into the spherical radius.
    If origin=None, the center of the box is taken as the coordinates' origin.
    """
    if origin is None:
        origin = frame.box.diagonal() / 2
    return ((frame - origin) ** 2).sum(axis=-1) ** 0.5


def polar_coordinates(x: ArrayLike, y: ArrayLike) -> (NDArray, NDArray):
    """Convert cartesian to polar coordinates."""
    radius = (x**2 + y**2) ** 0.5
    phi = np.arctan2(y, x)
    return radius, phi


def spherical_coordinates(
    x: ArrayLike, y: ArrayLike, z: ArrayLike
) -> (NDArray, NDArray, NDArray):
    """Convert cartesian to spherical coordinates."""
    xy, phi = polar_coordinates(x, y)
    radius = (x**2 + y**2 + z**2) ** 0.5
    theta = np.arccos(z / radius)
    return radius, phi, theta


def selector_radial_cylindrical(
    atoms: CoordinateFrame,
    r_min: float,
    r_max: float,
    origin: Optional[ArrayLike] = None,
) -> NDArray:
    box = atoms.box
    atoms = atoms % np.diag(box)
    if origin is None:
        origin = [box[0, 0] / 2, box[1, 1] / 2, box[2, 2] / 2]
    r_vec = (atoms - origin)[:, :2]
    r = np.linalg.norm(r_vec, axis=1)
    index = np.argwhere((r >= r_min) * (r < r_max))
    return index.flatten()

def map_coordinates(
    func: Callable[[CoordinateFrame, ...], NDArray]
) -> Callable[..., CoordinatesMap]:
    @wraps(func)
    def wrapped(coordinates: Coordinates, **kwargs) -> CoordinatesMap:
        return CoordinatesMap(coordinates, partial(func, **kwargs))

    return wrapped


@map_coordinates
def center_of_masses(
    frame: CoordinateFrame, atom_indices=None, shear: bool = False
) -> NDArray:
    if atom_indices is None:
        atom_indices = list(range(len(frame)))
    res_ids = frame.residue_ids[atom_indices]
    masses = frame.masses[atom_indices]
    if shear:
        coords = frame[atom_indices]
        box = frame.box
        sort_ind = res_ids.argsort(kind="stable")
        i = np.concatenate([[0], np.where(np.diff(res_ids[sort_ind]) > 0)[0] + 1])
        coms = coords[sort_ind[i]][res_ids - min(res_ids)]
        cor = pbc_diff(coords, coms, box)
        coords = coms + cor
    else:
        coords = frame.whole[atom_indices]
    mask = np.bincount(res_ids)[1:] != 0
    positions = np.array(
        [
            np.bincount(res_ids, weights=c * masses)[1:]
            / np.bincount(res_ids, weights=masses)[1:]
            for c in coords.T
        ]
    ).T[mask]
    return np.array(positions)


@map_coordinates
def pore_coordinates(
    frame: CoordinateFrame, origin: ArrayLike, sym_axis: str = "z"
) -> NDArray:
    """
    Map coordinates of a pore simulation so the pore has cylindrical symmetry.

    Args:
        frame: Coordinates of the simulation
        origin: Origin of the pore which will be the coordinates origin after mapping
        sym_axis (opt.): Symmtery axis of the pore, may be a literal direction
            'x', 'y' or 'z' or an array of shape (3,)
    """
    if sym_axis in ("x", "y", "z"):
        rot_axis = np.zeros(shape=(3,))
        rot_axis[["x", "y", "z"].index(sym_axis)] = 1
    else:
        rot_axis = sym_axis
    return rotate_axis(frame - origin, rot_axis)


@map_coordinates
def vectors(
    frame: CoordinateFrame,
    atom_indices_a: ArrayLike,
    atom_indices_b: ArrayLike,
    normed: bool = False,
) -> NDArray:
    """
    Compute the vectors between the atoms of two subsets.

    Args:
        frame: The Coordinates object the atoms will be taken from
        atom_indices_a: Mask or indices of the first atom subset
        atom_indices_b: Mask or indices of the second atom subset
        normed (opt.): If the vectors should be normed

    The definition of atoms_a/b can be any possible subript of a numpy array.
    They can, for example, be given as a masking array of bool values with the
    same length as the frames of the coordinates.
    Or there can be a list of indices selecting the atoms of these indices from each
    frame.

    It is possible to compute the means of several atoms before calculating the vectors,
    by using a two-dimensional list of indices. The following code computes the vectors
    between atoms 0, 3, 6 and the mean coordinate of atoms 1, 4, 7 and 2, 5, 8:

        >>> inds_a = [0, 3, 6]
        >>> inds_b = [[1, 4, 7], [2, 5, 8]]
        >>> vectors(coords, inds_a, inds_b)
        array([
            coords[0] - (coords[1] + coords[2])/2,
            coords[3] - (coords[4] + coords[5])/2,
            coords[6] - (coords[7] + coords[8])/2,
        ])
    """
    box = frame.box
    coords_a = frame[atom_indices_a]
    if len(coords_a.shape) > 2:
        coords_a = coords_a.mean(axis=0)
    coords_b = frame[atom_indices_b]
    if len(coords_b.shape) > 2:
        coords_b = coords_b.mean(axis=0)
    vec = pbc_diff(coords_a, coords_b, box=box)
    if normed:
        vec /= np.linalg.norm(vec, axis=-1).reshape(-1, 1)
    vec.reference = coords_a
    return vec


@map_coordinates
def dipole_vector(
    frame: CoordinateFrame, atom_indices: ArrayLike, normed: bool = None
) -> NDArray:
    coords = frame.whole[atom_indices]
    res_ids = frame.residue_ids[atom_indices]
    charges = frame.charges[atom_indices]
    mask = np.bincount(res_ids)[1:] != 0
    dipoles = np.array(
        [np.bincount(res_ids, weights=c * charges)[1:] for c in coords.T]
    ).T[mask]
    dipoles = np.array(dipoles)
    if normed:
        dipoles /= np.linalg.norm(dipoles, axis=-1).reshape(-1, 1)
    return dipoles


@map_coordinates
def sum_dipole_vector(
    coordinates: CoordinateFrame,
    atom_indices: ArrayLike,
    normed: bool = True,
) -> NDArray:
    coords = coordinates.whole[atom_indices]
    charges = coordinates.charges[atom_indices]
    dipole = np.array([c * charges for c in coords.T]).T
    if normed:
        dipole /= np.linalg.norm(dipole)
    return dipole


@map_coordinates
def normal_vectors(
    frame: CoordinateFrame,
    atom_indices_a: ArrayLike,
    atom_indices_b: ArrayLike,
    atom_indices_c: ArrayLike,
    normed: bool = True,
) -> NDArray:
    coords_a = frame[atom_indices_a]
    coords_b = frame[atom_indices_b]
    coords_c = frame[atom_indices_c]
    box = frame.box
    vectors_a = pbc_diff(coords_a, coords_b, box=box)
    vectors_b = pbc_diff(coords_a, coords_c, box=box)
    vec = np.cross(vectors_a, vectors_b)
    if normed:
        vec /= np.linalg.norm(vec, axis=-1).reshape(-1, 1)
    return vec


def displacements_without_drift(
    start_frame: CoordinateFrame, end_frame: CoordinateFrame, trajectory: Coordinates
) -> np.array:
    start_all = trajectory[start_frame.step]
    frame_all = trajectory[end_frame.step]
    displacements = (
        start_frame
        - end_frame
        - (np.average(start_all, axis=0) - np.average(frame_all, axis=0))
    )
    return displacements


@map_coordinates
def cylindrical_coordinates(
    frame: CoordinateFrame, origin: ArrayLike = None
) -> NDArray:
    if origin is None:
        origin = np.diag(frame.box) / 2
    x = frame[:, 0] - origin[0]
    y = frame[:, 1] - origin[1]
    z = frame[:, 2]
    radius = (x**2 + y**2) ** 0.5
    phi = np.arctan2(y, x)
    return np.array([radius, phi, z]).T


def layer_of_atoms(
    atoms: CoordinateFrame,
    thickness: float,
    plane_normal: ArrayLike,
    plane_offset: Optional[ArrayLike] = np.array([0, 0, 0]),
) -> np.array:
    if plane_offset is None:
        np.array([0, 0, 0])
    atoms = atoms - plane_offset
    distance = np.dot(atoms, plane_normal)
    return np.abs(distance) <= thickness


def next_neighbors(
    atoms: CoordinateFrame,
    query_atoms: Optional[CoordinateFrame] = None,
    number_of_neighbors: int = 1,
    distance_upper_bound: float = np.inf,
    distinct: bool = False,
    **kwargs
) -> Tuple[List, List]:
    """
    Find the N next neighbors of a set of atoms.

    Args:
        atoms:
            The reference atoms and also the atoms which are queried if `query_atoms`
            is net provided
        query_atoms (opt.): If this is not None, these atoms will be queried
        number_of_neighbors (int, opt.): Number of neighboring atoms to find
        distance_upper_bound (float, opt.):
            Upper bound of the distance between neighbors
        distinct (bool, opt.):
            If this is true, the atoms and query atoms are taken as distinct sets of
            atoms
    """
    dnn = 0
    if query_atoms is None:
        query_atoms = atoms
        dnn = 1
    elif not distinct:
        dnn = 1

    box = atoms.box
    if np.all(np.diag(np.diag(box)) == box):
        atoms = atoms % np.diag(box)
        tree = KDTree(atoms, boxsize=np.diag(box))
        distances, indices = tree.query(
            query_atoms,
            number_of_neighbors + dnn,
            distance_upper_bound=distance_upper_bound,
        )
        distances = distances[:, dnn:]
        indices = indices[:, dnn:]
        distances_new = []
        indices_new = []
        for dist, ind in zip(distances, indices):
            distances_new.append(dist[dist <= distance_upper_bound])
            indices_new.append(ind[dist <= distance_upper_bound])
        return distances_new, indices_new
    else:
        atoms_pbc, atoms_pbc_index = pbc_points(
            atoms, box, thickness=distance_upper_bound + 0.1, index=True, **kwargs
        )
        tree = KDTree(atoms_pbc)
        distances, indices = tree.query(
            query_atoms,
            number_of_neighbors + dnn,
            distance_upper_bound=distance_upper_bound,
        )
        distances = distances[:, dnn:]
        indices = indices[:, dnn:]
        distances_new = []
        indices_new = []
        for dist, ind in zip(distances, indices):
            distances_new.append(dist[dist <= distance_upper_bound])
            indices_new.append(atoms_pbc_index[ind[dist <= distance_upper_bound]])
        return distances_new, indices_new


def number_of_neighbors(
    atoms: CoordinateFrame,
    query_atoms: Optional[CoordinateFrame] = None,
    r_max: float = 1,
    distinct: bool = False,
    **kwargs
) -> Tuple[List, List]:
    """
    Find the N next neighbors of a set of atoms.

    Args:
        atoms:
            The reference atoms and also the atoms which are queried if `query_atoms`
            is net provided
        query_atoms (opt.): If this is not None, these atoms will be queried
        r_max (float, opt.):
            Upper bound of the distance between neighbors
        distinct (bool, opt.):
            If this is true, the atoms and query atoms are taken as distinct sets of
            atoms
    """
    dnn = 0
    if query_atoms is None:
        query_atoms = atoms
        dnn = 1
    elif not distinct:
        dnn = 1

    box = atoms.box
    if np.all(np.diag(np.diag(box)) == box):
        atoms = atoms % np.diag(box)
        tree = KDTree(atoms, boxsize=np.diag(box))
    else:
        atoms_pbc = pbc_points(atoms, box, thickness=r_max + 0.1, **kwargs)
        tree = KDTree(atoms_pbc)

    num_of_neighbors = tree.query_ball_point(query_atoms, r_max, return_length=True)
    return num_of_neighbors - dnn
