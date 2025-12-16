"""
Module that provides different readers for trajectory files.

It also provides a common interface layer between the file IO packages,
namely mdanalysis, and mdevaluate.
"""
from collections import namedtuple
import os
from os import path
from array import array
from zipfile import BadZipFile
import builtins
import re
import itertools

import numpy as np
import numpy.typing as npt
import MDAnalysis
from scipy import sparse

from .checksum import checksum
from .logging import logger
from . import atoms
from .coordinates import Coordinates

CSR_ATTRS = ("data", "indices", "indptr")
NOJUMP_MAGIC = 2016
Group_RE = re.compile("\[ ([-+\w]+) \]")


class NojumpError(Exception):
    pass


class WrongTopologyError(Exception):
    pass


class BaseReader:
    """Base class for trajectory readers."""

    @property
    def filename(self):
        return self.rd.filename

    @property
    def nojump_matrices(self):
        if self._nojump_matrices is None:
            raise NojumpError("Nojump Data not available: {}".format(self.filename))
        return self._nojump_matrices

    @nojump_matrices.setter
    def nojump_matrices(self, mats):
        self._nojump_matrices = mats

    def __init__(self, rd):
        self.rd = rd
        self._nojump_matrices = None
        if path.exists(nojump_load_filename(self)):
            load_nojump_matrices(self)

    def __getitem__(self, item):
        return self.rd[item]

    def __len__(self):
        return len(self.rd)

    def __checksum__(self):
        cache = array("L", self.rd._xdr.offsets.tobytes())
        return checksum(self.filename, str(cache))


def open_with_mdanalysis(
    topology: str,
    trajectory: str,
    index_file: str = None,
    charges: npt.ArrayLike = None,
    masses: npt.ArrayLike = None,
) -> (atoms.Atoms, BaseReader):
    """Open the topology and trajectory with mdanalysis."""
    uni = MDAnalysis.Universe(topology, trajectory, convert_units=False)
    reader = BaseReader(uni.trajectory)
    reader.universe = uni
    if topology.endswith(".tpr"):
        charges = uni.atoms.charges
        masses = uni.atoms.masses
    elif topology.endswith(".gro"):
        charges = charges
        masses = masses
    else:
        raise WrongTopologyError('Topology file should end with ".tpr" or ".gro"')
    indices = None
    if index_file:
        indices = load_indices(index_file)

    atms = atoms.Atoms(
        np.stack((uni.atoms.resids, uni.atoms.resnames, uni.atoms.names), axis=1),
        charges=charges,
        masses=masses,
        indices=indices,
    ).subset()

    return atms, reader


def load_indices(index_file: str):
    indices = {}
    index_array = None
    with open(index_file) as idx_file:
        for line in idx_file:
            m = Group_RE.search(line)
            if m is not None:
                group_name = m.group(1)
                index_array = indices.get(group_name, [])
                indices[group_name] = index_array
            else:
                elements = line.strip().split("\t")
                elements = [x.split(" ") for x in elements]
                elements = itertools.chain(*elements)  # make a flat iterator
                elements = [x for x in elements if x != ""]
                index_array += [int(x) - 1 for x in elements]
    return indices


def is_writeable(fname: str):
    """Test if a directory is actually writeable, by writing a temporary file."""
    fdir = os.path.dirname(fname)
    ftmp = os.path.join(fdir, str(np.random.randint(999999999)))
    while os.path.exists(ftmp):
        ftmp = os.path.join(fdir, str(np.random.randint(999999999)))

    if os.access(fdir, os.W_OK):
        try:
            with builtins.open(ftmp, "w"):
                pass
            os.remove(ftmp)
            return True
        except PermissionError:
            pass
    return False


def nojump_load_filename(reader: BaseReader):
    directory, fname = path.split(reader.filename)
    full_path = path.join(directory, ".{}.nojump.npz".format(fname))
    if not is_writeable(directory):
        user_data_dir = os.path.join("/data/", os.environ["HOME"].split("/")[-1])
        full_path_fallback = os.path.join(
            os.path.join(user_data_dir, ".mdevaluate/nojump"),
            directory.lstrip("/"),
            ".{}.nojump.npz".format(fname),
        )
        if os.path.exists(full_path_fallback):
            return full_path_fallback
    if os.path.exists(full_path) or is_writeable(directory):
        return full_path
    else:
        user_data_dir = os.path.join("/data/", os.environ["HOME"].split("/")[-1])
        full_path = os.path.join(
            os.path.join(user_data_dir, ".mdevaluate/nojump"),
            directory.lstrip("/"),
            ".{}.nojump.npz".format(fname),
        )
        return full_path


def nojump_save_filename(reader: BaseReader):
    directory, fname = path.split(reader.filename)
    full_path = path.join(directory, ".{}.nojump.npz".format(fname))
    if is_writeable(directory):
        return full_path
    else:
        user_data_dir = os.path.join("/data/", os.environ["HOME"].split("/")[-1])
        full_path_fallback = os.path.join(
            os.path.join(user_data_dir, ".mdevaluate/nojump"),
            directory.lstrip("/"),
            ".{}.nojump.npz".format(fname),
        )
        logger.info(
            "Saving nojump to {}, since original location is not writeable.".format(
                full_path_fallback
            )
        )
        os.makedirs(os.path.dirname(full_path_fallback), exist_ok=True)
        return full_path_fallback


def parse_jumps(trajectory: Coordinates):
    prev = trajectory[0].whole
    box = prev.box
    SparseData = namedtuple("SparseData", ["data", "row", "col"])
    jump_data = (
        SparseData(data=array("b"), row=array("l"), col=array("l")),
        SparseData(data=array("b"), row=array("l"), col=array("l")),
        SparseData(data=array("b"), row=array("l"), col=array("l")),
    )
    for i, curr in enumerate(trajectory):
        if i % 500 == 0:
            logger.debug("Parse jumps Step: %d", i)
        r3 = np.subtract(curr, prev)
        delta_z = np.array(np.rint(np.divide(r3[:, 2], box[2][2])), dtype=np.int8)
        r2 = np.subtract(
            r3,
            (np.rint(np.divide(r3[:, 2], box[2][2])))[:, np.newaxis]
            * box[2][np.newaxis, :],
            )
        delta_y = np.array(np.rint(np.divide(r2[:, 1], box[1][1])), dtype=np.int8)
        r1 = np.subtract(
            r2,
            (np.rint(np.divide(r2[:, 1], box[1][1])))[:, np.newaxis]
            * box[1][np.newaxis, :],
            )
        delta_x = np.array(np.rint(np.divide(r1[:, 0], box[0][0])), dtype=np.int8)
        delta = np.array([delta_x, delta_y, delta_z]).T
        prev = curr
        box = prev.box
        for d in range(3):
            (col,) = np.where(delta[:, d] != 0)
            jump_data[d].col.extend(col)
            jump_data[d].row.extend([i] * len(col))
            jump_data[d].data.extend(delta[col, d])

    return jump_data


def generate_nojump_matrices(trajectory: Coordinates):
    """
    Create the matrices with pbc jumps for a trajectory.
    """
    logger.info("generate Nojump matrices for: {}".format(trajectory))

    jump_data = parse_jumps(trajectory)
    N = len(trajectory)
    M = len(trajectory[0])

    trajectory.frames.nojump_matrices = tuple(
        sparse.csr_matrix((np.array(m.data), (m.row, m.col)), shape=(N, M))
        for m in jump_data
    )
    save_nojump_matrices(trajectory.frames)


def save_nojump_matrices(reader: BaseReader, matrices: npt.ArrayLike = None):
    if matrices is None:
        matrices = reader.nojump_matrices
    data = {"checksum": checksum(NOJUMP_MAGIC, checksum(reader))}
    for d, mat in enumerate(matrices):
        data["shape"] = mat.shape
        for attr in CSR_ATTRS:
            data["{}_{}".format(attr, d)] = getattr(mat, attr)

    np.savez(nojump_save_filename(reader), **data)


def load_nojump_matrices(reader: BaseReader):
    zipname = nojump_load_filename(reader)
    try:
        data = np.load(zipname, allow_pickle=True)
    except (AttributeError, BadZipFile, OSError):
        # npz-files can be corrupted, probably a bug for big arrays saved with
        # savez_compressed?
        logger.info("Removing zip-File: %s", zipname)
        os.remove(nojump_load_filename(reader))
        return
    try:
        if data["checksum"] == checksum(NOJUMP_MAGIC, checksum(reader)):
            reader.nojump_matrices = tuple(
                sparse.csr_matrix(
                    tuple(data["{}_{}".format(attr, d)] for attr in CSR_ATTRS),
                    shape=data["shape"],
                )
                for d in range(3)
            )
            logger.info(
                "Loaded Nojump matrices: {}".format(nojump_load_filename(reader))
            )
        else:
            logger.info("Invlaid Nojump Data: {}".format(nojump_load_filename(reader)))
    except KeyError:
        logger.info("Removing zip-File: %s", zipname)
        os.remove(nojump_load_filename(reader))
        return


def correct_nojump_matrices_for_whole(trajectory: Coordinates):
    reader = trajectory.frames
    frame = trajectory[0]
    box = frame.box.diagonal()
    cor = ((frame - frame.whole) / box).round().astype(np.int8)
    for d in range(3):
        reader.nojump_matrices[d][0] = cor[:, d]
    save_nojump_matrices(reader)


def energy_reader(file: str):
    """Reads a gromacs energy file with mdanalysis and returns an auxiliary file.
    Args:
        file: Filename of the energy file
    """
    return MDAnalysis.auxiliary.EDR.EDRReader(file)
