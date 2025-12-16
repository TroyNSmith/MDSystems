import os
from glob import glob
from typing import Optional

import pandas as pd

from . import atoms
from . import autosave
from . import checksum
from . import coordinates
from . import correlation
from . import distribution
from . import functions
from . import pbc
from . import reader
from . import system
from . import utils
from . import extra
from .logging import logger


def open(
    directory: str = "",
    topology: str = "*.tpr",
    trajectory: str = "*.xtc",
    nojump: bool = False,
    index_file: Optional[str] = None,
    charges: Optional[list[float]] = None,
    masses: Optional[list[float]] = None,
) -> coordinates.Coordinates:
    """
    Open a simulation from a directory.

    Args:
        directory: Directory of the simulation.
        topology (opt.):
            Descriptor of the topology file (tpr or gro). By default, a tpr file is
            used, if there is exactly one in the directoy.
        trajectory (opt.): Descriptor of the trajectory (xtc or trr file).
        nojump (opt.):
            If nojump matrices should be generated. They will alwyas be loaded
            if present
        index_file (opt.): Descriptor of the index file (ndx file).
        charges (opt.):
            List with charges for each atom. It Has to be the same length as the number
            of atoms in the system. Only used if topology file is a gro file.
        masses (opt.):
            List with masses for each atom. It Has to be the same length as the number
            of atoms in the system. Only used if topology file is a gro file.

    Returns:
        A Coordinate object of the simulation.

    Example:
        Open a simulation located in '/path/to/sim', where the trajectory is
        located in a sub-directory '/path/to/sim/out' and named for Example
        'nojump_traj.xtc'.

        >>> open('/path/to/sim', trajectory='out/nojump*.xtc')

    The file descriptors can use unix style pathname expansion to define the filenames.
    The default patterns use the recursive placeholder `**` which matches the base or
    any subdirctory, thus files in subdirectories with matching file type will be found
    too.
    For example: 'out/nojump*.xtc' would match xtc files in a subdirectory `out` that
    start with `nojump` and end with `.xtc`.

    For more details see: https://docs.python.org/3/library/glob.html
    """
    top_glob = glob(os.path.join(directory, topology), recursive=True)
    if top_glob is not None and len(top_glob) == 1:
        (top_file,) = top_glob
        logger.info("Loading topology: {}".format(top_file))
        if index_file is not None:
            index_glob = glob(os.path.join(directory, index_file), recursive=True)
            if index_glob is not None:
                index_file = index_glob[0]
            else:
                index_file = None
    else:
        raise FileNotFoundError("Topology file could not be identified.")

    traj_glob = glob(os.path.join(directory, trajectory), recursive=True)
    if traj_glob is not None and len(traj_glob) == 1:
        traj_file = traj_glob[0]
        logger.info("Loading trajectory: {}".format(traj_file))
    else:
        raise FileNotFoundError("Trajectory file could not be identified.")

    atom_set, frames = reader.open_with_mdanalysis(
        top_file,
        traj_file,
        index_file=index_file,
        charges=charges,
        masses=masses,
    )
    coords = coordinates.Coordinates(frames, atom_subset=atom_set)
    if nojump:
        try:
            frames.nojump_matrices
        except reader.NojumpError:
            reader.generate_nojump_matrices(coords)
    return coords


def open_energy(file: str) -> pd.DataFrame:
    """Reads a gromacs energy file and output the data in a pandas DataFrame.
    Args:
        file: Filename of the energy file
    Returns:
        A DataFrame with the different energies doe each time.
    """
    return pd.DataFrame(reader.energy_reader(file).data_dict)
