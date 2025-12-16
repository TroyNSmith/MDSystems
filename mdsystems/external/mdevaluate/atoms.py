import re

import numpy as np

from .checksum import checksum


def compare_regex(str_list: list[str], exp: str) -> np.ndarray:
    """
    Compare a list of strings with a regular expression.
    """
    regex = re.compile(exp)
    return np.array([regex.match(s) is not None for s in str_list])


class Atoms:
    """
    Basic container class for atom information.

    Args:
        atoms: N tuples of residue id, residue name and atom name.
        indices (optional): Dictionary of named atom index groups.

    Attributes:
        residue_ids: Indices of the atoms residues
        residue_names: Names of the atoms residues
        atom_names: Names of the atoms
        indices: Dictionary of named atom index groups, if specified

    """

    def __init__(self, atoms, indices=None, masses=None, charges=None, reader=None):
        self.residue_ids, self.residue_names, self.atom_names = atoms.T
        self.residue_ids = np.array([int(m) for m in self.residue_ids])
        self.indices = indices
        self.masses = masses
        self.charges = charges
        self.reader = reader

    def subset(self, *args, **kwargs):
        """
        Return a subset of these atoms with all atoms selected.

        All arguments are passed to the :meth:`AtomSubset.subset` method directly.

        """
        return AtomSubset(self).subset(*args, **kwargs)

    def __len__(self):
        return len(self.atom_names)


class AtomMismatch(Exception):
    pass


class AtomSubset:
    def __init__(self, atoms, selection=None, description=""):
        """
        Args:
            atoms: Base atom object
            selection (opt.): Selected atoms
            description (opt.): Descriptive string of the subset.
        """
        if selection is None:
            selection = np.ones(len(atoms), dtype="bool")
        self.selection = selection
        self.atoms = atoms
        self.description = description

    def subset(self, atom_name=None, residue_name=None, residue_id=None, indices=None):
        """
        Return a subset of the system. The selection is specified by one or more of
        the keyworss below. Names are matched as a regular expression with `re.match`.

        Args:
            atom_name: Specification of the atom name
            residue_name: Specification of the resiude name
            residue_id: Residue ID or list of IDs
            indices: List of atom indices
        """
        new_subset = self
        if atom_name is not None:
            new_subset &= AtomSubset(
                self.atoms,
                selection=compare_regex(self.atoms.atom_names, atom_name),
                description=atom_name,
            )

        if residue_name is not None:
            new_subset &= AtomSubset(
                self.atoms,
                selection=compare_regex(self.atoms.residue_names, residue_name),
                description=residue_name,
            )

        if residue_id is not None:
            if np.iterable(residue_id):
                selection = np.zeros(len(self.selection), dtype="bool")
                selection[np.in1d(self.atoms.residue_ids, residue_id)] = True
                new_subset &= AtomSubset(self.atoms, selection)
            else:
                new_subset &= AtomSubset(
                    self.atoms, self.atoms.residue_ids == residue_id
                )

        if indices is not None:
            selection = np.zeros(len(self.selection), dtype="bool")
            selection[indices] = True
            new_subset &= AtomSubset(self.atoms, selection)
        return new_subset

    @property
    def atom_names(self):
        return self.atoms.atom_names[self.selection]

    @property
    def residue_names(self):
        return self.atoms.residue_names[self.selection]

    @property
    def residue_ids(self):
        return self.atoms.residue_ids[self.selection]

    @property
    def indices(self):
        return np.where(self.selection)

    def __getitem__(self, slice):
        if isinstance(slice, str):
            indices = self.atoms.indices[slice]
            return self.atoms.subset()[indices] & self

        return self.subset(indices=self.indices[0].__getitem__(slice))

    def __and__(self, other):
        if self.atoms != other.atoms:
            raise AtomMismatch
        selection = self.selection & other.selection
        description = "{}_{}".format(self.description, other.description).strip("_")
        return AtomSubset(self.atoms, selection, description)

    def __or__(self, other):
        if self.atoms != other.atoms:
            raise AtomMismatch
        selection = self.selection | other.selection
        description = "{}_{}".format(self.description, other.description).strip("_")
        return AtomSubset(self.atoms, selection, description)

    def __invert__(self):
        selection = ~self.selection
        return AtomSubset(self.atoms, selection, self.description)

    def __repr__(self):
        return "Subset of Atoms ({} of {})".format(
            len(self.atoms.residue_names[self.selection]), len(self.atoms)
        )

    @property
    def summary(self):
        return "\n".join(
            [
                "{}{} {}".format(resid, resname, atom_names)
                for resid, resname, atom_names in zip(
                    self.residue_ids, self.residue_names, self.atom_names
                )
            ]
        )

    def __checksum__(self):
        return checksum(self.description)





