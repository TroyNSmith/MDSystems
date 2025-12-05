"""Functions for analyzing MD simulations."""

from functools import partial

import numpy as np

from . import pickles
from .atoms import System
from .iterators import time_average


class Spatial:
    """Functions for performing spatial analyses."""

    @staticmethod
    def radial_distribution(
        sys: System,
        ref: System | None = None,
        com: bool = False,
        mode: str = "total",
        start_frame: int = 0,
        end_frame: int | None = None,
        nbins: int = 100,
        rmin: float = 0.2,
        rmax: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute radial distribution function G(r) between sys and ref.

        Parameters
        ----------
        sys : System
            Trajectory to analyze.
        ref : System | None (default = None)
            Trajectory to compute distances from sys. If None, sys is compared to itself.
        com : Bool (default = False)
            If True, reduce coordinates to residue-level centers of mass.
        mode : str (default = "total", options = "intra", "inter")
            If total, compute all distance. If intra, compute distances between indices in sys and \
            ref where i == j. If inter, compute distances between indices in sys and ref \
            where i != j.
        start_frame : int (default = 0)
            Index of first frame to analyze in trajectory.
        end_frame : int | None (default = None)
            Index of final frame to analyze in trajectory.
        nbins : int (default = 100)
            Number of histogram bins in output.
        rmin : float (default = 0.2)
            Minimum distance to include in histogram.
        rmax : float (default = 1.0)
            Maximum distance to include in histogram.

        Returns
        -------
        g_r : np.ndarray[nbins, np.float32]
            Radial distribution function.
        bin_centers : np.ndarray[nbins, np.float32]
            Bin centers for radial distribution function.
        """
        ref = sys if ref is None else ref

        g_r = time_average(
            partial(pickles.rdf_distances, mode=mode, nbins=nbins, rmin=rmin, rmax=rmax),
            sys,
            ref,
            com,
            start_frame,
            end_frame,
        )

        bins = np.linspace(rmin, rmax, nbins + 1)
        bin_centers = (bins[1:] + bins[:-1]) / 2

        return g_r, bin_centers


class Temporal:
    """Functions for performing temporal analyses."""

    @staticmethod
    def mean_square_displacement():
        """Compute mean square displacement <r*r> for sys."""
        return
