"""Functions for analyzing MD simulations."""

from functools import partial

import numpy as np

from . import pickles
from .atoms import System
from .iterators import shifted_correlation, time_average


class Structural:
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
            partial(pickles.pairwise_distances, mode=mode, nbins=nbins, rmin=rmin, rmax=rmax),
            sys,
            ref,
            com,
            start_frame,
            end_frame,
        )

        bins = np.linspace(rmin, rmax, nbins + 1)
        bin_centers = (bins[1:] + bins[:-1]) / 2

        return g_r, bin_centers


class Dynamics:
    """Functions for performing temporal analyses."""

    @staticmethod
    def incoherent_scattering(
        sys: System,
        q: float,
        com: bool = False,
        start_frame: int = 0,
        end_frame: int | None = None,
        windows: int = 10,
        points: int = 100,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return incoherent scattering function computed with num. windows shifted correlations.

        Parameters
        ----------
        fn : Callable
            Function to call.
        q : float
            Magnitude of the scattering vector, q.
        sys : System
            Trajectory to analyze.
        com : Bool (default = False)
            If True, reduce coordinates to residue-level centers of mass.
        start_frame : int (default = 0)
            Index of first frame to analyze in trajectory.
        end_frame : int | None (default = None)
            Index of final frame to analyze in trajectory.
        windows : int (default = 10)
            Number of starting points in the analysis.
        points : int (default = 100)
            Number of points to analyze.

        Returns
        -------
        g_r : np.ndarray[any, np.float32]
            Radial distribution function.
        times : np.ndarray[any, np.float32]
            Bin centers for radial distribution function.
        """
        fn = partial(pickles.incoherent_scattering, q=q)
        return shifted_correlation(fn, sys, com, start_frame, end_frame, windows, points)

    @staticmethod
    def mean_square_displacement(
        sys: System,
        com: bool = False,
        start_frame: int = 0,
        end_frame: int | None = None,
        windows: int = 10,
        points: int = 100,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return mean square displacement computed with num. windows shifted correlations.

        Parameters
        ----------
        fn : Callable
            Function to call.
        sys : System
            Trajectory to analyze.
        com : Bool (default = False)
            If True, reduce coordinates to residue-level centers of mass.
        start_frame : int (default = 0)
            Index of first frame to analyze in trajectory.
        end_frame : int | None (default = None)
            Index of final frame to analyze in trajectory.
        windows : int (default = 10)
            Number of starting points in the analysis.
        points : int (default = 100)
            Number of points to analyze.

        Returns
        -------
        msd : np.ndarray[any, np.float32]
            Mean square displacement.
        times : np.ndarray[any, np.float32]
            Time values for mean square displacements.
        """
        fn = pickles.mean_square_displacements
        return shifted_correlation(fn, sys, com, start_frame, end_frame, windows, points)

    @staticmethod
    def van_hove_self(
        sys: System,
        com: bool = False,
        rmax: float = 2.0,
        nbins: int = 200,
        start_frame: int = 0,
        end_frame: int | None = None,
        windows: int = 10,
        points: int = 100,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return mean square displacement computed with num. windows shifted correlations.

        Parameters
        ----------
        fn : Callable
            Function to call.
        sys : System
            Trajectory to analyze.
        com : Bool (default = False)
            If True, reduce coordinates to residue-level centers of mass.
        start_frame : int (default = 0)
            Index of first frame to analyze in trajectory.
        end_frame : int | None (default = None)
            Index of final frame to analyze in trajectory.
        windows : int (default = 10)
            Number of starting points in the analysis.
        points : int (default = 100)
            Number of points to analyze.

        Returns
        -------
        G_s : np.ndarray[any, np.float32]
            Van Hove self correlation function.
        r_centers : np.ndarray[, np.float32]
            Radial bin centers.
        times : np.ndarray[any, np.float32]
            Time lag values.
        """
        r_edges = np.linspace(0.0, rmax, nbins + 1)
        r_centers = (r_edges[1:] + r_edges[:-1]) / 2

        fn = partial(pickles.van_hove_self, r_edges=r_edges)

        G_s, times = shifted_correlation(fn, sys, com, start_frame, end_frame, windows, points)

        return G_s, r_centers, times
