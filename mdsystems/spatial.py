"""Analytical functions pertaining to molecular distribution."""

from concurrent.futures import ProcessPoolExecutor
from copy import copy
from dataclasses import dataclass

import numpy as np

from .atoms import System


@dataclass
class RadialDistribution:
    """Container for radial distribution computations."""

    sys_a: System
    sys_b: System | None = None

    min_dist: float = 0.2
    max_dist: float = 1.0

    num_bins: int = 200

    @staticmethod
    def _rdf_frame(xyz_a, xyz_b, unitcell, num_bins, min_dist, max_dist, mode, unique):
        box = np.asarray(unitcell).diagonal() if unitcell.ndim == 2 else np.asarray(unitcell)
        # Apply minimum image convention
        dr = xyz_a[:, None, :] - xyz_b[None, :, :]
        dr -= box * np.round(dr / box)  # Dirac delta function

        dmat = np.sqrt((dr**2).sum(axis=-1))

        n, m = dmat.shape
        same_shape = n == m

        if mode == "intra":
            if not same_shape:
                raise ValueError("Intra mode requires xyz_a and xyz_b to have same shape.")

            dists = dmat[np.diag_indices(n)]

        elif mode == "inter":
            if same_shape:
                if not unique:
                    # upper triangle only
                    dists = dmat[np.triu_indices(n, k=1)]
                else:
                    # full matrix without diagonal
                    dists = dmat[~np.eye(n, dtype=bool)]
            else:
                # All n x m pairs
                dists = dmat.ravel()

        elif mode == "total":
            if same_shape and not unique:
                # diagonal + upper triangle only
                dists = dmat[np.triu_indices(n, k=0)]
            else:
                # All n x m pairs
                dists = dmat.ravel()

        else:
            raise ValueError(f"Unknown mode: {mode}")

        hist, _ = np.histogram(dists, bins=num_bins, range=(min_dist, max_dist))

        return hist

    def run(
        self, mode: str = "total", com: bool = False, unique: bool = True, stride: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute radial distribution function.

        Params
        ------
        mode   | str
            Type of RDF to compute. Options: "total", "inter", "intra".
            Note: "intra" assumes that residue IDs along the diagonal correspond to one another.
        com    | bool
            If True, use center of masses for computation.
        unique | bool
            If True, include both the upper and lower triangles in results.
        stride | int
            Number of frames to skip between each calculation.

        Returns
        -------
        g_r         | np.ndarray
            Radial distribution probability function with ideal gas correction.
        bin_centers | np.ndarray
            Bin centers corresponding to g_r.
        """
        if self.sys_b is None:
            self.sys_b = copy(self.sys_a)
            unique = False

        unitcell = self.sys_a.unitcell

        if stride is not None:
            assert stride >= 1, "Stride must be a positive integer."

            self.sys_a.stride_len = stride
            self.sys_b.stride_len = stride

        with ProcessPoolExecutor() as pool:
            futures = []

            for xyz_a, xyz_b in zip(
                self.sys_a.load_cartesian(com), self.sys_b.load_cartesian(com), strict=True
            ):
                futures.append(
                    pool.submit(
                        RadialDistribution._rdf_frame,
                        xyz_a,
                        xyz_b,
                        unitcell,
                        self.num_bins,
                        self.min_dist,
                        self.max_dist,
                        mode,
                        unique,
                    )
                )

            histograms = [f.result() for f in futures]

        all_hist = np.sum(histograms, axis=0)
        num_frames = len(histograms)

        bins = np.linspace(self.min_dist, self.max_dist, self.num_bins + 1)

        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        dr = bins[1] - bins[0]

        xyz_a0, xyz_b0 = next(
            zip(self.sys_a.load_cartesian(com), self.sys_b.load_cartesian(com), strict=True)
        )

        rho = len(xyz_b0) / self.sys_a.volume
        shell_vol = 4 * np.pi * bin_centers**2 * dr
        ideal_counts = rho * shell_vol * len(xyz_a0)

        g_r = all_hist / ideal_counts / num_frames

        return g_r, bin_centers
