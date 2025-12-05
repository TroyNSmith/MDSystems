"""Analytical functions pertaining to temporal distribution."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from .atoms import System


@dataclass
class MeanSquareDisplacement:
    """Compute MSD using System data structures."""

    sys: System
    start_t: float | None = None
    end_t: float | None = None
    windows: int = 10
    points: int = 100

    @staticmethod
    def _compute_msd_for_origin(xyz0: np.ndarray, xyz_lags: np.ndarray, box: np.ndarray):
        """
        Compute MSD for one origin over multiple lag frames.

        xyz0: [n_atoms, 3]
        xyz_lags: [n_lags, n_atoms, 3]
        box: [3].
        """
        dr = xyz_lags - xyz0  # broadcasting over lags
        dr -= box * np.round(dr / box)  # minimum image
        msd = np.mean(np.sum(dr**2, axis=2), axis=1)  # mean over atoms
        return msd

    def run(self, com=True):
        times = np.asarray(self.sys.times)
        box = self.sys.unitcell.diagonal()

        # Determine start and end indices
        start_t = self.start_t if self.start_t is not None else times[0]
        end_t = self.end_t if self.end_t is not None else times[-1]
        i_start = np.argmin(np.abs(times - start_t))
        i_end = np.argmin(np.abs(times - end_t))

        # Origin frames
        origin_indices = np.unique(
            np.linspace(i_start, i_end, num=self.windows, endpoint=False, dtype=int)
        )

        # Lag frames (log spaced)
        window_length = i_end - i_start
        lag_indices = np.unique(
            (np.logspace(0, np.log10(window_length + 1), num=self.points) - 1).astype(int)
        )
        lag_indices = lag_indices[lag_indices < len(times)]

        msd_accum = np.zeros(len(lag_indices))

        # Parallelize over origins
        with ThreadPoolExecutor() as pool:
            futures = []
            for origin in origin_indices:
                # Load origin frame
                xyz0, mask = next(self.sys.load_cartesian_frames([origin], com, return_mask=True))

                # Load all lag frames relative to this origin
                frame_indices = origin + lag_indices
                frame_indices = frame_indices[frame_indices < len(times)]

                xyz_lags = np.array(
                    list(self.sys.load_cartesian_frames(frame_indices, com, mask=mask))
                )

                futures.append(pool.submit(self._compute_msd_for_origin, xyz0, xyz_lags, box))

            results = [f.result() for f in futures]

        output = []
        for r in results:
            padded = np.full(len(lag_indices), np.nan)
            padded[: len(r)] = r
            output.append(padded)

        accumulated = np.nanmean(output, axis=0)  # average over origins
        t = times[lag_indices] - times[0]

        return t, msd_accum
