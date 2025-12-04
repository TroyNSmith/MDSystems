"""Analytical functions pertaining to temporal distribution."""

from dataclasses import dataclass

import numpy as np

from . import helpers
from .atoms import System


@dataclass
class MeanSquareDisplacement:
    """Container for MSD computations."""

    sys: System

    start_t: float | None = None
    end_t: float | None = None

    windows: int = 10
    points: int = 100

    @staticmethod
    def _compute_msd(xyz0, xyz, box):
        dr = xyz - xyz0
        dr -= box * np.round(dr / box)
        return np.mean(np.sum(dr * dr, axis=1))

    def run(self, com=True):
        times = np.asarray(self.sys.times)
        coords = list(self.sys.load_cartesian(com))

        box = self.sys.unitcell.diagonal()

        # Find nearest actual start/end times
        start_t = helpers.find_nearest(times, self.start_t or times[0])
        end_t = helpers.find_nearest(times, self.end_t or times[-1])

        i_start = np.where(times == start_t)[0][0]
        i_end = np.where(times == end_t)[0][0]

        # Origins
        origin_indices = np.unique(
            np.linspace(i_start, i_end, num=self.windows, endpoint=False, dtype=int)
        )

        # Determine lag times (log spaced)
        window_length = i_end - i_start
        ls = np.logspace(0, np.log10(window_length + 1), num=self.points)
        lag_indices = np.unique((ls - 1).astype(int))

        # Ensure lag frames stay within trajectory
        lag_indices = lag_indices[lag_indices < len(times)]

        msd = np.zeros(len(lag_indices))

        for o in origin_indices:
            xyz0 = coords[o]

            for j, lag in enumerate(lag_indices):
                if o + lag >= len(coords):
                    continue

                xyz = coords[o + lag]

                msd[j] += self._compute_msd(xyz0, xyz, box)

        msd /= len(origin_indices)

        t = times[lag_indices] - times[0]
        return t, msd
