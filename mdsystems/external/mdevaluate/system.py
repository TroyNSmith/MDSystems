import abc
from dataclasses import dataclass, field
from subprocess import run
from typing import Iterable

import pandas as pd
from tables import NoSuchNodeError


@dataclass(kw_only=True)
class MDSystem(abc.ABC):
    load_only_results: bool = False
    system_dir: str = field(init=False)

    @abc.abstractmethod
    def _add_description(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

    def save_results(self, data: pd.DataFrame, key: str) -> None:
        data = self._add_description(data)
        hdf5_file = f"{self.system_dir}/out/results.h5"
        data.to_hdf(hdf5_file, key=key, complevel=9, complib="blosc")

    def load_results(self, key: str) -> pd.DataFrame:
        hdf5_file = f"{self.system_dir}/out/results.h5"
        data = pd.read_hdf(hdf5_file, key=key)
        if isinstance(data, pd.DataFrame):
            return data
        else:
            raise TypeError("Result is not a DataFrame!")

    def cleanup_results(self) -> None:
        hdf5_file = f"{self.system_dir}/out/results.h5"
        hdf5_temp_file = f"{self.system_dir}/out/results_temp.h5"
        run(
            [
                "ptrepack",
                "--chunkshape=auto",
                "--propindexes",
                "--complevel=9",
                "--complib=blosc",
                hdf5_file,
                hdf5_temp_file,
            ]
        )
        run(["mv", hdf5_temp_file, hdf5_file])


def load_and_concat_data(systems: Iterable[MDSystem], key: str, verbose: bool = False):
    data = []
    for system in systems:
        try:
            data.append(system.load_results(key=key))
            if verbose:
                print(f"Load {system}")
        except (FileNotFoundError, KeyError, NoSuchNodeError):
            continue
    return pd.concat(data, ignore_index=True)
