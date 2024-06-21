from __future__ import annotations
from typing import Literal, Tuple


class ParamSweepReader:
    """
    Read parameter sweep results, extracting middle slices along the `detector_y` dim
    """

    def __init__(self, source: ParamSweepWriter) -> None:
        self._no_of_sweeps = source.no_of_sweeps
        self._extract_dim: Literal[1] = 1
        self._single_shape = source.single_shape
        self._total_shape = source.total_shape

    @property
    def no_of_sweeps(self) -> int:
        return self._no_of_sweeps

    @property
    def extract_dim(self) -> Literal[0, 1, 2]:
        return self._extract_dim

    @property
    def single_shape(self) -> Tuple[int, int, int]:
        return self._single_shape

    @property
    def total_shape(self) -> Tuple[int, int, int]:
        return self._total_shape


class ParamSweepWriter:
    """Write parameter sweep results, concatenating them along the `detector_y` dim"""

    def __init__(
        self,
        no_of_sweeps: int,
        single_shape: Tuple[int, int, int],
    ) -> None:
        self._concat_dim: Literal[1] = 1
        self._no_of_sweeps = no_of_sweeps
        self._single_shape = single_shape
        self._total_shape = (
            single_shape[0],
            single_shape[1] * no_of_sweeps,
            single_shape[2],
        )

    @property
    def no_of_sweeps(self) -> int:
        return self._no_of_sweeps

    @property
    def concat_dim(self) -> Literal[0, 1, 2]:
        return self._concat_dim

    @property
    def single_shape(self) -> Tuple[int, int, int]:
        return self._single_shape

    @property
    def total_shape(self) -> Tuple[int, int, int]:
        return self._total_shape

    def make_reader(self) -> ParamSweepReader:
        return ParamSweepReader(self)
