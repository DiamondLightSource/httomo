from __future__ import annotations
from typing import Literal, Optional, Tuple

import numpy as np

from httomo.runner.auxiliary_data import AuxiliaryData
from httomo.sweep_runner.param_sweep_block import ParamSweepBlock
from httomo.sweep_runner.param_sweep_store_interfaces import ParamSweepSource


class ParamSweepReader(ParamSweepSource):
    """
    Read parameter sweep results, extracting middle slices along the `detector_y` dim
    """

    def __init__(self, source: ParamSweepWriter) -> None:
        self._no_of_sweeps = source.no_of_sweeps
        self._extract_dim: Literal[1] = 1
        self._single_shape = source.single_shape
        self._total_shape = source.total_shape
        assert (
            source._data is not None
        ), "Reader should be created from writer with data not `None`"
        self._data = source._data
        self._aux_data = source.aux_data

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

    @property
    def aux_data(self) -> AuxiliaryData:
        return self._aux_data

    def read_sweep_results(self) -> ParamSweepBlock:
        slices = [slice(None, None, 1)] * 3
        slices_per_sweep = self.single_shape[self.extract_dim]
        first_middle_slice_index = slices_per_sweep // 2
        detector_y_slice = slice(first_middle_slice_index, None, slices_per_sweep)
        slices[self.extract_dim] = detector_y_slice
        return ParamSweepBlock(
            data=self._data[slices[0], slices[1], slices[2]],
            aux_data=self.aux_data,
            slicing_dim=self.extract_dim,
        )


class ParamSweepWriter:
    """Write parameter sweep results, concatenating them along the `detector_y` dim"""

    def __init__(self, no_of_sweeps: int) -> None:
        self._concat_dim: Literal[1] = 1
        self._no_of_sweeps = no_of_sweeps
        self._no_of_sweeps_written: int = 0
        self._single_shape: Optional[Tuple[int, int, int]] = None
        self._total_shape: Optional[Tuple[int, int, int]] = None
        self._data: Optional[np.ndarray] = None
        self._slices_per_sweep: Optional[int] = None

    @property
    def no_of_sweeps(self) -> int:
        return self._no_of_sweeps

    @property
    def concat_dim(self) -> Literal[0, 1, 2]:
        return self._concat_dim

    @property
    def single_shape(self) -> Tuple[int, int, int]:
        if self._single_shape is None:
            err_str = "Shape of single sweep result isn't known until the first write has occurred"
            raise ValueError(err_str)
        return self._single_shape

    @property
    def total_shape(self) -> Tuple[int, int, int]:
        if self._total_shape is None:
            err_str = (
                "Shape of full array holding sweep results isn't known until the first "
                "write has occurred"
            )
            raise ValueError(err_str)
        return self._total_shape

    def make_reader(self) -> ParamSweepReader:
        if self._data is None:
            raise ValueError("Cannot make reader when no data has been written yet")
        return ParamSweepReader(self)

    @property
    def no_of_sweeps_written(self) -> int:
        return self._no_of_sweeps_written

    def increment_no_of_sweeps_written(self):
        self._no_of_sweeps_written += 1

    @property
    def slices_per_sweep(self) -> int:
        if self._slices_per_sweep is None:
            raise ValueError(
                "Slices per sweep isn't known until the first write has occurred"
            )
        return self._slices_per_sweep

    @property
    def aux_data(self) -> AuxiliaryData:
        return self._aux_data

    def write_sweep_result(self, block: ParamSweepBlock):
        block.to_cpu()
        if self._data is None:
            self._single_shape = block.shape
            self._total_shape = (
                self.single_shape[0],
                self.single_shape[1] * self.no_of_sweeps,
                self.single_shape[2],
            )
            self._slices_per_sweep = self.single_shape[self.concat_dim]
            self._data = np.empty(shape=self.total_shape, dtype=block.data.dtype)
            self._aux_data = block.aux_data

        slices = [slice(None, None, 1)] * 3
        sweep_res_start = self.no_of_sweeps_written * self.slices_per_sweep
        slices[self._concat_dim] = slice(
            sweep_res_start,
            sweep_res_start + self.slices_per_sweep,
            1,
        )
        self._data[slices[0], slices[1], slices[2]] = block.data
        self.increment_no_of_sweeps_written()
