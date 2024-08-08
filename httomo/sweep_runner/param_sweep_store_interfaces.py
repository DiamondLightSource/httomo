from typing import Literal, Protocol, Tuple

from httomo.runner.auxiliary_data import AuxiliaryData
from httomo.runner.dataset import DataSetBlock
from httomo.sweep_runner.param_sweep_block import ParamSweepBlock


class ParamSweepSource(Protocol):
    """Source for reading parameter sweep results from"""

    @property
    def no_of_sweeps(self) -> int:
        """Number of parameter values used to generate sweep results"""
        ...  # pragma: no cover

    @property
    def extract_dim(self) -> Literal[0, 1, 2]:
        """Dimension to extract middle slices along - 0, 1, or 2"""
        ...  # pragma: no cover

    @property
    def single_shape(self) -> Tuple[int, int, int]:
        """Shape of single param sweep result"""
        ...  # pragma: no cover

    @property
    def total_shape(self) -> Tuple[int, int, int]:
        """
        Shape of 3D array that contains all param sweep results concatenated along
        `extract_dim`
        """
        ...  # pragma: no cover

    @property
    def aux_data(self) -> AuxiliaryData:
        """Get auxiliary data"""
        ...  # pragma: no cover

    def read_sweep_results(self) -> ParamSweepBlock:
        """
        Returns a single block containing the middle slices extracted along `extract_dim` from
        each sweep result
        """
        ...  # pragma: no cover


class ParamSweepSink(Protocol):
    """Sink for writing parameter sweep results to"""

    @property
    def no_of_sweeps(self) -> int:
        """Number of parameter values to sweep over"""
        ...  # pragma: no cover

    @property
    def concat_dim(self) -> Literal[0, 1, 2]:
        """Dimension to concatenate sweep results along - 0, 1, or 2"""
        ...  # pragma: no cover

    @property
    def single_shape(self) -> Tuple[int, int, int]:
        """Shape of single param sweep result"""
        ...  # pragma: no cover

    @property
    def total_shape(self) -> Tuple[int, int, int]:
        """
        Shape of 3D array to contain all param sweep results concatenated along `concat_dim`
        """
        ...  # pragma: no cover

    def write_sweep_result(self, block: DataSetBlock):
        """
        Writes single sweep result contained in a block to store along the `concat_dim` axis
        """
        ...  # pragma: no cover

    def finalize(self):
        """Method to be called after writing all blocks is done"""
        ...  # pragma: no cover

    def make_reader(self) -> ParamSweepSource:
        """
        Return a source from the sink, from which the data written to the sink can be read
        """
        ...  # pragma: no cover
