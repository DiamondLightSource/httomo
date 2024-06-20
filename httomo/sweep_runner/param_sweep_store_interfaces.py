from typing import Literal, Protocol, Tuple

from httomo.runner.auxiliary_data import AuxiliaryData
from httomo.runner.dataset import DataSetBlock


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

    def read_sweep_results(self) -> DataSetBlock:
        """
        Returns a single block containing the middle slices extracted along `extract_dim` from
        each sweep result
        """
        ...  # pragma: no cover
