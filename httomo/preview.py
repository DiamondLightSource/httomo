"""
Slicing configuration types used by loaders to crop the input data.
"""

from typing import List, NamedTuple, Optional, Tuple

import h5py
import numpy as np


class PreviewDimConfig(NamedTuple):
    """
    Slicing configuration for a single dimension in the 3D input data, in the form of start,
    stop, step.

    Notes
    -----

    Currently, only unit step for all three dimensions is supported (ie, a step value of 1).
    Thus, unit step is assumed for all three dimensions, and is not configurable in this type
    yet.
    """

    start: int
    stop: int


class PreviewConfig(NamedTuple):
    """
    Slicing configuration for all three dimensions in the 3D input data.
    """

    angles: PreviewDimConfig
    detector_y: PreviewDimConfig
    detector_x: PreviewDimConfig


class Preview:
    """
    Performs bounds checking of the slicing configuration, and calculations of the cropped data
    indices and global data shape.
    """

    def __init__(
        self,
        preview_config: PreviewConfig,
        dataset: h5py.Dataset,
        image_key: Optional[h5py.Dataset],
    ) -> None:
        self.config = preview_config
        self._dataset = dataset
        self._image_key = image_key
        self._check_within_data_bounds()
        self._data_indices: Optional[List[int]] = None
        self._global_shape: Optional[Tuple[int, int, int]] = None

    def _check_within_data_bounds(self) -> None:
        shape = self._dataset.shape
        for i, field in enumerate(self.config._fields):
            self._check_dimension(
                name=field,
                config=getattr(self.config, field),
                length=shape[i],
            )

    def _check_dimension(
        self,
        name: str,
        config: PreviewDimConfig,
        length: int,
    ) -> None:
        if config.stop > length:
            raise ValueError(
                f"Preview indices in {name} dim exceed bounds of data: "
                f"start={config.start}, stop={config.stop}"
            )

        if config.start >= config.stop:
            raise ValueError(
                f"Preview index error for {name}: start must be strictly smaller "
                f"than stop, but start={config.start}, stop={config.stop}"
            )

    def _calculate_data_indices(self) -> List[int]:
        if self._image_key is not None:
            indices = np.where(self._image_key[:] == 0)[0].tolist()
        else:
            no_of_angles = self._dataset.shape[0]
            indices = list(range(no_of_angles))

        preview_data_indices = np.arange(
            self.config.angles.start, self.config.angles.stop
        )

        intersection = np.intersect1d(indices, preview_data_indices)
        if not np.array_equal(preview_data_indices, intersection):
            self.config = PreviewConfig(
                angles=PreviewDimConfig(
                    start=intersection[0], stop=intersection[-1] + 1
                ),
                detector_y=self.config.detector_y,
                detector_x=self.config.detector_x,
            )

        return intersection.tolist()

    def _calculate_global_shape(self) -> Tuple[int, int, int]:
        return (
            len(self.data_indices),
            self.config.detector_y.stop - self.config.detector_y.start,
            self.config.detector_x.stop - self.config.detector_x.start,
        )

    @property
    def data_indices(self) -> List[int]:
        if self._data_indices is None:
            self._data_indices = self._calculate_data_indices()
        return self._data_indices

    @property
    def global_shape(self) -> Tuple[int, int, int]:
        if self._global_shape is None:
            self._global_shape = self._calculate_global_shape()
        return self._global_shape
