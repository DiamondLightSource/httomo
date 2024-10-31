"""
Dark-field/flat-field storage location configuration type and reading function, used by
loaders.
"""

from pathlib import Path
from typing import NamedTuple, Optional, Tuple

import h5py
import numpy as np

from httomo.preview import PreviewConfig


class DarksFlatsFileConfig(NamedTuple):
    """
    Configuration for where dark-field or flat-field images associated with the projection data
    are stored.

    Notes
    -----

    There are currently two supported configurations for where dark-field or flat-field images
    can be loaded from:

    1. Dark-field and flat-field images are stored in the same file as the projection images,
    and in the same dataset in that file.

    An image key dataset is present in the file to indicate (by values 0, 1, or 2) which images
    in the dataset are projections (0), darks (1), or flats (2). For this case, the
    `image_key_path` should be provided.

    Please see the NeXuS format manual for more information:
    https://manual.nexusformat.org/classes/base_classes/NXdetector.html#nxdetector-image-key-field

    2. Dark-field and flat-field images are stored in separate files from the file containing
    the projection images.

    Typically this case is such that dark-field images are in a dataset in one file, and
    flat-field images are in a dataset in another file. Furthermore, the dataset that either
    dark-field or flat-field images are stored in contains only one kind of image (ie, the
    dataset containing dark-field images will contain only dark-field images and nothing else).

    Therefore, an image key is not needed to distinguish between projection, dark-field, or
    flat-field images. For this case, the `image_key_path` should be given as `None`.
    """

    file: Path
    data_path: str
    image_key_path: Optional[str]


def get_darks_flats(
    darks_config: DarksFlatsFileConfig,
    flats_config: DarksFlatsFileConfig,
    preview_config: PreviewConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reads the dark-field and flat-field images from the file(s) defined in the input
    configurations.

    Parameters
    ----------
    darks_config : DarksFlatsFileConfig
        Configuration of where to get the dark-field images from.
    flats_config : DarksFlatsFileConfig
        Configuration of where to get the flat-field images from.
    preview_config : PreviewConfig
        Configuration of previewing to be applied to projection, dark-field, and flat-field
        images.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Index zero contains the dark-field images and index one contains the flat-field images.
    """

    def get_together() -> Tuple[np.ndarray, np.ndarray]:
        with h5py.File(darks_config.file, "r") as f:
            darks_indices = np.where(f[darks_config.image_key_path][:] == 2)[0]
            flats_indices = np.where(f[flats_config.image_key_path][:] == 1)[0]
            dataset: h5py.Dataset = f[darks_config.data_path]
            darks = dataset[
                darks_indices,
                preview_config.detector_y.start : preview_config.detector_y.stop,
                preview_config.detector_x.start : preview_config.detector_x.stop,
            ]
            flats = dataset[
                flats_indices,
                preview_config.detector_y.start : preview_config.detector_y.stop,
                preview_config.detector_x.start : preview_config.detector_x.stop,
            ]
        return darks, flats

    def get_separate(config: DarksFlatsFileConfig):
        with h5py.File(config.file, "r") as f:
            return f[config.data_path][
                :,
                preview_config.detector_y.start : preview_config.detector_y.stop,
                preview_config.detector_x.start : preview_config.detector_x.stop,
            ]

    if darks_config.file != flats_config.file:
        darks = get_separate(darks_config)
        flats = get_separate(flats_config)
        return darks, flats

    return get_together()
