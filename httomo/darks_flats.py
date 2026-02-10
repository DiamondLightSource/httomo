"""
Dark-field/flat-field storage location configuration type and reading function, used by
loaders.
"""

from enum import Enum
from pathlib import Path
from typing import NamedTuple, Optional, Tuple

import h5py
import numpy as np

from httomo.preview import PreviewConfig


class ImageType(Enum):
    """
    Darks/flats image types mapped to their associated integer value with respect to an image
    key.

    Notes
    -----
    See the following link for more information on an "image key" in the context of the NXtomo
    application definition:
    https://manual.nexusformat.org/classes/base_classes/NXdetector.html#nxdetector-image-key-field.
    """

    Flat = 1
    Dark = 2


class DarksFlatsFileConfig(NamedTuple):
    """
    Configuration for where dark-field or flat-field images associated with the projection data
    are stored.

    Notes
    -----

    There are currently **five** supported configurations for where dark-field or flat-field images
    can be loaded **or** ignored:

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

    3. Dark-field and flat-field images are stored in separate files with own unique or identical image keys.
    This can be a new dataset or two different dataset. Therefore, the image_key_path parameter should be provided
    for both flats and darks.

    4. Dark-field or flat-field images are not available in the dataset, yet some processing of data is still required
    (e.g., the case of applying distortion correction only).

    5. Dark-field or flat-field images are available in the dataset but they need to be ignored, for instance to proceed
    with the other types of data correction or reconstruction avoiding normalisation to d/f.
    """

    file: Path
    data_path: str
    image_key_path: Optional[str]
    ignore: bool = False


def get_darks_flats(
    darks_config: DarksFlatsFileConfig,
    flats_config: DarksFlatsFileConfig,
    preview_config: PreviewConfig,
    datatype: np.dtype,
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
    datatype: np.dtype
        Data type for any dummy darks/flats that need to be generated

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Index zero contains the dark-field images and index one contains the flat-field images.
    """

    def get_together(
        darks_config: DarksFlatsFileConfig, flats_config: DarksFlatsFileConfig
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get darks and flats from the same file and the same dataset within that file.

        Notes
        -----
        If the dataset contains no images marked as darks/flats, then dummy darks/flats will be
        generated accordingly.
        """
        with h5py.File(darks_config.file, "r") as f:
            dataset: h5py.Dataset = f[darks_config.data_path]
            if darks_config.image_key_path is not None:
                darks_indices = np.where(
                    f[darks_config.image_key_path][:] == ImageType.Dark.value
                )[0]
            else:
                darks_indices = []
            if flats_config.image_key_path is not None:
                flats_indices = np.where(
                    f[flats_config.image_key_path][:] == ImageType.Flat.value
                )[0]
            else:
                flats_indices = []

            if len(darks_indices) == 0:
                # there are no darks in the data file so we generate a dummy array
                darks = generate_dummy(ImageType.Dark)
            else:
                darks = dataset[
                    darks_indices,
                    preview_config.detector_y.start : preview_config.detector_y.stop,
                    preview_config.detector_x.start : preview_config.detector_x.stop,
                ]
            if len(flats_indices) == 0:
                # there are no flats in the data file so we generate a dummy array
                flats = generate_dummy(ImageType.Flat)
            else:
                flats = dataset[
                    flats_indices,
                    preview_config.detector_y.start : preview_config.detector_y.stop,
                    preview_config.detector_x.start : preview_config.detector_x.stop,
                ]
        return darks, flats

    def get_separate(config: DarksFlatsFileConfig, image_type: ImageType) -> np.ndarray:
        """
        Get darks/flats from a separate file that may or may not contain an image key.
        """
        with h5py.File(config.file, "r") as f:
            if config.image_key_path is None:
                return f[config.data_path][
                    :,
                    preview_config.detector_y.start : preview_config.detector_y.stop,
                    preview_config.detector_x.start : preview_config.detector_x.stop,
                ]

            indices = np.where(f[config.image_key_path][:] == image_type.value)[0]
            return f[config.data_path][
                indices,
                preview_config.detector_y.start : preview_config.detector_y.stop,
                preview_config.detector_x.start : preview_config.detector_x.stop,
            ]

    def generate_dummy(image_type: ImageType) -> np.ndarray:
        """
        Generate dummy darks or flats data.
        """
        if image_type == ImageType.Dark:
            return np.zeros(
                (
                    1,
                    preview_config.detector_y.stop - preview_config.detector_y.start,
                    preview_config.detector_x.stop - preview_config.detector_x.start,
                ),
                dtype=datatype,
            )
        else:
            return np.ones(
                (
                    1,
                    preview_config.detector_y.stop - preview_config.detector_y.start,
                    preview_config.detector_x.stop - preview_config.detector_x.start,
                ),
                dtype=datatype,
            )

    if darks_config.ignore and flats_config.ignore:
        return generate_dummy(ImageType.Dark), generate_dummy(ImageType.Flat)

    if darks_config.ignore and not flats_config.ignore:
        return generate_dummy(ImageType.Dark), get_separate(
            flats_config, ImageType.Flat
        )

    if not darks_config.ignore and flats_config.ignore:
        return get_separate(darks_config, ImageType.Dark), generate_dummy(
            ImageType.Flat
        )

    if (
        not darks_config.ignore
        and not flats_config.ignore
        and darks_config.file != flats_config.file
    ):
        darks = get_separate(darks_config, ImageType.Dark)
        flats = get_separate(flats_config, ImageType.Flat)
        return darks, flats

    if (
        not darks_config.ignore
        and not flats_config.ignore
        and darks_config.file == flats_config.file
    ):
        return get_together(darks_config, flats_config)

    raise ValueError("Unsupported darks and/or flats config")
