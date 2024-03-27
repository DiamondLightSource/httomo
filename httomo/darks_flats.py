from pathlib import Path
from typing import NamedTuple, Optional, Tuple

import h5py
import numpy as np

from httomo.preview import PreviewConfig


class DarksFlatsFileConfig(NamedTuple):
    file: Path
    data_path: str
    image_key_path: Optional[str]


def get_darks_flats(
    darks_config: DarksFlatsFileConfig,
    flats_config: DarksFlatsFileConfig,
    preview_config: PreviewConfig,
) -> Tuple[np.ndarray, np.ndarray]:
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
