from pathlib import Path
from typing import Literal, Optional, Tuple

import click
import nxtomo
import numpy as np
from nxtomo.nxobject.nxdetector import ImageKey
from skimage.data import shepp_logan_phantom
from skimage.transform import radon


@click.command()
@click.argument("filename", type=click.STRING)
@click.argument("data_path", type=click.STRING)
@click.argument(
    "out_dir",
    type=click.Path(exists=True, file_okay=False, writable=True, path_type=Path),
)
@click.option(
    "--no-of-angles",
    type=click.INT,
    default=None,
    help="Number of rotation angles between 0 and 180 degrees",
)
@click.option(
    "--no-of-darks",
    type=click.INT,
    default=20,
    help="Number of dark field images to generate",
)
@click.option(
    "--no-of-flats",
    type=click.INT,
    default=20,
    help="Number of flat field images to generate",
)
def main(
    filename: str,
    data_path: str,
    out_dir: Path,
    no_of_angles: Optional[int] = None,
    no_of_darks: int = 20,
    no_of_flats: int = 20,
):
    (projections, angles) = generate_projections(no_of_angles)
    darks = generate_darks(no_of_darks, projections.shape[1], projections.shape[2])
    flats = generate_flats(no_of_flats, projections.shape[1], projections.shape[2])
    tomo_entry = nxtomo.NXtomo()
    tomo_entry.instrument.detector.data = np.concatenate([darks, flats, projections])
    tomo_entry.instrument.detector.image_key_control = np.concatenate(
        [
            [ImageKey.DARK_FIELD] * no_of_darks,
            [ImageKey.FLAT_FIELD] * no_of_flats,
            [ImageKey.PROJECTION] * angles.shape[0],
        ]
    )
    tomo_entry.sample.rotation_angle = np.concatenate(
        [np.zeros(no_of_darks + no_of_flats), angles]
    )
    tomo_entry.save(file_path=str(out_dir / filename), data_path=data_path)


def generate_projections(
    no_of_angles: Optional[int],
) -> Tuple[
    np.ndarray[tuple[Literal[3]], np.dtype[np.uint16]],
    np.ndarray[tuple[Literal[1]], np.dtype[np.float32]],
]:
    phantom = shepp_logan_phantom()
    no_of_angles = no_of_angles if no_of_angles is not None else max(phantom.shape)
    angles = np.linspace(0, 180, no_of_angles, endpoint=False, dtype=np.float32)
    sinogram = radon(image=phantom, theta=angles)
    sinograms = np.asarray([sinogram] * 20, dtype=np.uint16)
    projections = np.swapaxes(sinograms, 2, 0)
    projections = np.swapaxes(projections, 1, 2)
    return projections, angles


def generate_darks(
    number: int, height: int, width: int
) -> np.ndarray[tuple[Literal[3]], np.dtype[np.uint16]]:
    return np.zeros((number, height, width), dtype=np.uint16)


def generate_flats(
    number: int, height: int, width: int
) -> np.ndarray[tuple[Literal[3]], np.dtype[np.uint16]]:
    return np.ones((number, height, width), dtype=np.uint16)


if __name__ == "__main__":
    main()
