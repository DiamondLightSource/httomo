#!/usr/bin/env python3
"""Create an NXtomo file that can be loaded by HTTomo.

The module can be imported to pack NumPy arrays, or run as a command-line
program to pack stacks of TIFF files.  Projection data must use the axis order
``(rotation_angle, detector_y, detector_x)``.
"""

from __future__ import annotations

import argparse
import glob
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np


def _nx_group(parent: h5py.Group, name: str, nx_class: str) -> h5py.Group:
    """Create a NeXus group and set its ``NX_class`` attribute."""
    group = parent.create_group(name)
    group.attrs["NX_class"] = nx_class
    return group


def _as_frame_stack(array: np.ndarray, name: str) -> np.ndarray:
    """Return a numeric array with shape ``(frames, detector_y, detector_x)``."""
    array = np.asarray(array)
    if array.ndim == 2:
        array = array[np.newaxis, ...]
    if array.ndim != 3:
        raise ValueError(f"{name} must be a 2D image or a 3D frame stack")
    if 0 in array.shape:
        raise ValueError(f"{name} must not contain an empty dimension")
    if not (
        np.issubdtype(array.dtype, np.integer)
        or np.issubdtype(array.dtype, np.floating)
    ):
        raise TypeError(f"{name} must have a real numeric dtype, not {array.dtype}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return array


# [write-nxtomo-start]
def write_nxtomo(
    output_file: str | Path,
    projections: np.ndarray,
    angles: np.ndarray,
    *,
    flats: np.ndarray | None = None,
    darks: np.ndarray | None = None,
    sample_name: str = "sample",
    compression: str | None = "gzip",
    overwrite: bool = False,
) -> Path:
    """Write NumPy arrays to an HTTomo-compatible NXtomo file.

    Parameters
    ----------
    output_file
        Destination ``.nxs`` or ``.h5`` file.
    projections
        Projection stack with shape ``(angles, detector_y, detector_x)``.
    angles
        One rotation angle in degrees per projection.
    flats, darks
        Optional calibration stacks. A single 2D image is also accepted.
    sample_name
        Descriptive sample name stored in the NXsample group.
    compression
        HDF5 compression: ``"gzip"``, ``"lzf"``, or ``None``.
    overwrite
        Replace ``output_file`` if it already exists.
    """
    projections = _as_frame_stack(projections, "projections")
    angles = np.asarray(angles)
    if angles.ndim != 1 or len(angles) != len(projections):
        raise ValueError("angles must be 1D with one value per projection")
    if (
        not (
            np.issubdtype(angles.dtype, np.integer)
            or np.issubdtype(angles.dtype, np.floating)
        )
        or not np.isfinite(angles).all()
    ):
        raise ValueError("angles must contain finite numeric values")

    optional_stacks = []
    for name, stack in (("darks", darks), ("flats", flats)):
        if stack is None:
            optional_stacks.append(None)
            continue
        stack = _as_frame_stack(stack, name)
        if stack.shape[1:] != projections.shape[1:]:
            raise ValueError(
                f"{name} frame shape {stack.shape[1:]} does not match "
                f"projection frame shape {projections.shape[1:]}"
            )
        optional_stacks.append(stack)
    darks, flats = optional_stacks

    compression = None if compression == "none" else compression
    if compression not in (None, "gzip", "lzf"):
        raise ValueError("compression must be 'gzip', 'lzf', or None")

    stacks = [stack for stack in (darks, flats, projections) if stack is not None]
    output_dtype = np.result_type(*(stack.dtype for stack in stacks))
    dark_count = 0 if darks is None else len(darks)
    flat_count = 0 if flats is None else len(flats)
    projection_count = len(projections)
    frame_count = dark_count + flat_count + projection_count
    detector_y, detector_x = projections.shape[1:]

    image_key = np.concatenate(
        (
            np.full(dark_count, 2, dtype=np.int8),
            np.full(flat_count, 1, dtype=np.int8),
            np.zeros(projection_count, dtype=np.int8),
        )
    )
    # NXtomo requires an angle for every frame. Calibration frames use the
    # first projection angle; HTTomo selects projection angles via image_key.
    frame_angles = np.concatenate(
        (np.full(dark_count + flat_count, angles[0]), angles)
    ).astype(np.float32, copy=False)

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "w-"
    now = datetime.now(timezone.utc).isoformat()

    with h5py.File(output_file, mode) as nexus_file:
        nexus_file.attrs.update(
            default="entry", file_name=output_file.name, file_time=now
        )

        entry = _nx_group(nexus_file, "entry", "NXentry")
        entry.attrs["default"] = "data"
        entry.create_dataset("definition", data="NXtomo")
        entry.create_dataset("title", data=sample_name)
        entry.create_dataset("start_time", data=now)

        instrument = _nx_group(entry, "instrument", "NXinstrument")
        detector = _nx_group(instrument, "detector", "NXdetector")
        data = detector.create_dataset(
            "data",
            shape=(frame_count, detector_y, detector_x),
            dtype=output_dtype,
            chunks=(1, min(detector_y, 256), min(detector_x, 256)),
            compression=compression,
        )
        data.attrs.update(
            interpretation="image",
            axes="frame,detector_y,detector_x",
        )
        key = detector.create_dataset("image_key", data=image_key)
        key.attrs["meaning"] = "0=projection; 1=flat; 2=dark"

        first = 0
        for stack in (darks, flats, projections):
            if stack is not None:
                data[first : first + len(stack)] = stack
                first += len(stack)

        sample = _nx_group(entry, "sample", "NXsample")
        sample.create_dataset("name", data=sample_name)
        rotation_angle = sample.create_dataset("rotation_angle", data=frame_angles)
        rotation_angle.attrs["units"] = "deg"

        # NXdata contains hard links, not copies. These paths are also the
        # locations used by HTTomo's automatic NXtomo discovery.
        nx_data = _nx_group(entry, "data", "NXdata")
        nx_data.attrs["signal"] = "data"
        nx_data.attrs["axes"] = np.asarray(
            ["rotation_angle", ".", "."], dtype=h5py.string_dtype()
        )
        nx_data["data"] = data
        nx_data["image_key"] = key
        nx_data["rotation_angle"] = rotation_angle

        entry.create_dataset("end_time", data=datetime.now(timezone.utc).isoformat())

    return output_file


# [write-nxtomo-end]


def _natural_sort_key(path: str) -> list[tuple[int, int | str]]:
    """Sort numbered TIFF names in human order (for example, 2 before 10)."""
    return [
        (0, int(part)) if part.isdigit() else (1, part.lower())
        for part in re.split(r"(\d+)", path)
    ]


def load_tiff_stack(pattern: str) -> np.ndarray:
    """Load the 2D TIFF files matched by a glob pattern into a 3D array."""
    paths = sorted(glob.glob(pattern), key=_natural_sort_key)
    if not paths:
        raise FileNotFoundError(f"No TIFF files match {pattern!r}")

    try:
        import tifffile
    except ImportError as error:
        raise RuntimeError(
            "Reading TIFF files requires tifffile: python -m pip install tifffile"
        ) from error

    frames = []
    for path in paths:
        frame = tifffile.imread(path)
        if frame.ndim != 2:
            raise ValueError(f"{path} is not a single 2D grayscale image")
        frames.append(frame)
    try:
        return np.stack(frames)
    except ValueError as error:
        raise ValueError(
            "All TIFF images in a stack must have the same shape"
        ) from error


# [write-tiffs-start]
def write_nxtomo_from_tiffs(
    output_file: str | Path,
    projection_pattern: str,
    angles: np.ndarray,
    *,
    flat_pattern: str | None = None,
    dark_pattern: str | None = None,
    **kwargs,
) -> Path:
    """Load TIFF stacks and pass them to :func:`write_nxtomo`."""
    projections = load_tiff_stack(projection_pattern)
    flats = load_tiff_stack(flat_pattern) if flat_pattern else None
    darks = load_tiff_stack(dark_pattern) if dark_pattern else None
    return write_nxtomo(
        output_file,
        projections,
        angles,
        flats=flats,
        darks=darks,
        **kwargs,
    )


# [write-tiffs-end]


def _load_angles(path: Path) -> np.ndarray:
    """Load angles from a NumPy ``.npy`` file or a text/CSV file."""
    if path.suffix.lower() == ".npy":
        return np.load(path)
    delimiter = "," if path.suffix.lower() == ".csv" else None
    return np.loadtxt(path, delimiter=delimiter)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pack TIFF stacks into an HTTomo-compatible NXtomo file."
    )
    parser.add_argument("output_file", type=Path)
    parser.add_argument(
        "--projections",
        required=True,
        metavar="GLOB",
        help="Glob matching projection TIFF files (quote it in the shell)",
    )
    parser.add_argument(
        "--angles",
        required=True,
        type=Path,
        help="One-dimensional angles file (.npy, .txt, or .csv), in degrees",
    )
    parser.add_argument("--flats", metavar="GLOB", help="Glob matching flat TIFFs")
    parser.add_argument("--darks", metavar="GLOB", help="Glob matching dark TIFFs")
    parser.add_argument("--sample-name", default="sample")
    parser.add_argument(
        "--compression", choices=("none", "gzip", "lzf"), default="gzip"
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Command-line entry point."""
    args = _parser().parse_args(argv)
    output_file = write_nxtomo_from_tiffs(
        args.output_file,
        args.projections,
        _load_angles(args.angles),
        flat_pattern=args.flats,
        dark_pattern=args.darks,
        sample_name=args.sample_name,
        compression=args.compression,
        overwrite=args.overwrite,
    )
    print(f"Wrote {output_file}")


if __name__ == "__main__":
    main()
