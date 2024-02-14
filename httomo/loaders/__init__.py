from mpi4py import MPI

from httomo.darks_flats import DarksFlatsFileConfig
from httomo.loaders.standard_tomo_loader import RawAngles, StandardLoaderWrapper
from httomo.runner.loader import LoaderInterface
from httomo.runner.methods_repository_interface import MethodRepository


def make_loader(
    repo: MethodRepository, module_path: str, method_name: str, comm: MPI.Comm, **kwargs
) -> LoaderInterface:
    """Produces a loader interface. Only StandardTomoWrapper is supported right now,
    and this method has been added for backwards compatibility. Supporting other loaders
    is a topic that still needs to be explored."""

    if "standard_tomo" not in method_name:
        raise NotImplementedError(
            "Only the standard_tomo loader is currently supported"
        )

    # the following will raise KeyError if not present
    in_file = kwargs["in_file"]
    data_path = kwargs["data_path"]
    image_key_path = kwargs["image_key_path"]
    rotation_angles = kwargs["rotation_angles"]
    angles_path = rotation_angles["data_path"]
    # these will have defaults if not given
    darks: dict = kwargs.get("darks", dict())
    darks_file = darks.get("file", in_file)
    darks_path = darks.get("data_path", data_path)
    darks_image_key = darks.get("image_key_path", image_key_path)
    flats: dict = kwargs.get("darks", dict())
    flats_file = flats.get("file", in_file)
    flats_path = flats.get("data_path", data_path)
    flats_image_key = flats.get("image_key_path", image_key_path)
    # TODO: handle these
    dimension = int(kwargs.get("dimension", 1)) - 1
    preview = kwargs.get("preview", (None, None, None))
    pad = int(kwargs.get("pad", 0))

    return StandardLoaderWrapper(
        comm,
        in_file=in_file,
        data_path=data_path,
        image_key_path=image_key_path,
        darks=DarksFlatsFileConfig(
            file=darks_file, data_path=darks_path, image_key_path=darks_image_key
        ),
        flats=DarksFlatsFileConfig(
            file=flats_file, data_path=flats_path, image_key_path=flats_image_key
        ),
        angles=RawAngles(data_path=angles_path),
    )
