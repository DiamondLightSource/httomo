FROM mambaorg/micromamba:2.0.8-debian12-slim AS build

RUN micromamba install -y python=3.11 blas[build=mkl] 'numpy<2' scipy scikit-image tqdm pyyaml pillow click cuda-version=12.8
RUN micromamba install -y openmpi=4.1.6 mpi
RUN micromamba install -y h5py[build=*openmpi*] hdf5plugin
RUN micromamba install -y mpi4py
RUN micromamba install -y aiofiles graypy imageio loguru pluggy pytest iniconfig
RUN micromamba install -y tomopy=1.15
RUN micromamba install -y cupy=12.3.0 nvtx
RUN micromamba install -y -c astra-toolbox astra-toolbox
RUN micromamba run python -m pip install ccpi-regularisation-cupy
RUN micromamba run python -m pip install tomobar
RUN micromamba run python -m pip install --no-deps httomolib httomolibgpu httomo-backends
RUN micromamba clean -y --all --force-pkgs-dirs

COPY . .
RUN micromamba run python -m pip install --no-deps .

FROM nvidia/cuda:12.8.1-base-ubuntu22.04 AS deploy

COPY --from=build /opt/conda /opt/conda

ENTRYPOINT ["/opt/conda/bin/python", "-m", "httomo", "run"]
