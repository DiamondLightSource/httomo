FROM mambaorg/micromamba:2.0.8-debian12-slim AS build

RUN micromamba install -y python=3.12 blas[build=mkl] 'numpy=2.4.*' scipy tqdm pyyaml pillow click gcc
RUN micromamba install -y openmpi=4.1.6 mpi
RUN micromamba install -y h5py[build=*openmpi*] hdf5plugin
RUN micromamba install -y mpi4py
RUN micromamba install -y aiofiles graypy imageio loguru pluggy pytest iniconfig
RUN micromamba install -y tomopy=1.15.3 scikit-image
RUN micromamba install -y cuda-version=12.9 cuda-cudart==12.9.* cupy=14.0.* nvtx
RUN micromamba install -y -c astra-toolbox -c conda-forge astra-toolbox::astra-toolbox
RUN micromamba run python -m pip install tomobar
RUN micromamba run python -m pip install --no-deps httomolib httomolibgpu httomo-backends
RUN micromamba clean -y --all --force-pkgs-dirs

COPY . .
RUN micromamba run python -m pip install --no-deps .

FROM nvidia/cuda:12.9.1-base-ubuntu24.04 AS deploy

COPY --from=build /opt/conda /opt/conda
ENV CONDA_PREFIX=/opt/conda

ENTRYPOINT ["/opt/conda/bin/python", "-m", "httomo", "run"]
