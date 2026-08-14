FROM mambaorg/micromamba:2.0.8-debian12-slim AS build

RUN micromamba install -y -c astra-toolbox -c conda-forge python=3.12 blas[build=mkl] 'numpy=2.4.*' tqdm pyyaml pillow click gcc openmpi=4.1.6 mpi h5py[build=*openmpi*] hdf5plugin mpi4py aiofiles graypy imageio loguru iniconfig tomopy=1.15.3 scikit-image cuda-version=12.9 cuda-cudart==12.9.* cupy=14.0.* nvtx astra-toolbox::astra-toolbox
RUN micromamba run python -m pip install --no-cache-dir --no-deps tomobar httomolib httomolibgpu httomo-backends
RUN micromamba uninstall -n base -y gcc
RUN micromamba clean -y --all --force-pkgs-dirs

COPY . .
RUN micromamba run python -m pip install --no-cache-dir --no-deps .

FROM nvidia/cuda:12.9.1-base-ubuntu24.04 AS deploy

COPY --from=build /opt/conda /opt/conda
ENV CONDA_PREFIX=/opt/conda
ENV OMPI_MCA_plm_rsh_agent=

ENTRYPOINT ["/opt/conda/bin/python", "-m", "httomo", "run"]
