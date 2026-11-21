FROM mambaorg/micromamba:1.5-focal-cuda-11.7.1

USER root

RUN apt update && apt install -y git && apt clean

USER $MAMBA_USER

COPY --chown=$MAMBA_USER:$MAMBA_USER ./environment.yml /home/env.yaml

RUN micromamba install -y -n base -f /home/env.yaml && \
    micromamba clean --all --yes
