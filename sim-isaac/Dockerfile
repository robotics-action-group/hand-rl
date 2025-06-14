
FROM nvcr.io/nvidia/isaac-lab:2.1.0 AS base

# ARG DOCKER_PROJ_PATH_ARG
# ENV DOCKER_PROJ_PATH=/workspace/c3po

USER root

# Copy the source code directory (files to exclude are defined in .dockerignore)
# COPY ../ ${DOCKER_PROJ_PATH}

# # Install whatever you need as additional dependencies.
RUN bash -i -c "source ${HOME}/.bashrc && \
    apt-get update && \
    apt-get install -y nano htop && \
    pip install --upgrade pip && \
    pip install ipykernel wandb moviepy pytest hf_xet tensordict torchrl"
    # cd ${DOCKER_PROJ_PATH}/source/c3po && \
    # pip install -e ."

# make working directory as the Isaac Lab directory
# this is the default directory when the container is run
# WORKDIR /workspace/c3po

# COPY entrypoint.sh /usr/local/bin
# # RUN chmod +x /workspace/entrypoint.sh

# RUN chmod a+x /usr/local/bin/entrypoint.sh \
#   && ln -s /usr/local/bin/entrypoint.sh /

# ENTRYPOINT ["bash", "-i", "entrypoint.sh"]
