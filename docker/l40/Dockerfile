# syntax=docker/dockerfile:1.7
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GDRN_UID=1010
ARG GDRN_GID=1010
ARG GDRN_USER=gdrn
ARG GIT_COMMIT=unknown
ARG GIT_REMOTE=unknown

LABEL org.opencontainers.image.title="GDRNPP managed experiments on L40"
LABEL org.opencontainers.image.revision="${GIT_COMMIT}"
LABEL org.opencontainers.image.source="${GIT_REMOTE}"

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    CUDA_HOME=/usr/local/cuda \
    FORCE_CUDA=1 \
    TORCH_CUDA_ARCH_LIST=8.9 \
    MAX_JOBS=4 \
    VIRTUAL_ENV=/opt/gdrn-venv \
    PATH=/opt/gdrn-venv/bin:/usr/local/cuda/bin:${PATH} \
    PYTHONNOUSERSITE=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    GDRN_GIT_COMMIT=${GIT_COMMIT} \
    GDRN_GIT_REMOTE=${GIT_REMOTE} \
    PYTHONPATH=/workspace/gdrnpp:/opt/bop_toolkit:/opt/bop_renderer/build \
    LD_LIBRARY_PATH=/workspace/gdrnpp/core/csrc/uncertainty_pnp/lib:/opt/gdrn-venv/lib/python3.10/site-packages/torch/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash ca-certificates \
        python3 python3-dev python3-pip python3-venv \
        build-essential cmake ninja-build pkg-config patchelf \
        libassimp-dev libeigen3-dev libgoogle-glog-dev \
        libegl1-mesa-dev libgl1-mesa-dev libglvnd-dev \
        libosmesa6 libosmesa6-dev \
        libglib2.0-0 libx11-6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv "${VIRTUAL_ENV}" \
    && python -m pip install --no-cache-dir pip==24.0 setuptools==69.5.1 wheel==0.43.0

COPY docker/l40/requirements.lock /tmp/requirements.lock
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --no-deps --no-build-isolation --require-hashes \
        --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
        --extra-index-url https://download.pytorch.org/whl/cu121 \
        --requirement /tmp/requirements.lock \
    && python -m pip install --no-cache-dir \
        --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
        pytest==8.4.2 iniconfig==2.1.0 pluggy==1.6.0 pygments==2.19.2 \
    && rm /tmp/requirements.lock

COPY docker/l40/vendor /tmp/vendor
RUN cd /tmp/vendor \
    && sha256sum --check SHA256SUMS \
    && python -m pip install --no-cache-dir --no-deps \
        pytorch_lightning-1.6.4-py3-none-any.whl \
        imgaug-0.4.0-py2.py3-none-any.whl \
        black-25.11.0-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl \
        click-8.1.8-py3-none-any.whl \
        cloudpickle-3.1.2-py3-none-any.whl \
        hydra_core-1.3.4-py3-none-any.whl \
        mypy_extensions-1.1.0-py3-none-any.whl \
        pathspec-1.1.1-py3-none-any.whl \
        pytokens-0.4.1-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl \
        chardet-5.2.0-py3-none-any.whl \
    && mkdir -p /tmp/detectron2 \
    && tar -xzf detectron2-02b5c4e295e990042a714712c21dc79b731e8833.tar.gz --strip-components=1 -C /tmp/detectron2 \
    && python -m pip install --no-cache-dir --no-deps --no-build-isolation /tmp/detectron2 \
    && rm -rf /tmp/detectron2 \
    && mkdir -p /opt/bop_toolkit \
    && tar -xzf bop-toolkit-clean.tgz -C /opt/bop_toolkit

COPY . /workspace/gdrnpp
COPY docker/l40/build_native.sh /usr/local/bin/build-gdrn-native
COPY docker/l40/verify_environment.py /usr/local/bin/verify-gdrn-environment
COPY docker/l40/verify_native.sh /usr/local/bin/verify-gdrn-native

RUN chmod 0755 /usr/local/bin/build-gdrn-native /usr/local/bin/verify-gdrn-environment /usr/local/bin/verify-gdrn-native \
    && /usr/local/bin/build-gdrn-native /tmp/vendor \
    && rm -rf /tmp/vendor \
    && rm -f /workspace/gdrnpp/bop_renderer \
    && ln -s /opt/bop_renderer /workspace/gdrnpp/bop_renderer \
    && mkdir -p /workspace/gdrnpp/datasets /workspace/gdrnpp/pretrained_models /workspace/gdrnpp/output \
    && /usr/local/bin/verify-gdrn-environment --build \
    && /usr/local/bin/verify-gdrn-native

RUN cd /workspace/gdrnpp \
    && PYTHONPATH=/workspace/gdrnpp PYTHONPYCACHEPREFIX=/tmp/gdrnpp-pycache python -m pytest -q -o cache_dir=/tmp/gdrnpp-pytest-cache research/experiment_system/tests research/managed_runtime/tests research/cpm_head/tests research/quality_coverage/tests research/pnp_control/tests research/stage3c_runtime/tests research/pose_head_utilization/tests research/pbr_validation/tests research/oracle_diagnostic/tests research/pose_aggregation/tests

COPY docker/l40/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json
ENV __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json

RUN groupadd --gid "${GDRN_GID}" "${GDRN_USER}" \
    && useradd --uid "${GDRN_UID}" --gid "${GDRN_GID}" --create-home --shell /bin/bash "${GDRN_USER}" \
    && chown -R "${GDRN_UID}:${GDRN_GID}" /home/${GDRN_USER} /workspace /opt/bop_renderer /opt/bop_toolkit

WORKDIR /workspace/gdrnpp
USER ${GDRN_USER}
CMD ["bash"]
