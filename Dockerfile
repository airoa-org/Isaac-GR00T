FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=on

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl wget git git-lfs \
    build-essential pkg-config \
    python3 python3-venv python3-dev \
    libdav1d7 libaom-dev \
    cmake ninja-build \
    iproute2 iputils-ping net-tools dnsutils \
    lcm \
 && rm -rf /var/lib/apt/lists/* \
 && git lfs install

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /workspace

COPY pyproject.toml ./

RUN uv venv --python=3.10 .venv
ENV VIRTUAL_ENV=/workspace/.venv
ENV PATH="/workspace/.venv/bin:${PATH}"

RUN uv pip install --index-url https://download.pytorch.org/whl/cu121 \
    "torch==2.4.*" "torchvision==0.19.*" "torchaudio==2.4.*"

RUN uv pip install --upgrade setuptools wheel \
 && uv pip install ninja cmake

COPY . .

RUN uv pip install -e ".[base]"

RUN uv pip install --no-build-isolation "flash-attn==2.7.1.post4"

ENV LCM_DEFAULT_URL=udpm://239.255.76.67:7667?ttl=1

EXPOSE 7667/udp

ENV PYTHONPATH=/workspace
CMD ["/bin/bash"]
