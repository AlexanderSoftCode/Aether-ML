# ==============================================================================
# Aether-ML ROCm 7.14 Container
# Target GPU Architecture: RDNA4 (gfx1201)
# Runtime Capabilities: rocWMMA / Matrix Cores, Python 3.14
# ==============================================================================

FROM ubuntu:26.04

ENV DEBIAN_FRONTEND=noninteractive

# Target architecture configuration
ENV ROCM_HOME=/opt/rocm \
    ROCM_PATH=/opt/rocm \
    HIP_PATH=/opt/rocm \
    AMDGPU_TARGETS=gfx1201 \
    HCC_AMDGPU_TARGET=gfx1201 \
    GPU_TARGETS=gfx1201 \
    PATH="/opt/rocm/bin:${PATH}"

ENV CUPY_CACHE_DIR=/app/.devcontainer/.cupy_cache

# ------------------------------------------------------------------------------
# Step 1: Install Base OS Packages & Setup ROCm 7.14 Apt Repository
# ------------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    wget \
    gnupg \
    build-essential \
    git \
    sudo \
    gosu \
    && mkdir --parents --mode=0755 /etc/apt/keyrings \
    && wget https://repo.amd.com/rocm/packages-multi-arch/gpg/rocm.gpg -O - | \
       gpg --dearmor | \
       tee /etc/apt/keyrings/amdrocm.gpg > /dev/null \
    && echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/amdrocm.gpg] https://repo.amd.com/rocm/packages-multi-arch/ubuntu2604 stable main" \
       > /etc/apt/sources.list.d/rocm.list \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------------------------
# Step 2: Install ROCm Dev Meta-Package & Python 3.14
# ------------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    amdrocm-core-dev7.14-gfx1201 \
    python3.14 \
    python3.14-dev \
    python3.14-venv \
    && rm -rf /var/lib/apt/lists/*

RUN find /opt/rocm/ -name "rocwmma.hpp" | grep -q . || \
    (echo "ERROR: rocwmma.hpp not found under /opt/rocm/" && exit 1)

# ------------------------------------------------------------------------------
# Step 3: Python Virtual Environment & CuPy Build
# ------------------------------------------------------------------------------
ENV VIRTUAL_ENV=/opt/venv
RUN python3.14 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:${PATH}"

# 3a: Base build dependencies and utilities and normal prereqs
RUN pip install --no-cache-dir --upgrade pip setuptools wheel cython
RUN pip install --no-cache-dir numpy==2.5.3 safetensors==0.8.0 jupyterlab

# 3b: Compile CuPy from source targeting ROCm HIP gfx1201
ENV CUPY_INSTALL_USE_HIP=1
RUN pip install --no-cache-dir --no-binary cupy cupy==14.2.0

# 3c: Validation check
# NOTE: keep `from` off the start of a continuation line - the devcontainer CLI
# parses Dockerfiles line-by-line and reads a leading `from` as a FROM instruction.
RUN python -c "import cupy, safetensors, numpy; from cupy_backends.cuda.api import runtime; \
assert runtime.is_hip, 'cupy was NOT built against ROCm/HIP (CUPY_INSTALL_USE_HIP ignored?)'; \
print(f'Build check passed: CuPy {cupy.__version__} (HIP), SafeTensors {safetensors.__version__}, NumPy {numpy.__version__}')"

# ------------------------------------------------------------------------------
# Step 4: User Setup & Restricted Sudo
# ------------------------------------------------------------------------------
RUN mkdir -p /app \
    && echo "ubuntu ALL=(root) NOPASSWD: /usr/local/bin/fix-gpu-groups.sh" > /etc/sudoers.d/ubuntu \
    && chmod 0440 /etc/sudoers.d/ubuntu \
    && chown -R ubuntu:ubuntu /app /opt/venv

COPY --chmod=0755 .devcontainer/rocm-gfx1201/fix-gpu-groups.sh /usr/local/bin/fix-gpu-groups.sh
COPY --chmod=0755 .devcontainer/rocm-gfx1201/entrypoint.sh /usr/local/bin/entrypoint.sh

RUN for f in /usr/local/bin/fix-gpu-groups.sh /usr/local/bin/entrypoint.sh; do \
        [ -x "$f" ] || { echo "ERROR: $f missing or not executable (check /.dockerignore)" >&2; exit 1; }; \
    done; \
    command -v gosu > /dev/null || { echo "ERROR: gosu not on PATH; entrypoint.sh privilege drop would fail" >&2; exit 1; }

WORKDIR /app

EXPOSE 8888
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--no-browser"]
