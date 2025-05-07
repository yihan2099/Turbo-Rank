# ──────────────────────────────────────────────────────────────
# TensorRT 25.04  (CUDA 12.4, Python 3.10)  –  final ≈ 6 GB
# ──────────────────────────────────────────────────────────────
FROM nvcr.io/nvidia/tensorrt:25.04-py3

# 1 — Workspace
WORKDIR /workspace/turbo-rank
ENV NVIDIA_VISIBLE_DEVICES=all
ENV PYTHONUNBUFFERED=1

# 2 — Install third-party Python deps first (max layer-cache hit)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3 — Copy project source after deps (code edits ≠ rebuild deps)
COPY . .

# 4 — Install project in editable mode
RUN pip install --no-cache-dir -e .

# 5 — Build C++ TensorRT backend only
RUN apt-get update -qq && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential cmake && \
    cmake -S deploy -B deploy/build \
          -DBACKEND_TRT=ON  -DBACKEND_ORT=OFF && \
    cmake --build deploy/build --parallel $(nproc) && \
    apt-get purge -y build-essential cmake && \
    apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

# 6 — Default entry-point
CMD ["python", "-m", "deploy.server", "--backend", "tensorrt"]