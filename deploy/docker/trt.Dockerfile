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

# 5 — Verify TensorRT installation and set up environment
RUN echo "Verifying TensorRT installation..." && \
    python -c "import tensorrt as trt; print('TensorRT successfully imported')" && \
    echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu" >> /etc/profile && \
    echo "export LIBRARY_PATH=\$LIBRARY_PATH:/usr/lib/x86_64-linux-gnu" >> /etc/profile && \
    echo "export CPATH=\$CPATH:/usr/include" >> /etc/profile && \
    echo "TensorRT is properly installed and configured."

# 6 — Build C++ TensorRT backend only
RUN apt-get update -qq && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential cmake && \
    # Ensure the build directory is clean
    rm -rf deploy/build && \
    mkdir -p deploy/build && \
    # Run CMake with verbose output
    # cmake -S deploy -B deploy/build \
    #       -DBACKEND_TRT=ON  -DBACKEND_ORT=OFF && \
    # cmake --build deploy/build --parallel $(nproc) --verbose && \
    apt-get purge -y build-essential cmake && \
    apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

# Copy entrypoint script
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# # Default entry-point
# ENTRYPOINT ["/entrypoint.sh"]
# CMD ["--backend", "tensorrt"]