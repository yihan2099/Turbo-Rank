FROM nvcr.io/nvidia/tensorrt:25.04-py3

WORKDIR /workspace/turbo-rank
ENV NVIDIA_VISIBLE_DEVICES=all
ENV PYTHONUNBUFFERED=1

# ---------- 1. third-party deps ----------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---------- 2. project source ----------
COPY . .

# ---------- 3. install project ----------
RUN pip install --no-cache-dir -e .

# ---------- 4. Install ONNX Runtime ----------
ENV ORT_VERSION=1.21.1
ENV ORT_ROOT=/usr/local/onnxruntime-${ORT_VERSION}
ENV ORT_DIR=onnxruntime-linux-x64-gpu-${ORT_VERSION}
ENV ORT_URL=https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${ORT_DIR}.tgz

RUN mkdir -p ${ORT_ROOT} && \
    curl -fL ${ORT_URL} -o /tmp/onnxruntime.tgz && \
    tar -C ${ORT_ROOT} -xzf /tmp/onnxruntime.tgz && \
    ln -sf ${ORT_ROOT}/${ORT_DIR}/include /usr/local/include/onnxruntime && \
    ln -sf ${ORT_ROOT}/${ORT_DIR}/lib/libonnxruntime.so* /usr/local/lib/ && \
    ldconfig && \
    rm /tmp/onnxruntime.tgz

# ---------- 5. build C++ ONNX Runtime binary ----------
RUN apt-get update -qq && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential cmake && \
    # Ensure the build directory is clean
    rm -rf deploy/build && \
    mkdir -p deploy/build && \
    # Run CMake with verbose output
    # cmake -S deploy -B deploy/build \
    #     -DBACKEND_ORT=ON  -DBACKEND_TRT=OFF \
    #     -DORT_ROOT=${ORT_ROOT}/${ORT_DIR} \
    #     -DORT_INCLUDE_DIR=${ORT_ROOT}/${ORT_DIR}/include \
    #     -DORT_LIB=${ORT_ROOT}/${ORT_DIR}/lib/libonnxruntime.so.${ORT_VERSION} && \
    # cmake --build deploy/build --parallel $(nproc) --verbose && \
    apt-get purge -y build-essential cmake && \
    apt-get autoremove -y && rm -rf /var/lib/apt/lists/*


# Copy entrypoint script
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# # Default entry-point
# ENTRYPOINT ["/entrypoint.sh"]
# CMD ["--backend", "onnxrt"]