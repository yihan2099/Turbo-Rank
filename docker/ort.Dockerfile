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

# ---------- 4. build C++ TensorRT binary ----------
RUN apt-get update -qq && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential cmake && \
    cmake -S deploy -B deploy/build \
        -DBACKEND_ORT=ON  -DBACKEND_TRT=OFF \
        -DORT_ROOT=/usr/local/onnxruntime-1.21.1/onnxruntime-linux-x64-gpu-1.21.1/ \
        -DORT_INCLUDE_DIR=/usr/local/onnxruntime-1.21.1/onnxruntime-linux-x64-gpu-1.21.1/include \
        -DORT_LIB=/usr/local/onnxruntime-1.21.1/onnxruntime-linux-x64-gpu-1.21.1/lib/libonnxruntime.so.1.21.1 && \
    cmake --build deploy/build --parallel $(nproc) && \
    apt-get purge -y build-essential cmake && \
    apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

CMD ["python", "-m", "deploy.server", "--backend", "tensorrt"]