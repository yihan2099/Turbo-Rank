# Turbo-Rank Deployment

This directory contains the deployment infrastructure for Turbo-Rank, providing high-performance inference capabilities through multiple backend options.

## Architecture Overview

The deployment strategy consists of:

- **ONNX Runtime Backend**: Lightweight, cross-platform inference using the ONNX Runtime library
- **TensorRT Backend**: High-performance, GPU-optimized inference using NVIDIA TensorRT
- **gRPC Server**: Python-based service that exposes model inference capabilities over gRPC

## Components

- `server.py`: gRPC server that delegates inference requests to the native backends
- `infer_ort.cpp`: ONNX Runtime-based inference implementation
- `infer_trt.cpp`: TensorRT-based inference implementation
- `CMakeLists.txt`: Build configuration for the native backends
- `Makefile`: Utility scripts for installing dependencies and building backends
- `scripts/`: Helper scripts for Docker deployment

## Prerequisites

- CUDA Toolkit (for TensorRT backend)
- CMake 3.18+
- C++17 compatible compiler
- Python 3.8+

## Installation

### ONNX Runtime

```bash
# Install ONNX Runtime
make install-ort
```

### TensorRT

```bash
# Install TensorRT
make install-trt
```

## Building

```bash
# Build both backends
make all

# Or build specific backends
make ort
make trt
```

The executables will be created in the `build/` directory.

## Running

### Direct Execution

```bash
# Run ONNX Runtime backend
make run-ort

# Run TensorRT backend
make run-trt
```

### gRPC Server

```bash
# Start the server with ONNX Runtime backend
python server.py --backend onnxrt --model /path/to/model.onnx --port 50051

# Start the server with TensorRT backend
python server.py --backend tensorrt --model /path/to/model.onnx --port 50051
```

## Docker Deployment

Docker deployment is supported through the provided scripts:

```bash
# Build and start TensorRT container
./scripts/build_dockers.sh

# Or manually with Docker Compose
docker compose up --build trt
docker compose up -d trt

docker compose up --build ort
docker compose up -d ort
```

## Performance Considerations

- TensorRT backend provides the highest throughput on NVIDIA GPUs with FP16 optimization
- ONNX Runtime is more portable and supports various hardware accelerators
- For production deployment, consider using the TensorRT backend when possible

## Customization

You can modify the inference parameters in the C++ files:
- Batch size (B)
- History length (H)
- Sequence length (L)

Adjust these parameters to match your model and throughput requirements.
