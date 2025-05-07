# Development Container for Turbo-Rank

This directory contains configuration for a VSCode Development Container that supports both ONNX Runtime (ORT) and TensorRT (TRT) development for Turbo-Rank.

## Features

- Based on NVIDIA's TensorRT 25.04 container (CUDA 12.4, Python 3.10)
- Pre-installed ONNX Runtime 1.21.1 with GPU support
- Development tools for C++ and Python
- GPU acceleration support
- Non-root user setup with sudo access
- Common extensions for Python and C++ development

## Requirements

- Docker installed on your host machine
- NVIDIA Container Toolkit (for GPU support)
- Visual Studio Code with the Remote - Containers extension

## Usage

1. Open this repository in Visual Studio Code
2. When prompted, click "Reopen in Container" or run the "Remote-Containers: Reopen in Container" command from the command palette
3. VSCode will build the container and open the workspace inside it
4. All development tools and dependencies will be available in the container

## Building C++ Components

To build the C++ components for either backend:

### For TensorRT Backend:

```bash
cd /workspace/turbo-rank
cmake -S deploy -B deploy/build -DBACKEND_TRT=ON -DBACKEND_ORT=OFF
cmake --build deploy/build --parallel $(nproc)
```

### For ONNX Runtime Backend:

```bash
cd /workspace/turbo-rank
cmake -S deploy -B deploy/build -DBACKEND_ORT=ON -DBACKEND_TRT=OFF -DORT_ROOT=$ORT_ROOT
cmake --build deploy/build --parallel $(nproc)
```

## Customization

You can customize the container by modifying:

- `Dockerfile`: Add additional system packages or configuration
- `devcontainer.json`: Adjust VS Code settings, extensions, or container options

## Troubleshooting

If you encounter issues with GPU access, make sure:
1. NVIDIA drivers are properly installed on the host
2. NVIDIA Container Toolkit is correctly configured
3. Your user has permissions to access the GPU 