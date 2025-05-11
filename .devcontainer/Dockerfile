# ──────────────────────────────────────────────────────────────
# Development container for Turbo-Rank with TensorRT and ONNX Runtime
# ──────────────────────────────────────────────────────────────
FROM nvcr.io/nvidia/tensorrt:25.04-py3

# Environment variables
WORKDIR /workspace
ENV NVIDIA_VISIBLE_DEVICES=all
ENV PYTHONUNBUFFERED=1

# Install development tools
RUN apt-get update -qq && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    vim \
    nano \
    python3-pip \
    python3-dev \
    sudo \
    openssh-client \
    libssl-dev \
    libffi-dev \
    locales && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Configure locale
RUN localedef -i en_US -c -f UTF-8 -A /usr/share/locale/locale.alias en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LC_ALL en_US.UTF-8

# Install ONNX Runtime for GPU
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

# Setup TensorRT environment variables
RUN echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu" >> /etc/profile && \
    echo "export LIBRARY_PATH=\$LIBRARY_PATH:/usr/lib/x86_64-linux-gnu" >> /etc/profile && \
    echo "export CPATH=\$CPATH:/usr/include" >> /etc/profile

# Create a non-root user with sudo access
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Check if group exists and create only if it doesn't
RUN if getent group $USER_GID > /dev/null 2>&1; then \
    groupmod -n $USERNAME $(getent group $USER_GID | cut -d: -f1); \
    else \
    groupadd --gid $USER_GID $USERNAME; \
    fi && \
    # Check if user exists and create only if it doesn't
    if getent passwd $USER_UID > /dev/null 2>&1; then \
    usermod -l $USERNAME -d /home/$USERNAME -m $(getent passwd $USER_UID | cut -d: -f1); \
    else \
    useradd --uid $USER_UID --gid $USER_GID -m $USERNAME; \
    fi && \
    echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Set up project directory with correct permissions
RUN mkdir -p /workspace/turbo-rank && chown -R $USERNAME:$USERNAME /workspace || \
    mkdir -p /workspace/turbo-rank && chown -R 1000:1000 /workspace

# Switch to non-root user for subsequent commands
USER $USERNAME
ENV HOME=/home/$USERNAME

# Install Python development tools
RUN pip install --user --no-cache-dir \
    pylint \
    black \
    isort \
    mypy \
    pytest \
    pytest-cov \
    ipython \
    jupyter \
    notebook

# Set shell to bash
SHELL ["/bin/bash", "-c"] 