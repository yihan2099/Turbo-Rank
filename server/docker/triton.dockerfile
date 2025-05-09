# Dockerfile.triton
FROM nvcr.io/nvidia/tritonserver:24.06-py3

# Copy your model repository into the container
# (assumes you have a local folder `models/`)
# COPY server/models /models

# Expose Triton ports
EXPOSE 8000 8001 8002

# Launch Triton pointing at /models
ENTRYPOINT [ "tritonserver", "--model-repository=/models" ]