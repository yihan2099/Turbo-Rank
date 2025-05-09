# Turbo-Rank Server with Triton Inference Server

This directory contains configuration and deployment files for serving Turbo-Rank models using NVIDIA Triton Inference Server.

## Getting Started

### Prerequisites

- Docker
- NVIDIA GPU with CUDA support
- NVIDIA Docker runtime

### Building the Docker Image

```bash
./build_docker.sh
```

### Directory Structure

- `docker/`: Contains Dockerfile and related configuration files
- `models/`: Contains model configurations and model files
  - `onnx/`: ONNX model repository

## Model Configuration

### Basic Configuration

A Triton model configuration (`config.pbtxt`) defines how the model should be served. Here's an example from our ONNX model:

```
name: "onnx"
platform: "onnxruntime_onnx"
max_batch_size: 8

input [
  {
    name: "candidate_tokens"
    data_type: TYPE_INT64
    dims: [ -1 ]
  },
  {
    name: "history_tokens"
    data_type: TYPE_INT64
    dims: [ -1, -1 ]
  }
]

output [
  {
    name: "score"
    data_type: TYPE_FP32
    reshape: { shape: [ ] }
    dims: [ 1 ]
  }
]

instance_group [
  { kind: KIND_GPU count: 1 }
]

dynamic_batching {
  preferred_batch_size: [ 1, 4, 8 ]
  max_queue_delay_microseconds: 100
}
```

### Dynamic Batching

Dynamic batching allows Triton to group inference requests to improve throughput. Configure it using:

```
dynamic_batching {
  preferred_batch_size: [ 1, 4, 8 ]
  max_queue_delay_microseconds: 100
}
```

## Using Reshape in Triton Model Configuration

The `ModelTensorReshape` property in a Triton model configuration allows you to specify that the input or output shape accepted by the inference API differs from the shape expected or produced by the underlying model or backend.

### Input Reshape

For inputs, `reshape` can be used when the model expects a different shape than what the inference API provides. For example, if a model expects a batched input of shape `[batch-size]`, but the inference API requires `[batch-size, 1]`, you can use:

```
input [
  {
    name: "in"
    dims: [ 1 ]
    reshape: { shape: [ ] }
  }
  ...
]
```

### Output Reshape

Similarly, for outputs, if the model produces `[batch-size]` but the API expects `[batch-size, 1]`, use:

```
output [
  {
    name: "out"
    dims: [ 1 ]
    reshape: { shape: [ ] }
  }
  ...
]
```

For more details, see the [Triton Model Configuration documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/archives/triton_inference_server_1140/user-guide/docs/model_configuration.html?utm_source=chatgpt.com#reshape).

## Deployment

### Running the Triton Server

```bash
docker-compose up -d
```

### Checking Server Status

```bash
curl -v localhost:8000/v2/health/ready
```

### Making Inference Requests

Example using Python client:

```python
import tritonclient.http as httpclient
import numpy as np

client = httpclient.InferenceServerClient(url="localhost:8000")

# Prepare inputs
candidate_tokens = np.array([[101, 2054, 2003, 1996, 2265, 102]], dtype=np.int64)
history_tokens = np.array([[[101, 2054, 2003, 1996, 2265, 102]]], dtype=np.int64)

# Create input tensors
inputs = [
    httpclient.InferInput("candidate_tokens", candidate_tokens.shape, "INT64"),
    httpclient.InferInput("history_tokens", history_tokens.shape, "INT64")
]
inputs[0].set_data_from_numpy(candidate_tokens)
inputs[1].set_data_from_numpy(history_tokens)

# Define output
outputs = [httpclient.InferRequestedOutput("score")]

# Perform inference
response = client.infer("onnx", inputs=inputs, outputs=outputs)
score = response.as_numpy("score")
print(f"Ranking score: {score}")
```

## Monitoring and Performance Tuning

- Access metrics at: `localhost:8002/metrics`
- For performance optimization, consider adjusting:
  - `max_batch_size`
  - `preferred_batch_size`
  - `max_queue_delay_microseconds`
  - Number of instances in `instance_group`