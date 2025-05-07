import argparse
import subprocess
import sys
import time
from concurrent import futures
from pathlib import Path

import grpc
from grpc_health.v1 import health_pb2, health_pb2_grpc
from grpc_health.v1.health import HealthServicer

class InferenceServicer:
    def __init__(self, exe_path, model_path):
        self.exe_path = exe_path
        self.model_path = model_path
        # Initialize backend subprocess here if needed

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["onnxrt", "tensorrt"], required=True)
    parser.add_argument("--model", default="models/onnx/model.onnx")
    parser.add_argument("--port", type=int, default=50051)
    args = parser.parse_args()

    exe = {
        "onnxrt": Path("deploy/build/infer_ort"),
        "tensorrt": Path("deploy/build/infer_trt")
    }[args.backend]

    # Create gRPC server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    
    # Add health service
    health_servicer = HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
    
    # Set all services to SERVING status
    health_servicer.set('', health_pb2.HealthCheckResponse.SERVING)
    
    # Start server
    server.add_insecure_port(f'[::]:{args.port}')
    server.start()
    
    print(f"Server started on port {args.port}")
    print(f"Using {args.backend} backend with model {args.model}")
    
    try:
        while True:
            time.sleep(86400)  # One day in seconds
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == "__main__":
    sys.exit(main())