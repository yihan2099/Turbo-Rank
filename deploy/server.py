import argparse
from concurrent import futures
import logging
from pathlib import Path
import sys
import time

import grpc

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["onnxrt", "tensorrt"], required=True)
    parser.add_argument("--model", default="models/onnx/nrms.onnx")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--max_workers", type=int, default=10)
    args = parser.parse_args()

    # Get the executable path for the selected backend
    exe = {
        "onnxrt": Path("deploy/build/infer_ort"),
        "tensorrt": Path("deploy/build/infer_trt")
    }[args.backend]
    
    # Create gRPC server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=args.max_workers))
    
    # Start server
    server.add_insecure_port(f'[::]:{args.port}')
    server.start()
    
    logger.info(f"Server started on port {args.port}")
    logger.info(f"Using {args.backend} backend with model {args.model}")
    
    try:
        while True:
            time.sleep(86400)  # One day in seconds
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == "__main__":
    sys.exit(main())