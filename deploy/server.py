import argparse
import subprocess
import sys
from pathlib import Path

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["onnxrt", "tensorrt"], required=True)
    parser.add_argument("--model", default="models/onnx/model.onnx")
    args = parser.parse_args()

    exe = {
        "onnxrt":  Path("deploy/build/infer_ort"),
        "tensorrt": Path("deploy/build/infer_trt")
    }[args.backend]

    subprocess.run([str(exe), "--model", args.model], check=True)

if __name__ == "__main__":
    sys.exit(main())