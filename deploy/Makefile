# ─────────────────────────────────────────────────────────────
# deploy/Makefile      · build + install for ORT / TensorRT
# ─────────────────────────────────────────────────────────────

# ========== Versions ==========
ORT_VERSION ?= 1.21.1
TRT_MAJOR   ?= 8.5.1            # TensorRT SDK version
CUDA_TAG    ?= cuda11.7         # matches the TRT local-repo package

# ========== Paths ==========
# — ORT —
ORT_ROOT        ?= /usr/local/onnxruntime-$(ORT_VERSION)/onnxruntime-linux-x64-gpu-$(ORT_VERSION)
ORT_INCLUDE_DIR ?= $(ORT_ROOT)/include
ORT_LIB         ?= $(ORT_ROOT)/lib/libonnxruntime.so.$(ORT_VERSION)

# — TensorRT —
TRT_ROOT ?= /usr                  # override only if TRT headers/libs are elsewhere

# ========== Build directories ==========
BIN_DIR   := build
ORT_BUILD := $(BIN_DIR)/ort
TRT_BUILD := $(BIN_DIR)/trt

# ───────── Download URLs (computed) ─────────
ORT_TGZ := onnxruntime-linux-x64-gpu-$(ORT_VERSION).tgz
ORT_URL := https://github.com/microsoft/onnxruntime/releases/download/v$(ORT_VERSION)/$(ORT_TGZ)

TRT_DEB := nv-tensorrt-local-repo-ubuntu2004-$(TRT_MAJOR)-$(CUDA_TAG)_1.0-1_amd64.deb
TRT_URL := https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/$(TRT_DEB)

# ---------- Public phony targets ----------
.PHONY: all install-ort install-trt ort trt run-ort run-trt clean
all: ort trt

# ─────────────────────────────────────────────
# 1. ONNX Runtime · system-wide install
# ─────────────────────────────────────────────
install-ort:
	@echo "⇒ Installing ONNX Runtime $(ORT_VERSION)…"
	curl -fL $(ORT_URL) -o /tmp/$(ORT_TGZ)
	sudo mkdir -p $(ORT_ROOT)
	sudo tar -C $(ORT_ROOT) -xzf /tmp/$(ORT_TGZ)
	sudo ln -sf $(ORT_INCLUDE_DIR) /usr/local/include/onnxruntime
	sudo ln -sf $(ORT_ROOT)/lib/libonnxruntime.so* /usr/local/lib/
	sudo ldconfig
	rm /tmp/$(ORT_TGZ)
	@echo "✓ ONNX Runtime ready under $(ORT_ROOT)"

# ─────────────────────────────────────────────
# 2. TensorRT SDK · APT repo install
# ─────────────────────────────────────────────
install-trt:
	@echo "⇒ Adding NVIDIA ML repo key…"
	sudo apt-key adv --fetch-keys \
	  https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
	@echo "⇒ Adding repo to sources.list.d…"
	echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/ /" \
	  | sudo tee /etc/apt/sources.list.d/nvidia-ml.list > /dev/null
	sudo apt-get update -qq
	@echo "⇒ Installing TensorRT dev packages…"
	sudo apt-get install -y libnvinfer-dev libnvonnxparsers-dev python3-libnvinfer
	@echo "✓ TensorRT SDK installed."

# ─────────────────────────────────────────────
# 3. Build rules (both via CMake now)
# ─────────────────────────────────────────────
# — ORT —
ort: $(ORT_BUILD)/infer_ort

$(ORT_BUILD)/infer_ort:
	@echo "⇒ Configuring ORT backend…"
	cmake -S . -B $(ORT_BUILD) \
	      -DBACKEND_ORT=ON -DBACKEND_TRT=OFF \
	      -DORT_ROOT=$(ORT_ROOT) \
	      -DORT_INCLUDE_DIR=$(ORT_INCLUDE_DIR) \
	      -DORT_LIB=$(ORT_LIB)
	cmake --build $(ORT_BUILD) --parallel $(shell nproc)

run-ort: ort
	@echo "⇒ Running ORT inference…"
	./$(ORT_BUILD)/infer_ort

# — TensorRT —
trt: $(TRT_BUILD)/infer_trt

$(TRT_BUILD)/infer_trt:
	@echo "⇒ Configuring TensorRT backend…"
	cmake -S . -B $(TRT_BUILD) \
	      -DBACKEND_ORT=OFF -DBACKEND_TRT=ON \
	      -DTRT_ROOT=$(TRT_ROOT)\
		  -DTRT_INCLUDE_DIR=/usr/include/x86_64-linux-gnu \
		  -DNVINFER_LIB=/usr/lib/x86_64-linux-gnu/libnvinfer.so \
		  -DNVONNXPARSER_LIB=/usr/lib/x86_64-linux-gnu/libnvonnxparser.so
	cmake --build $(TRT_BUILD) --parallel $(shell nproc)

run-trt: trt
	@echo "⇒ Running TRT inference…"
	./$(TRT_BUILD)/infer_trt

# ─────────────────────────────────────────────
# 4. House-keeping
# ─────────────────────────────────────────────
$(BIN_DIR) $(ORT_BUILD) $(TRT_BUILD):
	mkdir -p $@

clean:
	@echo "Cleaning…"
	rm -rf $(BIN_DIR)