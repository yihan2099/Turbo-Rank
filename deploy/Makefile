# deploy/Makefile

ORT_VERSION := 1.21.1
ORT_TARBALL := onnxruntime-linux-x64-gpu-$(ORT_VERSION).tgz
ORT_DIR     := onnxruntime-linux-x64-gpu-$(ORT_VERSION)
ORT_URL     := https://github.com/microsoft/onnxruntime/releases/download/v$(ORT_VERSION)/$(ORT_TARBALL)

CXX         := g++
BUILD_DIR   := build
dEPLOY_DIR  := .
SRC         := infer.cpp
TARGET      := infer

CXXFLAGS    := -O2 -std=c++17 -I$(dEPLOY_DIR)/include
LDFLAGS     := -L$(dEPLOY_DIR)/lib -lonnxruntime

.PHONY: all download extract setup build run clean

all: setup build

# Download the ONNX Runtime dev bundle if not present
download:
	@echo "Downloading ONNX Runtime dev bundle..."
	test -f $(ORT_TARBALL) || wget $(ORT_URL)

# Extract headers & libs
extract: download
	@echo "Extracting headers & libs..."
	test -d $(ORT_DIR) || tar -xzf $(ORT_TARBALL)

setup: extract
	@echo "Populating include/ and lib/..."
	mkdir -p include lib
	cp -r $(ORT_DIR)/include/* include/
	cp -r $(ORT_DIR)/lib/*     lib/
	@echo "Cleaning up tarball and extracted files..."
	rm -rf $(ORT_TARBALL) $(ORT_DIR)

build: setup
	@echo "Building inference binary in $(BUILD_DIR)/$(TARGET)..."
	mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(SRC) $(LDFLAGS) -o $(BUILD_DIR)/$(TARGET)

run: build
	@echo "Running inference..."
	LD_LIBRARY_PATH=$(PWD)/lib:$$LD_LIBRARY_PATH $(BUILD_DIR)/$(TARGET)

clean:
	@echo "Removing build artifacts..."
	rm -rf $(BUILD_DIR) include lib $(TARGET)
