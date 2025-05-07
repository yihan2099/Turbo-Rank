cmake_minimum_required(VERSION 3.18)
project(turbo_infer LANGUAGES CXX)

# ─────────── User-visible options ───────────
option(BACKEND_TRT "Build TensorRT executable" ON)
option(BACKEND_ORT "Build ONNX-Runtime executable" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Helper macro ---------------------------------------------------------------
function(add_backend_exe name src incs libs)
    add_executable(${name} ${src})
    target_include_directories(${name} PRIVATE ${incs})
    target_link_libraries(${name} PRIVATE ${libs})
    set_target_properties(${name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    )
endfunction()

# ─────────── TensorRT back-end ───────────
if(BACKEND_TRT)
    # 1. User-overridable cache variables
    set(TRT_ROOT ""                CACHE PATH     "Root of TensorRT SDK")
    set(TRT_INCLUDE_DIR ""         CACHE PATH     "Explicit path to TRT headers")
    set(NVINFER_LIB ""             CACHE FILEPATH "Path to libnvinfer.so")
    set(NVONNXPARSER_LIB ""        CACHE FILEPATH "Path to libnvonnxparser.so")

    # 2. Resolve include directory
    if(NOT TRT_INCLUDE_DIR)
        list(APPEND _trt_inc_hints
             ${TRT_ROOT}/include
             /usr/include/x86_64-linux-gnu
             /usr/local/include)
        find_path(TRT_INCLUDE_DIR NvInfer.h HINTS ${_trt_inc_hints} NO_DEFAULT_PATH)
    endif()

    # 3. Resolve libraries
    if(NOT (NVINFER_LIB AND NVONNXPARSER_LIB))
        list(APPEND _trt_lib_hints
             ${TRT_ROOT}/lib
             /usr/lib/x86_64-linux-gnu
             /usr/local/lib)
        if(NOT NVINFER_LIB)
            find_library(NVINFER_LIB nvinfer HINTS ${_trt_lib_hints} NO_DEFAULT_PATH)
        endif()
        if(NOT NVONNXPARSER_LIB)
            find_library(NVONNXPARSER_LIB nvonnxparser HINTS ${_trt_lib_hints} NO_DEFAULT_PATH)
        endif()
    endif()

    # 4. Validate discovery
    if(NOT TRT_INCLUDE_DIR OR NOT NVINFER_LIB OR NOT NVONNXPARSER_LIB)
        message(FATAL_ERROR
            "TensorRT headers or libraries not found. "
            "Set TRT_ROOT or the explicit TRT_INCLUDE_DIR / NVINFER_LIB / NVONNXPARSER_LIB variables.")
    endif()

    # 5. CUDA runtime (provides cudaMalloc / cudaMemcpy / streams)
    find_package(CUDAToolkit REQUIRED)          # exports CUDA::cudart

    # 6. Promote libs to imported targets
    add_library(tensorrt::nvinfer SHARED IMPORTED)
    set_target_properties(tensorrt::nvinfer PROPERTIES
        IMPORTED_LOCATION             "${NVINFER_LIB}"
        INTERFACE_INCLUDE_DIRECTORIES "${TRT_INCLUDE_DIR}")

    add_library(tensorrt::nvonnxparser SHARED IMPORTED)
    set_target_properties(tensorrt::nvonnxparser PROPERTIES
        IMPORTED_LOCATION             "${NVONNXPARSER_LIB}"
        INTERFACE_INCLUDE_DIRECTORIES "${TRT_INCLUDE_DIR}")

    # 7. Build the demo executable
    add_backend_exe(
        infer_trt
        infer_trt.cpp
        "${TRT_INCLUDE_DIR}"
        tensorrt::nvinfer tensorrt::nvonnxparser CUDA::cudart
    )
endif()

# ─────────── ONNX Runtime back-end ───────────
if(BACKEND_ORT)
    set(ORT_ROOT ""            CACHE PATH "Root of ONNX Runtime installation")
    set(ORT_INCLUDE_DIR ""     CACHE PATH "Explicit path to ORT headers")
    set(ORT_LIB ""             CACHE FILEPATH "Explicit path to libonnxruntime.so")

    if(NOT ORT_INCLUDE_DIR)
        list(APPEND _inc_hints
             ${ORT_ROOT}/include
             /usr/local/include/onnxruntime
             /usr/include/onnxruntime)
        find_path(ORT_INCLUDE_DIR onnxruntime_cxx_api.h HINTS ${_inc_hints} NO_DEFAULT_PATH)
    endif()

    if(NOT ORT_LIB)
        list(APPEND _lib_hints
             ${ORT_ROOT}/lib
             /usr/local/lib
             /usr/lib)
        find_library(ORT_LIB onnxruntime HINTS ${_lib_hints} NO_DEFAULT_PATH)
    endif()

    if(NOT ORT_INCLUDE_DIR OR NOT ORT_LIB)
        message(FATAL_ERROR
            "ONNX Runtime headers or library not found. "
            "Set ORT_ROOT or ORT_INCLUDE_DIR / ORT_LIB.")
    endif()

    add_library(onnxruntime::onnxruntime SHARED IMPORTED)
    set_target_properties(onnxruntime::onnxruntime PROPERTIES
        IMPORTED_LOCATION             "${ORT_LIB}"
        INTERFACE_INCLUDE_DIRECTORIES "${ORT_INCLUDE_DIR}")

    add_backend_exe(
        infer_ort
        infer_ort.cpp
        "${ORT_INCLUDE_DIR}"
        onnxruntime::onnxruntime)
endif()