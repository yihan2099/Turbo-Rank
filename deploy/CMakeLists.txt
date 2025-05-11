cmake_minimum_required(VERSION 3.18)
project(turbo_infer LANGUAGES CXX)

# ─────────── User‐visible options ───────────
option(BACKEND_TRT "Build TensorRT executable" ON)
option(BACKEND_ORT "Build ONNX-Runtime executable" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ─────────── Find required packages ───────────
find_package(Threads REQUIRED)
find_package(CUDAToolkit REQUIRED)    # provides CUDA::cudart

# ─────────── Helper macro ───────────
function(add_backend_exe name src incs libs)
    add_executable(${name} ${src})
    target_include_directories(${name} PRIVATE ${incs})
    target_link_libraries(${name} PRIVATE ${libs})
    set_target_properties(${name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    )
endfunction()

# ─────────── TensorRT build ───────────
if(BACKEND_TRT)
    # Allow overriding TRT_ROOT via -DTRT_ROOT=…
    set(TRT_ROOT   "/usr/local/TensorRT"  CACHE PATH "TensorRT root directory")
    find_path(TRT_INCLUDE_DIR NvInfer.h
        HINTS "${TRT_ROOT}/include" /usr/include /usr/include/x86_64-linux-gnu
    )
    find_library(NVINFER_LIB       nvinfer
        HINTS "${TRT_ROOT}/lib" /usr/lib/x86_64-linux-gnu
    )
    find_library(NVONNXPARSER_LIB  nvonnxparser
        HINTS "${TRT_ROOT}/lib" /usr/lib/x86_64-linux-gnu
    )

    add_backend_exe(infer_trt
        infer_trt.cpp
        "${TRT_INCLUDE_DIR}"
        "${NVINFER_LIB};${NVONNXPARSER_LIB};CUDA::cudart;Threads::Threads"
    )
endif()

# ─────────── ONNX-Runtime build ───────────
if(BACKEND_ORT)
    # Allow overriding ORT_ROOT via -DORT_ROOT=…
    set(ORT_ROOT   "/usr/local/onnxruntime"  CACHE PATH "ONNX-Runtime root directory")
    find_path(ORT_INCLUDE_DIR onnxruntime_cxx_api.h
        HINTS "${ORT_ROOT}/include" /usr/include /usr/include/x86_64-linux-gnu
    )
    find_library(ORT_LIB       onnxruntime
        HINTS "${ORT_ROOT}/lib" /usr/lib/x86_64-linux-gnu
    )

    add_backend_exe(infer_ort
        infer_ort.cpp
        "${ORT_INCLUDE_DIR}"
        "${ORT_LIB};CUDA::cudart;Threads::Threads"
    )
endif()