#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>

#include <cassert>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

using namespace nvinfer1;

// ─────────────────── Logger ───────────────────
class Logger final : public ILogger {
    void log(Severity s, const char* msg) noexcept override {
        if (s <= Severity::kWARNING) std::cout << "[TRT] " << msg << '\n';
    }
} gLogger;

// Unique-ptr wrapper (TensorRT 10 objects have normal destructors)
template <typename T>
using TRTUnique = std::unique_ptr<T>;

template <typename T>
inline TRTUnique<T> make_trt_unique(T* t) { return TRTUnique<T>(t); }

inline void checkCuda(cudaError_t e) { assert(e == cudaSuccess); }

int main() {
    constexpr int64_t B = 2, H = 4, L = 50;               // shapes
    const char* onnxPath = "../models/onnx/nrms.onnx";

    // ───── 1. Parse ONNX & build engine ─────
    auto builder  = make_trt_unique(createInferBuilder(gLogger));
    auto network  = make_trt_unique(builder->createNetworkV2(/*flags=*/0));
    auto parser   = make_trt_unique(nvonnxparser::createParser(*network, gLogger));

    std::ifstream ifs(onnxPath, std::ios::binary);
    if (!ifs) { std::cerr << "Cannot open " << onnxPath << '\n'; return 1; }
    const std::string onnxData((std::istreambuf_iterator<char>(ifs)),
                               std::istreambuf_iterator<char>());
    if (!parser->parse(onnxData.data(), onnxData.size())) {
        std::cerr << "ONNX parse failed\n"; return 1;
    }

    // Optimisation profile (mandatory for explicit dynamic dims)
    auto profile = builder->createOptimizationProfile();
    auto candIn  = network->getInput(0);   // "candidate_tokens"
    auto histIn  = network->getInput(1);   // "history_tokens"

    profile->setDimensions(candIn->getName(), OptProfileSelector::kMIN,
                           Dims2{B, L});
    profile->setDimensions(candIn->getName(), OptProfileSelector::kOPT,
                           Dims2{B, L});
    profile->setDimensions(candIn->getName(), OptProfileSelector::kMAX,
                           Dims2{B, L});

    profile->setDimensions(histIn->getName(), OptProfileSelector::kMIN,
                           Dims3{B, H, L});
    profile->setDimensions(histIn->getName(), OptProfileSelector::kOPT,
                           Dims3{B, H, L});
    profile->setDimensions(histIn->getName(), OptProfileSelector::kMAX,
                           Dims3{B, H, L});

    auto config = make_trt_unique(builder->createBuilderConfig());
    config->addOptimizationProfile(profile);
    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1ULL << 30); // 1 GB
    if (builder->platformHasFastFp16()) config->setFlag(BuilderFlag::kFP16);

    auto engine = make_trt_unique(builder->buildEngineWithConfig(*network, *config));
    if (!engine) { std::cerr << "Engine build failed\n"; return 1; }

    // ───── 2. Allocate buffers ─────
    std::vector<int64_t> candHost(B * L, 1);
    std::vector<int64_t> histHost(B * H * L, 1);
    std::vector<float>   scoreHost(B, 0.f);

    void* dCand{}; void* dHist{}; void* dScore{};
    checkCuda(cudaMalloc(&dCand,  candHost.size()  * sizeof(int64_t)));
    checkCuda(cudaMalloc(&dHist,  histHost.size()  * sizeof(int64_t)));
    checkCuda(cudaMalloc(&dScore, scoreHost.size() * sizeof(float)));

    checkCuda(cudaMemcpy(dCand, candHost.data(),
                         candHost.size()*sizeof(int64_t), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(dHist, histHost.data(),
                         histHost.size()*sizeof(int64_t), cudaMemcpyHostToDevice));

    // ───── 3. Execute ─────
    auto context = make_trt_unique(engine->createExecutionContext());
    if (!context) { std::cerr << "Context creation failed\n"; return 1; }

    // Register device buffers by tensor name (name-based execution)
    context->setInputTensorAddress (candIn->getName(), dCand);
    context->setInputTensorAddress (histIn->getName(), dHist);
    const char* scoreName = engine->getIOTensorName(engine->getNbIOTensors()-1);
    context->setOutputTensorAddress(scoreName, dScore);

    // Provide shapes once
    context->setInputShape(candIn->getName(), Dims2{B, L});
    context->setInputShape(histIn->getName(), Dims3{B, H, L});

    cudaStream_t stream; checkCuda(cudaStreamCreate(&stream));
    if (!context->enqueueV3(stream)) {
        std::cerr << "enqueueV3 failed\n"; return 1;
    }

    checkCuda(cudaMemcpyAsync(scoreHost.data(), dScore,
                              scoreHost.size()*sizeof(float),
                              cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    std::cout << "Scores:";
    for (float s : scoreHost) std::cout << ' ' << s;
    std::cout << '\n';

    cudaFree(dCand); cudaFree(dHist); cudaFree(dScore);
    cudaStreamDestroy(stream);
    return 0;
}