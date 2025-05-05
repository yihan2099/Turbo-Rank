#include <onnxruntime_cxx_api.h>
#include <vector>
#include <iostream>

int main() {
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "nrms");
  Ort::SessionOptions opts;
  opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  Ort::Session sess(env, "../models/onnx/nrms.onnx", opts);

  // shapes
  const int64_t B = 2, H = 4, L = 50;
  std::vector<int64_t> cand_shape{B, L}, hist_shape{B, H, L};
  std::vector<int64_t> cand_data(B * L, 1), hist_data(B * H * L, 1);

  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value cand_tensor = Ort::Value::CreateTensor<int64_t>(
      memory_info, cand_data.data(), cand_data.size(), cand_shape.data(), cand_shape.size());
  Ort::Value hist_tensor = Ort::Value::CreateTensor<int64_t>(
      memory_info, hist_data.data(), hist_data.size(), hist_shape.data(), hist_shape.size());

  const char* input_names[]  = {"candidate_tokens", "history_tokens"};
  const char* output_names[] = {"score"};

  // *** Pass an array of Ort::Value (not Ort::Value*) ***
  Ort::Value inputs[] = { std::move(cand_tensor), std::move(hist_tensor) };

  // This overload takes (RunOptions, const char* const*, const Value*, size_t, const char* const*, size_t)
  auto outputs = sess.Run(
      Ort::RunOptions{nullptr},
      input_names,
      inputs,        // array of Ort::Value
      2,             // number of inputs
      output_names,
      1              // number of outputs
  );

  // Now outputs is std::vector<Ort::Value>
  // You can call the template method directly:
  float* scores = outputs.front().GetTensorMutableData<float>();

  std::cout << "Scores:";
  for (int i = 0; i < B; ++i) {
    std::cout << " " << scores[i];
  }
  std::cout << "\n";

  return 0;
}