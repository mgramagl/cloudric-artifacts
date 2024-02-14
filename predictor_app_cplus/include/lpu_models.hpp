#ifndef LPU_MODELS_H_
#define LPU_MODELS_H_

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>


#define VERBOSE
// #define TIME_PROFILE

#define MAX_SNR 30.0
#define MAX_MCS 27
#define MAX_TBS 286976
#define DEADLINE 3000.0

#ifdef TIME_PROFILE
using clock_time = std::chrono::system_clock;
#endif


/**
 * @brief Compute the product over all the elements of a vector
 * @tparam T
 * @param v: input vector
 * @return the product
 */
template <typename T>
size_t vectorProduct(const std::vector<T>& v) {
  return std::accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
}

class LpuModels {
  public:
    /**
     * @brief Constructor
     * @param modelFilepath: path to the .onnx file
     */
    LpuModels(std::string file_path);

    /**
     * @brief Perform inference o
     * @return the index of the predicted class
     */
    std::pair<std::vector<float>,float> inference(const std::vector<float>& values);

  private:
    // ORT Environment
    std::shared_ptr<Ort::Env> mEnv;

    // Session
    std::shared_ptr<Ort::Session> mSession;

    // Inputs
    std::vector<std::string> mInputNames;
    std::vector<std::vector<int64_t>> mInputDims;
    std::vector<const char *> input_names_char;

    // Outputs
    std::vector<std::string> mOutputNames;
    std::vector<std::vector<int64_t>> mOutputDims;
    std::vector<const char *> output_names_char;

};


#endif  // LPU_MODELS_H_
