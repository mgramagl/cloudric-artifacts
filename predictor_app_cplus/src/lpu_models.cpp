#include "lpu_models.hpp"

#include <algorithm>
#include <sstream>
#include <iterator>
#ifdef OUTDATA

// pretty prints a shape dimension vector
std::string print_shape(const std::vector<std::int64_t> &v)
{
    std::stringstream ss("");
    for (std::size_t i = 0; i < v.size() - 1; i++)
        ss << v[i] << "x";
    ss << v[v.size() - 1];
    return ss.str();
}

/**
 * @brief Print ONNX tensor type
 * https://github.com/microsoft/onnxruntime/blob/rel-1.6.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L93
 * @param os
 * @param type
 * @return std::ostream&
 */
std::ostream &operator<<(std::ostream &os,
                         const ONNXTensorElementDataType &type)
{
    switch (type)
    {
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED:
        os << "undefined";
        break;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        os << "float";
        break;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
        os << "uint8_t";
        break;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
        os << "int8_t";
        break;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
        os << "uint16_t";
        break;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
        os << "int16_t";
        break;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
        os << "int32_t";
        break;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
        os << "int64_t";
        break;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
        os << "std::string";
        break;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
        os << "bool";
        break;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_float16:
        os << "float16";
        break;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_float:
        os << "float";
        break;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
        os << "uint32_t";
        break;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
        os << "uint64_t";
        break;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
        os << "float real + float imaginary";
        break;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
        os << "float real + float imaginary";
        break;
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_Bfloat16:
        os << "bfloat16";
        break;
    default:
        break;
    }

    return os;
}
#endif

LpuModels::LpuModels(std::string file_path)
{
    std::string instanceName{"Multiplier inference"};
    mEnv = std::make_shared<Ort::Env>(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                                      instanceName.c_str());

    /**************** Create ORT session ******************/
    // Set up options for session
    Ort::SessionOptions sessionOptions;
    // Enable CUDA
    // sessionOptions.AppendExecutionProvider_CUDA(OrtCUDAProviderOptions{});
    // Sets graph optimization level (Here, enable all possible optimizations)
    sessionOptions.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL);
    // Create session by loading the onnx model
    mSession = std::make_shared<Ort::Session>(*mEnv, file_path.c_str(),
                                              sessionOptions);

    /**************** Create allocator ******************/
    // Allocator is used to get model information
    Ort::AllocatorWithDefaultOptions allocator;

    /**************** Input info ******************/
    // Get the number of input nodes
    size_t numInputNodes = mSession->GetInputCount();
#ifdef OUTDATA
    std::cout << "******* Model information below *******" << std::endl;
    std::cout << "Number of Input Nodes: " << numInputNodes << std::endl;
#endif

    // Get the name and shape of the input

    for (std::size_t i = 0; i < mSession->GetInputCount(); i++)
    {
        mInputNames.emplace_back(mSession->GetInputNameAllocated(i, allocator).get());
        mInputDims.emplace_back(mSession->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
        ONNXTensorElementDataType inputType = mSession->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetElementType();
#ifdef OUTDATA
        std::cout << "Input Name: " << mInputNames.at(i) << std::endl;
        std::cout << "Input Dimensions: " << print_shape(mInputDims.at(i)) << std::endl;
        std::cout << "Input Type: " << inputType << std::endl;
#endif
    }
    for (int j = 0; j <  mInputNames.size(); ++j)
        this->input_names_char.push_back(nullptr);
    std::transform(std::begin(mInputNames), std::end(mInputNames), std::begin(input_names_char),
                   [&](const std::string &str)
                   { return str.c_str(); });

    /**************** Output info ******************/
    // Get the number of output nodes
    size_t numOutputNodes = mSession->GetOutputCount();
#ifdef OUTDATA
    std::cout << "Number of Output Nodes: " << numOutputNodes << std::endl;
#endif

    // Get the name and shape of the input

    for (std::size_t i = 0; i < mSession->GetOutputCount(); i++)
    {
        mOutputNames.emplace_back(mSession->GetOutputNameAllocated(i, allocator).get());
        mOutputDims.emplace_back(mSession->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
        ONNXTensorElementDataType outputType = mSession->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetElementType();

#ifdef OUTDATA
        std::cout << "Output Name: " << mOutputNames.at(i) << std::endl;
        std::cout << "Output Dimensions: " << print_shape(mOutputDims.at(i)) << std::endl;
        std::cout << "Output Type: " << outputType << std::endl;

#endif
    }
    for (int j = 0; j <  mOutputNames.size(); ++j)
        this->output_names_char.push_back(nullptr);    

    std::transform(std::begin(mOutputNames), std::end(mOutputNames), std::begin(output_names_char),
                   [&](const std::string &str)
                   { return str.c_str(); });
}

std::vector<float> LpuModels::inference(const std::vector<float> &values)
{

#ifdef TIME_PROFILE
    const auto before = clock_time::now();
#endif
    size_t inputTensorSize = vectorProduct(mInputDims.at(0));
    std::vector<float> inputTensorValues(values);
    std::vector<Ort::Value> inputTensors;
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    std::vector<std::int64_t> inputDim = mInputDims.at(0);
    inputTensors.emplace_back(Ort::Value::CreateTensor<float>(
        memoryInfo, inputTensorValues.data(), inputTensorValues.size(),
       inputDim.data(), inputDim.size()));

#ifdef OUTDATA
    std::cout<<"Input Tensor Data"<<std::endl;
    std::cout<<print_shape(inputTensors[0].GetTensorTypeAndShapeInfo().GetShape())<<std::endl;
    std::copy(inputTensorValues.begin(), inputTensorValues.end(), std::ostream_iterator<float>(std::cout, " "));
    std::cout<<std::endl;
#endif

    // Create output tensor (including size and value)
    size_t outputTensorSize = vectorProduct(mOutputDims.at(0));
    std::vector<float> outputTensorValues(outputTensorSize);

    // Assign memory for output tensors
    // outputTensors will be used by the Session Run for inference
    std::vector<std::int64_t> outputDim = mOutputDims.at(0);

    std::vector<Ort::Value> outputTensors;
    outputTensors.emplace_back(Ort::Value::CreateTensor<float>(
        memoryInfo, outputTensorValues.data(), outputTensorValues.size(),
       outputDim.data(), outputDim.size()));

#ifdef TIME_PROFILE
    const auto before1 = clock_time::now();
#endif

    mSession->Run(Ort::RunOptions{nullptr}, this->input_names_char.data(),
                  inputTensors.data(), 1, this->output_names_char.data(),
                  outputTensors.data(), 1);

#ifdef TIME_PROFILE
    auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(clock_time::now() - before1);
    // std::cout << "The inference takes " << duration1.count() << "us" << std::endl;
    // std::cout << duration1.count() << std::endl;
#endif

    return outputTensorValues;
}