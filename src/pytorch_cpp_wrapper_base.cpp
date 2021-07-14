/*
 * A wrapper class of PyTorch C++ to use PyTorch model
 * Shigemichi Matsuzaki
 *
 */


#include <torch/torch.h>
#include "pytorch_cpp_wrapper/pytorch_cpp_wrapper_base.h"
#include <torch/script.h> // One-stop header.
#include <torch/data/transforms/tensor.h> // One-stop header.
#include <c10/util/ArrayRef.h>
#include <opencv2/opencv.hpp>
#include "opencv2/highgui/highgui.hpp"
#include <typeinfo>

PyTorchCppWrapperBase::PyTorchCppWrapperBase() {}

PyTorchCppWrapperBase::PyTorchCppWrapperBase(const std::string & filename) {
  // Import model
  import_module(filename);
}

PyTorchCppWrapperBase::PyTorchCppWrapperBase(const char* filename) {
  // Import model
  import_module(std::string(filename));
}

/**
 * @brief import a network 
 * @param filename
 * @return true if import succeeded 
 */
bool
PyTorchCppWrapperBase::import_module(const std::string & filename)
{
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module_ = torch::jit::load(filename);
    // Set evaluation mode
    module_.eval();
    std::cout << module_.is_training() << std::endl;

    std::cout << "Import succeeded" << std::endl;
    return true;
  }
  catch (const c10::Error& e) {
    std::cerr << e.what();
    return false;
  }
}

/**
 * @brief convert an image(cv::Mat) to a tensor (at::Tensor)
 * @param[in] img
 * @param[out] tensor
 * @param[in] whether to use GPU
 */
void
PyTorchCppWrapperBase::img2tensor(cv::Mat & img, at::Tensor & tensor, const bool & use_gpu)
{
  // Get the size of the input image
  int height = img.size().height;
  int width  = img.size().width;

  // Create a vector of inputs.
  std::vector<int64_t>shape = {1, height, width, 3};
  if(use_gpu) {
    tensor = torch::from_blob(img.data, at::IntList(shape), at::ScalarType::Byte).to(torch::kFloat).to(torch::kCUDA);
  } else {
    tensor = torch::from_blob(img.data, at::IntList(shape), at::ScalarType::Byte).to(torch::kFloat).to(torch::kCPU);
  }
  tensor = at::transpose(tensor, 1, 2);
  tensor = at::transpose(tensor, 1, 3); 
}

/**
 * @brief convert a tensor (at::Tensor) to an image (cv::Mat)
 * @param[in] tensor
 * @param[out] img
 */
void
PyTorchCppWrapperBase::tensor2img(at::Tensor tensor, cv::Mat & img)
{
  // Get the size of the input image
  int height = tensor.sizes()[0];
  int width  = tensor.sizes()[1];

  tensor = tensor.to(torch::kCPU);

  // Convert to OpenCV
  img = cv::Mat(height, width, CV_8U, tensor. template data<uint8_t>());
}

/**
 * @brief Take element-wise argmax 
 * @param[in]  tensor
 * @param[out] tensor that has index of max value in each element
 */
at::Tensor 
PyTorchCppWrapperBase::get_argmax(at::Tensor input_tensor)
{
  // Calculate argmax to get a label on each pixel
  at::Tensor output = at::argmax(input_tensor, 1).to(torch::kCPU).to(at::kByte);

  return output;
}

  /**
   * @brief Take element-wise entropy 
   * @param[in]  tensor
   * @param[out] tensor that has index of max value in each element
   */
at::Tensor
PyTorchCppWrapperBase::get_entropy(at::Tensor input_tensor)
{
  input_tensor.to(torch::kCUDA);
  // Calculate the entropy at each pixel
  at::Tensor log_p = torch::log_softmax(input_tensor, /*dim=*/1);//at::argmax(input_tensor, 1).to(torch::kCPU).to(at::kByte);
  at::Tensor p = torch::softmax(input_tensor, /*dim=*/1);

  at::Tensor entropy = -torch::sum(p * log_p, /*dim=*/1);

  return entropy;
}
