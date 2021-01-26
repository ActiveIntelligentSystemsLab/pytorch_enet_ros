/*
 * A wrapper class of PyTorch C++ to use PyTorch model
 * Shigemichi Matsuzaki
 *
 */


#include "pytorch_cpp_wrapper/pytorch_cpp_wrapper.h"
#include <torch/script.h> // One-stop header.
#include <torch/data/transforms/tensor.h> // One-stop header.
#include <c10/util/ArrayRef.h>
#include <opencv2/opencv.hpp>
#include "opencv2/highgui/highgui.hpp"
#include <typeinfo>

//namespace mpl {

PyTorchCppWrapper::PyTorchCppWrapper() {
  std::vector<float> mean_vec{0.485, 0.456, 0.406};
  std::vector<float> std_vec{0.229, 0.224, 0.225};
  
}

PyTorchCppWrapper::PyTorchCppWrapper(const std::string filename) {
  // Import
  import_module(filename);
  std::vector<float> mean_vec{0.485, 0.456, 0.406};
  std::vector<float> std_vec{0.229, 0.224, 0.225};
  
}

PyTorchCppWrapper::PyTorchCppWrapper(const char* filename) {
  // Import
  import_module(std::string(filename));

  std::vector<float> mean_vec{0.485, 0.456, 0.406};
  std::vector<float> std_vec{0.229, 0.224, 0.225};
  
}

bool
PyTorchCppWrapper::import_module(const std::string filename)
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
    std::cerr << "error loading the model\n";
    return false;
  }
}

void
PyTorchCppWrapper::img2tensor(cv::Mat & img, at::Tensor & tensor, const bool use_gpu)
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

void
PyTorchCppWrapper::tensor2img(at::Tensor tensor, cv::Mat & img)
{
  std::cout << tensor.sizes() << std::endl;
  // Get the size of the input image
  int height = tensor.sizes()[0];
  int width  = tensor.sizes()[1];

  // Convert to OpenCV
  img = cv::Mat(height, width, CV_8U, tensor. template data<uint8_t>());
}

at::Tensor
PyTorchCppWrapper::get_output(at::Tensor input_tensor)
{
  // Execute the model and turn its output into a tensor.
  auto outputs_tmp = module_.forward({input_tensor}); //.toTuple();

  auto outputs = outputs_tmp.toTuple();

  at::Tensor output1 = outputs->elements()[0].toTensor();
  at::Tensor output2 = outputs->elements()[1].toTensor();

  return output1 + 0.5 * output2;
}

at::Tensor 
PyTorchCppWrapper::get_argmax(at::Tensor input_tensor)
{
  // Calculate argmax to get a label on each pixel
  at::Tensor output = at::argmax(input_tensor, 1).to(torch::kCPU).to(at::kByte);

  return output;
}
//} // namespace mpl
