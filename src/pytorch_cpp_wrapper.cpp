/*
 * A wrapper class of PyTorch C++ to use PyTorch model
 * Shigemichi Matsuzaki
 *
 */


#include "pytorch_cpp_wrapper/pytorch_cpp_wrapper.h"

//namespace mpl {

PyTorchCppWrapper::PyTorchCppWrapper() {

}

PyTorchCppWrapper::PyTorchCppWrapper(const std::string filename) {
  // Import
  import_module(filename);
}

PyTorchCppWrapper::PyTorchCppWrapper(const char* filename) {
  // Import
  import_module(std::string(filename));
}

void
PyTorchCppWrapper::import_module(const std::string filename)
{
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module_ = torch::jit::load(filename);
    std::cout << "Import succeeded" << std::endl;
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
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
  at::Tensor output = module_->forward({input_tensor}).toTensor();

  return output;
}

at::Tensor 
PyTorchCppWrapper::get_argmax(at::Tensor input_tensor)
{
  // Calculate argmax to get a label on each pixel
  at::Tensor output = at::argmax(input_tensor, 1).to(torch::kCPU).to(at::kByte);

  return output;
}
//} // namespace mpl
