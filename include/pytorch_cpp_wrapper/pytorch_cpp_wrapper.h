#ifndef PYTORCH_CPP_WRAPPER
#define PYTORCH_CPP_WRAPPER

#include <torch/script.h> // One-stop header.
#include <opencv2/opencv.hpp>
#include "opencv2/highgui/highgui.hpp"

#include <iostream>
#include <memory>

//namespace mpl {
class PyTorchCppWrapper {
private :
//  std::shared_ptr<torch::jit::script::Module> module_;
  torch::jit::script::Module module_;

public:
  PyTorchCppWrapper();
  PyTorchCppWrapper(const std::string filename);
  PyTorchCppWrapper(const char* filename);

  bool import_module(const std::string filename);
  void img2tensor(cv::Mat & img, at::Tensor & tensor, const bool use_gpu = true);
  void tensor2img(at::Tensor tensor, cv::Mat & img);
  at::Tensor get_output(at::Tensor input_tensor);
  at::Tensor get_argmax(at::Tensor input_tensor);
};
//}
#endif
