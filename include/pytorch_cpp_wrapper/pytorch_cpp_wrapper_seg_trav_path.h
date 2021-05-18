#ifndef PYTORCH_CPP_WRAPPER
#define PYTORCH_CPP_WRAPPER

#include <torch/script.h> // One-stop header.
#include <torch/data/transforms/tensor.h> // One-stop header.
#include <c10/util/ArrayRef.h>
#include <opencv2/opencv.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "pytorch_cpp_wrapper/pytorch_cpp_wrapper_base.h"

#include <iostream>
#include <memory>


class PyTorchCppWrapperSegTravPath : public PyTorchCppWrapperBase {
private :
  // c = P(s|y=1) in PU learning, calculated during training
  float c_{0.3};

public:
  std::tuple<at::Tensor, at::Tensor, at::Tensor> get_output(at::Tensor input_tensor);
};
#endif
