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

class PyTorchCppWrapperSeg : public PyTorchCppWrapperBase {
private :
  // c = P(s|y=1) in PU learning, calculated during training
  float c_{0.3};
  bool use_aux_branch_{false};

public:
  PyTorchCppWrapperSeg(const std::string & filename, const int class_num);
  PyTorchCppWrapperSeg(const char* filename, const int class_num);

  /**
   * @brief Get outputs from the model
   * @param[in] input_tensor Input tensor
   * @return A tuple of output tensors (segmentation)
   */
  at::Tensor get_output(at::Tensor input_tensor);
};
#endif