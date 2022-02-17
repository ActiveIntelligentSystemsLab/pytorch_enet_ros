/*
 * A wrapper class of PyTorch C++ to use PyTorch model
 * Shigemichi Matsuzaki
 *
 */

#include <torch/torch.h>
#include "pytorch_cpp_wrapper/pytorch_cpp_wrapper_seg.h"
#include <torch/script.h> // One-stop header.
#include <torch/data/transforms/tensor.h> // One-stop header.
#include <c10/util/ArrayRef.h>
#include <opencv2/opencv.hpp>
#include "opencv2/highgui/highgui.hpp"
#include <typeinfo>

PyTorchCppWrapperSeg::PyTorchCppWrapperSeg(const std::string & filename, const int class_num)
  : PyTorchCppWrapperBase(filename, class_num)
{ }

PyTorchCppWrapperSeg::PyTorchCppWrapperSeg(const char* filename, const int class_num)
  : PyTorchCppWrapperBase(filename, class_num)
{ }

  /**
   * @brief Get outputs from the model
   * @param[in] input_tensor Input tensor
   * @return A tuple of output tensors (segmentation)
   */
at::Tensor
PyTorchCppWrapperSeg::get_output(at::Tensor input_tensor)
{
  // Execute the model and turn its output into a tensor.
  auto outputs_tmp = module_.forward({input_tensor}); //.toTuple();

  at::Tensor segmentation;
  // If the network has two branches
  if(use_aux_branch_) {
    auto outputs = outputs_tmp.toTuple();
  
    at::Tensor output1 = outputs->elements()[0].toTensor();
    at::Tensor output2 = outputs->elements()[1].toTensor();
  
    segmentation = output1 + 0.5 * output2;
  } else {
    // If there's only one segmentation branch, directly use the output
    segmentation = outputs_tmp.toTensor();
  }

  return segmentation;
}


//} // namespace mpl
