/*
 * A wrapper class of PyTorch C++ to use PyTorch model
 * Shigemichi Matsuzaki
 *
 */


#include <torch/torch.h>
#include "pytorch_cpp_wrapper/pytorch_cpp_wrapper_seg_trav_path.h"
#include <torch/script.h> // One-stop header.
#include <torch/data/transforms/tensor.h> // One-stop header.
#include <c10/util/ArrayRef.h>
#include <opencv2/opencv.hpp>
#include "opencv2/highgui/highgui.hpp"
#include <typeinfo>

PyTorchCppWrapperSegTravPath::PyTorchCppWrapperSegTravPath(const std::string & filename, const int class_num)
  : PyTorchCppWrapperBase(filename, class_num)
{ }

PyTorchCppWrapperSegTravPath::PyTorchCppWrapperSegTravPath(const char* filename, const int class_num)
  : PyTorchCppWrapperBase(filename, class_num)
{ }

/**
 * @brief Get outputs from the model
 * @param[in] input_tensor Input tensor
 * @return A tuple of output tensors (segmentation, traversability, and path (points))
 */
std::tuple<at::Tensor, at::Tensor, at::Tensor>
PyTorchCppWrapperSegTravPath::get_output(at::Tensor input_tensor)
{
  // Execute the model and turn its output into a tensor.
  auto outputs_tmp = module_.forward({input_tensor}); //.toTuple();

  auto outputs = outputs_tmp.toTuple();

  at::Tensor output1 = outputs->elements()[0].toTensor();
  at::Tensor output2 = outputs->elements()[1].toTensor();
  at::Tensor prob = outputs->elements()[2].toTensor();
  at::Tensor path = outputs->elements()[3].toTensor();

  // Divide probability by c
  prob = torch::sigmoid(prob) / c_;
  // Limit the values in range [0, 1]
  prob = at::clamp(prob, 0.0, 1.0);

//  return output1 + 0.5 * output2;
  at::Tensor segmentation = output1 + 0.5 * output2;

  path = torch::sigmoid(path);

  return std::forward_as_tuple(segmentation, prob, path);
}
