#ifndef PYTORCH_CPP_WRAPPER_BASE
#define PYTORCH_CPP_WRAPPER_BASE

#include <torch/script.h> // One-stop header.
#include <torch/data/transforms/tensor.h> // One-stop header.
#include <c10/util/ArrayRef.h>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <memory>

/**
 * @brief this class is a base class of C++ wrapper of PyTorch
 */
class PyTorchCppWrapperBase {
protected :
  torch::jit::script::Module module_;
  int class_num_;
  float max_entropy_;

public:
  PyTorchCppWrapperBase();
  PyTorchCppWrapperBase(const std::string & filename, const int class_num);
  PyTorchCppWrapperBase(const char* filename, const int class_num);

  /**
   * @brief import a network 
   * @param filename
   * @return true if import succeeded 
   */
  bool import_module(const std::string & filename);

  /**
   * @brief convert an image(cv::Mat) to a tensor (at::Tensor)
   * @param[in] img
   * @param[out] tensor
   * @param[in] whether to use GPU
   */
  void img2tensor(cv::Mat & img, at::Tensor & tensor, const bool & use_gpu = true);

  /**
   * @brief convert a tensor (at::Tensor) to an image (cv::Mat)
   * @param[in] tensor
   * @param[out] img
   */
  void tensor2img(at::Tensor tensor, cv::Mat & img);

  /**
   * @brief convert a tensor (at::Tensor) to an image (cv::Mat)
   * @param[in] tensor
   * @return converted CV image
   */
  cv::Mat tensor2img(at::Tensor tensor);

  /**
   * @brief Take element-wise argmax 
   * @param[in]  tensor
   * @param[out] tensor that has index of max value in each element
   */
  at::Tensor get_argmax(at::Tensor input_tensor);

  /**
   * @brief Take element-wise entropy 
   * @param[in]  tensor
   * @param[out] tensor that has index of max value in each element
   */
  at::Tensor get_entropy(at::Tensor input_tensor, const bool normalize);

};
//}
#endif
