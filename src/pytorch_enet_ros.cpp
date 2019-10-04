/*
 * A ROS node to do inference using PyTorch model
 * Shigemichi Matsuzaki
 *
 */

#include <pytorch_enet_ros/pytorch_enet_ros.h>

PyTorchENetROS::PyTorchENetROS() 
  : it_(nh_)
{
  sub_image_ = it_.subscribe("image", 1, &PyTorchENetROS::image_callback, this);
  pub_label_image_ = it_.advertise("label", 1);
  pub_color_image_ = it_.advertise("color_label", 1);

  std::string filename;
  nh_.param<std::string>("model_file", filename, "");
  pt_wrapper_.import_module(filename);
}

void
PyTorchENetROS::image_callback(const sensor_msgs::ImageConstPtr& msg)
{
  cv_bridge::CvImagePtr cv_ptr;
  // Convert the image message to a cv_bridge object
  try
  {
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }

  // Run inference
  inference(cv_ptr->image);
}

void
PyTorchENetROS::inference(cv::Mat & input_img)
{
  at::Tensor input_tensor;
  pt_wrapper_.img2tensor(input_img, input_tensor);

  // Execute the model and turn its output into a tensor.
//  at::Tensor output = module->forward({input_tensor}).toTensor();
  at::Tensor output = pt_wrapper_.get_output(input_tensor);
  // Calculate argmax to get a label on each pixel
  // at::Tensor output_args = at::argmax(output, 1).to(torch::kCPU).to(at::kByte);
  at::Tensor output_args = pt_wrapper_.get_argmax(output);

  // Convert to OpenCV
  // cv::Mat mat(height, width, CV_8U, output_args[0]. template data<uint8_t>());
  cv::Mat mat;
  pt_wrapper_.tensor2img(output_args[0], mat);

  std::cout << output_args[0].sizes() << std::endl;
  std::cout << mat.size() << std::endl;

  // Generate an image message
  sensor_msgs::ImagePtr label_msg = cv_bridge::CvImage(std_msgs::Header(), "mono8", mat).toImageMsg();
//  sensor_msgs::ImagePtr color_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", image).toImageMsg();
  
  pub_label_image_.publish(label_msg);
}
