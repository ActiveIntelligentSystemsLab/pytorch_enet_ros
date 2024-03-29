/*
 * A ROS node to do inference using PyTorch model
 * Shigemichi Matsuzaki
 *
 */

#include <pytorch_ros/pytorch_seg_ros.h>

PyTorchSegROS::PyTorchSegROS(ros::NodeHandle & nh) 
  : it_(nh), nh_(nh)
{
  sub_image_ = it_.subscribe("image", 1, &PyTorchSegROS::image_callback, this);
  pub_label_image_ = it_.advertise("label", 10);
  pub_color_image_ = it_.advertise("color_label", 10);
  pub_uncertainty_image_ = it_.advertise("uncertainty", 10);
  get_label_image_server_ = nh_.advertiseService("get_label_image", &PyTorchSegROS::image_inference_srv_callback, this);

  // Import the model
  std::string filename;
  nh_.param<std::string>("model_file", filename, "");
  pt_wrapper_ptr_.reset(new PyTorchCppWrapperSeg(filename, 4));
  if(!pt_wrapper_ptr_->import_module(filename)) {
    ROS_ERROR("Failed to import the model file [%s]", filename.c_str());
    ros::shutdown();
  }

  // Import color map image
  std::string colormap_name;
  nh_.param<std::string>("colormap", colormap_name, "");
  colormap_ = cv::imread(colormap_name);
  if(colormap_.empty()) {
    ROS_ERROR("Failed to import the colormap file [%s]", colormap_name.c_str());
    ros::shutdown();
  }
  
}

void
PyTorchSegROS::image_callback(const sensor_msgs::ImageConstPtr& msg)
{
  ROS_INFO("[PyTorchSegROS image_callback] Let's start!!");

  // Convert the image message to a cv_bridge object
  cv_bridge::CvImagePtr cv_ptr = msg_to_cv_bridge(msg);

  // Run inference
  sensor_msgs::ImagePtr label_msg;
  sensor_msgs::ImagePtr color_label_msg;
  sensor_msgs::ImagePtr uncertainty_msg;
  std::tie(label_msg, color_label_msg, uncertainty_msg) = inference(cv_ptr->image);

  // Set header
  label_msg->header = msg->header;
  color_label_msg->header = msg->header;
  uncertainty_msg->header = msg->header;

  pub_label_image_.publish(label_msg);
  pub_color_image_.publish(color_label_msg);
  pub_uncertainty_image_.publish(uncertainty_msg);
}

/*
 * image_inference_srv_callback : Callback for the service
 */
bool
PyTorchSegROS::image_inference_srv_callback(semantic_segmentation_srvs::GetLabelAndProbability::Request  & req,
                                             semantic_segmentation_srvs::GetLabelAndProbability::Response & res)
{
  ROS_INFO("[PyTorchSegROS image_inference_srv_callback] Start");

  // Convert the image message to a cv_bridge object
  cv_bridge::CvImagePtr cv_ptr = msg_to_cv_bridge(req.img);

  // Run inference
  sensor_msgs::ImagePtr label_msg;
  sensor_msgs::ImagePtr color_label_msg;
  sensor_msgs::ImagePtr uncertainty_msg;
  std::tie(label_msg, color_label_msg, uncertainty_msg) = inference(cv_ptr->image);

  res.label_img = *label_msg;
  res.colorlabel_img = *color_label_msg;
  res.uncertainty_img = *uncertainty_msg;

  return true;
}

/*
 * inference : Forward the given input image through the network and return the inference result
 */
std::tuple<sensor_msgs::ImagePtr, sensor_msgs::ImagePtr, sensor_msgs::ImagePtr>
PyTorchSegROS::inference(cv::Mat & input_img)
{

  // The size of the original image, to which the result of inference is resized back.
  int height_orig = input_img.size().height;
  int width_orig  = input_img.size().width;

  cv::Size s(480, 256);
  // Resize the input image
  cv::resize(input_img, input_img, s);

  at::Tensor input_tensor;
  pt_wrapper_ptr_->img2tensor(input_img, input_tensor);

  // Normalize from [0, 255] -> [0, 1]
  input_tensor /= 255.0;
  // z-normalization
  std::vector<float> mean_vec{0.485, 0.456, 0.406};
  std::vector<float> std_vec{0.229, 0.224, 0.225};
  for(int i = 0; i < mean_vec.size(); i++) {
    input_tensor[0][i] = (input_tensor[0][i] - mean_vec[i]) / std_vec[i];
  }
//  std::cout << input_tensor.sizes() << std::endl;

  // Execute the model and turn its output into a tensor.
  at::Tensor segmentation;
  segmentation = pt_wrapper_ptr_->get_output(input_tensor);

  at::Tensor output_args = pt_wrapper_ptr_->get_argmax(segmentation);

  // Uncertainty of segmentation
  at::Tensor uncertainty = pt_wrapper_ptr_->get_entropy(segmentation, true);
  uncertainty = (uncertainty[0]*255).to(torch::kCPU).to(torch::kByte);

  // Convert to OpenCV
  cv::Mat label;
  cv::Mat uncertainty_cv;
  pt_wrapper_ptr_->tensor2img(output_args[0], label);
  pt_wrapper_ptr_->tensor2img(uncertainty, uncertainty_cv);

  // Set the size
  cv::Size s_orig(width_orig, height_orig);
  // Resize the input image back to the original size
  cv::resize(label, label, s_orig, cv::INTER_NEAREST);
  cv::resize(uncertainty_cv, uncertainty_cv, s_orig, cv::INTER_LINEAR);
  // Generate color label image
  cv::Mat color_label;
  label_to_color(label, color_label);

  // Generate an image message
  sensor_msgs::ImagePtr label_msg = cv_bridge::CvImage(std_msgs::Header(), "mono8", label).toImageMsg();
  sensor_msgs::ImagePtr color_label_msg = cv_bridge::CvImage(std_msgs::Header(), "rgb8", color_label).toImageMsg();
  sensor_msgs::ImagePtr uncertainty_msg = cv_bridge::CvImage(std_msgs::Header(), "mono8", uncertainty_cv).toImageMsg();
  
  return std::forward_as_tuple(label_msg, color_label_msg, uncertainty_msg);
}

/*
 * label_to_color : Convert a label image to color label image for visualization
 */ 
void
PyTorchSegROS::label_to_color(cv::Mat& label, cv::Mat& color)
{
  cv::cvtColor(label, color, CV_GRAY2BGR);
  cv::LUT(color, colormap_, color);
}

/*
 * msg_to_cv_bridge : Generate a cv_image pointer instance from a given image message pointer
 */
cv_bridge::CvImagePtr
PyTorchSegROS::msg_to_cv_bridge(sensor_msgs::ImageConstPtr msg)
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
    return nullptr;
  }

  return cv_ptr;
}

/*
 * msg_to_cv_bridge : Generate a cv_image pointer instance from a given message
 */
cv_bridge::CvImagePtr
PyTorchSegROS::msg_to_cv_bridge(sensor_msgs::Image msg)
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
    return nullptr;
  }

  return cv_ptr;
}
