/*
 * A ROS node to do inference using PyTorch model
 * Shigemichi Matsuzaki
 *
 */

#include <pytorch_ros/pytorch_seg_trav_path_ros.h>

PyTorchSegTravPathROS::PyTorchSegTravPathROS(ros::NodeHandle & nh) 
  : it_(nh), nh_(nh)
{
  sub_image_ = it_.subscribe("image", 1, &PyTorchSegTravPathROS::image_callback, this);
  pub_label_image_ = it_.advertise("label", 1);
  pub_color_image_ = it_.advertise("color_label", 1);
  pub_prob_image_ = it_.advertise("prob", 1);
  pub_start_point_ = nh_.advertise<geometry_msgs::PointStamped>("start_point", 1);
  pub_end_point_ = nh_.advertise<geometry_msgs::PointStamped>("end_point", 1);
  get_label_image_server_ = nh_.advertiseService("get_label_image", &PyTorchSegTravPathROS::image_inference_srv_callback, this);

  // Import the model
  std::string filename;
  nh_.param<std::string>("model_file", filename, "");
  if(!pt_wrapper_.import_module(filename)) {
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

/** 
 * @brief Image callback
 * @param[in] msg  Message
 */
void
PyTorchSegTravPathROS::image_callback(const sensor_msgs::ImageConstPtr& msg)
{
  ROS_INFO("[PyTorchSegTravPathROS image_callback] Let's start!!");

  // Convert the image message to a cv_bridge object
  cv_bridge::CvImagePtr cv_ptr = msg_to_cv_bridge(msg);
  stamp_of_current_image_ = msg->header.stamp;

  // Run inference
  sensor_msgs::ImagePtr label_msg;
  sensor_msgs::ImagePtr color_label_msg;
  sensor_msgs::ImagePtr prob_msg;
  geometry_msgs::PointStampedPtr start_point_msg;
  geometry_msgs::PointStampedPtr end_point_msg;
  std::tie(label_msg, color_label_msg, prob_msg, start_point_msg, end_point_msg) = inference(cv_ptr->image);

  // Set header
  label_msg->header = msg->header;
  color_label_msg->header = msg->header;
  prob_msg->header = msg->header;

  // Publish the messages
  pub_label_image_.publish(label_msg);
  pub_color_image_.publish(color_label_msg);
  pub_prob_image_.publish(prob_msg);
  pub_start_point_.publish(start_point_msg);
  pub_end_point_.publish(end_point_msg);
}

/** 
 * @brief Main function for inference using the model
 * @param[in] input_image OpenCV image 
 * @return    A tuple of messages of the inference results
 */
bool
PyTorchSegTravPathROS::image_inference_srv_callback(semantic_segmentation_srvs::GetLabelAndProbability::Request  & req,
                                             semantic_segmentation_srvs::GetLabelAndProbability::Response & res)
{
  ROS_INFO("[PyTorchSegTravPathROS image_inference_srv_callback] Start");

  // Convert the image message to a cv_bridge object
  cv_bridge::CvImagePtr cv_ptr = msg_to_cv_bridge(req.img);

  // Run inference
  sensor_msgs::ImagePtr label_msg;
  sensor_msgs::ImagePtr color_label_msg;
  sensor_msgs::ImagePtr prob_msg;
  geometry_msgs::PointStampedPtr start_point_msg;
  geometry_msgs::PointStampedPtr end_point_msg;
  std::tie(label_msg, color_label_msg, prob_msg, start_point_msg, end_point_msg) = inference(cv_ptr->image);

  res.label_img = *label_msg;
  res.colorlabel_img = *color_label_msg;
  res.prob_img = *prob_msg;

  return true;
}

/** 
 * @brief Service callback
 * @param[in] req  Request
 * @param[in] res  Response
 * @return    True if the service succeeded
 */
std::tuple<sensor_msgs::ImagePtr, sensor_msgs::ImagePtr, sensor_msgs::ImagePtr, geometry_msgs::PointStampedPtr, geometry_msgs::PointStampedPtr>
PyTorchSegTravPathROS::inference(cv::Mat & input_img)
{

  // The size of the original image, to which the result of inference is resized back.
  int height_orig = input_img.size().height;
  int width_orig  = input_img.size().width;

  cv::Size s(480, 256);
  // Resize the input image
  cv::resize(input_img, input_img, s);

  at::Tensor input_tensor;
  pt_wrapper_.img2tensor(input_img, input_tensor);

  // Normalize from [0, 255] -> [0, 1]
  input_tensor /= 255.0;
  // z-normalization
  std::vector<float> mean_vec{0.485, 0.456, 0.406};
  std::vector<float> std_vec{0.229, 0.224, 0.225};
  for(int i = 0; i < mean_vec.size(); i++) {
    input_tensor[0][i] = (input_tensor[0][i] - mean_vec[i]) / std_vec[i];
  }

  // Execute the model and turn its output into a tensor.
  at::Tensor segmentation;
  at::Tensor prob;
  at::Tensor points;
  std::tie(segmentation, prob, points) = pt_wrapper_.get_output(input_tensor);

  at::Tensor output_args = pt_wrapper_.get_argmax(segmentation);

  // Convert to OpenCV
  cv::Mat label;
  cv::Mat prob_cv;
  pt_wrapper_.tensor2img(output_args[0], label);
  pt_wrapper_.tensor2img((prob[0][0]*255).to(torch::kByte), prob_cv);

  // Set the size
  cv::Size s_orig(width_orig, height_orig);
  // Resize the input image back to the original size
  cv::resize(label, label, s_orig, cv::INTER_NEAREST);
  cv::resize(prob_cv, prob_cv, s_orig, cv::INTER_LINEAR);
  // Generate color label image
  cv::Mat color_label;
  label_to_color(label, color_label);

  // Generate an image message and point messages
  sensor_msgs::ImagePtr label_msg = cv_bridge::CvImage(std_msgs::Header(), "mono8", label).toImageMsg();
  sensor_msgs::ImagePtr color_label_msg = cv_bridge::CvImage(std_msgs::Header(), "rgb8", color_label).toImageMsg();
  sensor_msgs::ImagePtr prob_msg = cv_bridge::CvImage(std_msgs::Header(), "mono8", prob_cv).toImageMsg();
  geometry_msgs::PointStampedPtr start_point_msg(new geometry_msgs::PointStamped), end_point_msg(new geometry_msgs::PointStamped);
  std::tie(start_point_msg, end_point_msg) = tensor_to_points(points, width_orig, height_orig);
  
  return std::forward_as_tuple(label_msg, color_label_msg, prob_msg, start_point_msg, end_point_msg);
}

/** 
 * @brief Convert a tensor with a size of (1, 4) to start and end points (x, y)
 * @param[in] point_tensor  (1, 4) tensor
 * @param[in] width         Original width of the image
 * @param[in] height        Original height of the image
 * @return                  A tuple of start and end points as geometry_msgs::PointStampedPtr
 */
std::tuple<geometry_msgs::PointStampedPtr, geometry_msgs::PointStampedPtr>
PyTorchSegTravPathROS::tensor_to_points(const at::Tensor point_tensor, const int & width, const int & height)
{
  geometry_msgs::PointStampedPtr start_point_msg(new geometry_msgs::PointStamped), end_point_msg(new geometry_msgs::PointStamped);
  // Important: put the data on the CPU before accessing the data.
  // Absense of this code will result in runtime error.
  at::Tensor points = point_tensor.to(torch::kCPU);
  auto points_a = points.accessor<float, 2>();

  // Initialize messgaes
  start_point_msg->header.stamp = stamp_of_current_image_;//ros::Time::now();
  start_point_msg->header.frame_id = "kinect2_rgb_optical_frame";
  end_point_msg->header.stamp = stamp_of_current_image_;//ros::Time::now();
  end_point_msg->header.frame_id = "kinect2_rgb_optical_frame";
  // Point tensor has coordinate values normalized with the width and height.
  // Therefore each value is multiplied by width or height.
  start_point_msg->point.x = points_a[0][0] * width;
  start_point_msg->point.y = points_a[0][1] * height;
  end_point_msg->point.x = points_a[0][2] * width;
  end_point_msg->point.y = points_a[0][3] * height;

  return std::forward_as_tuple(start_point_msg, end_point_msg);
}

/** 
  * @brief Convert a label image to color label image for visualization
  * @param[in]  label        Label image
  * @param[out] color_label  Color image mapped from the label image
  */
void
PyTorchSegTravPathROS::label_to_color(cv::Mat& label, cv::Mat& color_label)
{
  cv::cvtColor(label, color_label, CV_GRAY2BGR);
  cv::LUT(color_label, colormap_, color_label);
}

/** 
 * @brief Convert Image message to cv_bridge
 * @param[in] msg  Pointer of image message
 * @return    cv_bridge
 */
cv_bridge::CvImagePtr
PyTorchSegTravPathROS::msg_to_cv_bridge(sensor_msgs::ImageConstPtr msg)
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

/** 
 * @brief Convert Image message to cv_bridge
 * @param[in] msg  Image message
 * @return    cv_bridge
 */
cv_bridge::CvImagePtr
PyTorchSegTravPathROS::msg_to_cv_bridge(sensor_msgs::Image msg)
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
