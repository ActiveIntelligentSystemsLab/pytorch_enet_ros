/*
 * A ROS node to do inference using PyTorch model
 * Shigemichi Matsuzaki
 *
 */

#include <pytorch_enet_ros/pytorch_seg_trav_path.h>

PyTorchSegTravPathROS::PyTorchSegTravPathROS(ros::NodeHandle & nh) 
  : it_(nh), nh_(nh)
{
  sub_image_ = it_.subscribe("image", 1, &PyTorchSegTravPathROS::image_callback, this);
  pub_label_image_ = it_.advertise("label", 1);
  pub_color_image_ = it_.advertise("color_label", 1);
  pub_prob_image_ = it_.advertise("prob", 1);
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

void
PyTorchSegTravPathROS::image_callback(const sensor_msgs::ImageConstPtr& msg)
{
  ROS_INFO("[PyTorchSegTravPathROS image_callback] Let's start!!");

  // Convert the image message to a cv_bridge object
  cv_bridge::CvImagePtr cv_ptr = msg_to_cv_bridge(msg);

  // Run inference
  sensor_msgs::ImagePtr label_msg;
  sensor_msgs::ImagePtr color_label_msg;
  sensor_msgs::ImagePtr prob_msg;
  geometry_msgs::PointPtr start_point_msg;
  geometry_msgs::PointPtr end_point_msg;
  std::tie(label_msg, color_label_msg, prob_msg, start_point_msg, end_point_msg) = inference(cv_ptr->image);

  // Set header
  label_msg->header = msg->header;
  color_label_msg->header = msg->header;
  prob_msg->header = msg->header;

  pub_label_image_.publish(label_msg);
  pub_color_image_.publish(color_label_msg);
  pub_prob_image_.publish(prob_msg);
}

/*
 * image_inference_srv_callback : Callback for the service
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
  geometry_msgs::PointPtr start_point_msg;
  geometry_msgs::PointPtr end_point_msg;
  std::tie(label_msg, color_label_msg, prob_msg, start_point_msg, end_point_msg) = inference(cv_ptr->image);

  res.label_img = *label_msg;
  res.colorlabel_img = *color_label_msg;
  res.prob_img = *prob_msg;

  return true;
}

/*
 * inference : Forward the given input image through the network and return the inference result
 */
std::tuple<sensor_msgs::ImagePtr, sensor_msgs::ImagePtr, sensor_msgs::ImagePtr, geometry_msgs::PointPtr, geometry_msgs::PointPtr>
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
//  std::cout << input_tensor.sizes() << std::endl;

  // Execute the model and turn its output into a tensor.
  at::Tensor segmentation;
  at::Tensor prob;
  at::Tensor points;
  std::tie(segmentation, prob, points) = pt_wrapper_.get_output(input_tensor);
//  at::Tensor output = pt_wrapper_.get_output(input_tensor);
  // Calculate argmax to get a label on each pixel
//  at::Tensor output_args = pt_wrapper_.get_argmax(output);

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
  geometry_msgs::PointPtr start_point_msg(new geometry_msgs::Point), end_point_msg(new geometry_msgs::Point);
  ROS_INFO("Let's get points");
  points = points.to(torch::kCPU);
  auto points_a = points.accessor<float,2>();
  start_point_msg->x = points_a[0][0] * width_orig;
  start_point_msg->y = points_a[0][1] * height_orig;
  end_point_msg->x = points_a[0][2] * width_orig;
  end_point_msg->y = points_a[0][3] * height_orig;
  
  ROS_INFO("Start: (%.3f, %.3f), End: (%.3f, %.3f)", start_point_msg->x, start_point_msg->y, end_point_msg->x, end_point_msg->y);


  return std::forward_as_tuple(label_msg, color_label_msg, prob_msg, start_point_msg, end_point_msg);
}

/*
 * label_to_color : Convert a label image to color label image for visualization
 */ 
void
PyTorchSegTravPathROS::label_to_color(cv::Mat& label, cv::Mat& color)
{
  cv::cvtColor(label, color, CV_GRAY2BGR);
  cv::LUT(color, colormap_, color);
}

/*
 * msg_to_cv_bridge : Generate a cv_image pointer instance from a given image message pointer
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

/*
 * msg_to_cv_bridge : Generate a cv_image pointer instance from a given message
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

// cv::Mat PyTorchSegTravPathROS::getLookUpTable(const std::string dataset)
// {
//   cv::Mat lookUpTable(1, 256, CV_8UC3);
// 
//   if(dataset == "greenhouse") {
// 
//   } else if(dataset == "camvid") {
// 
//   }
// }
