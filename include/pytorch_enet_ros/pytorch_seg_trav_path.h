/*
 * A ROS node to do inference using PyTorch model
 * Shigemichi Matsuzaki
 *
 */

#ifndef PYTORCH_SEG_TRAV_PATH
#define PYTORCH_SEG_TRAV_PATH

#include <ros/ros.h>

#include <opencv2/opencv.hpp>
#include<image_transport/image_transport.h>
#include<cv_bridge/cv_bridge.h>
#include<geometry_msgs/Point.h>
//#include<semantic_segmentation_srvs/GetLabelImage.h>
#include<semantic_segmentation_srvs/GetLabelAndProbability.h>

#include"pytorch_cpp_wrapper/pytorch_cpp_wrapper_seg_trav_path.h"

#include <iostream>
#include <memory>
#include <tuple>

class PyTorchSegTravPathROS {
private:
  ros::NodeHandle nh_;

  ros::ServiceServer get_label_image_server_;

  image_transport::ImageTransport it_;

  image_transport::Subscriber sub_image_;
  image_transport::Publisher  pub_label_image_;
  image_transport::Publisher  pub_color_image_;
  image_transport::Publisher  pub_prob_image_;

//  PyTorchCppWrapper pt_wrapper_;
  PyTorchCppWrapperSegTravPath pt_wrapper_;

  cv::Mat colormap_;

public:
  PyTorchSegTravPathROS(ros::NodeHandle & nh); 

  void image_callback(const sensor_msgs::ImageConstPtr& msg); 
  std::tuple<sensor_msgs::ImagePtr, sensor_msgs::ImagePtr, sensor_msgs::ImagePtr, geometry_msgs::PointPtr, geometry_msgs::PointPtr> inference(cv::Mat & input_image);
  bool image_inference_srv_callback(semantic_segmentation_srvs::GetLabelAndProbability::Request  & req,
                                    semantic_segmentation_srvs::GetLabelAndProbability::Response & res);
  cv_bridge::CvImagePtr msg_to_cv_bridge(sensor_msgs::ImageConstPtr msg);
  cv_bridge::CvImagePtr msg_to_cv_bridge(sensor_msgs::Image msg);
  void label_to_color(cv::Mat& label, cv::Mat& color_label);
};

#endif
