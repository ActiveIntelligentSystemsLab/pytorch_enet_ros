/*
 * A ROS node to do inference using PyTorch model
 * Shigemichi Matsuzaki
 *
 */

#ifndef PYTORCH_ENET_ROS
#define PYTORCH_ENET_ROS

#include <ros/ros.h>

#include <opencv2/opencv.hpp>

#include<image_transport/image_transport.h>
#include<cv_bridge/cv_bridge.h>

#include"pytorch_cpp_wrapper/pytorch_cpp_wrapper.h"

#include <iostream>
#include <memory>

class PyTorchENetROS {
private:
  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;

  image_transport::Subscriber sub_image_;
  image_transport::Publisher  pub_label_image_;
  image_transport::Publisher  pub_color_image_;

  PyTorchCppWrapper pt_wrapper_;

public:
  PyTorchENetROS();

  void image_callback(const sensor_msgs::ImageConstPtr& msg); 
  void inference(cv::Mat & input_image);

};

#endif
