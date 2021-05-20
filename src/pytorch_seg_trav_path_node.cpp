/*
 * A wrapper class of PyTorch C++ to use PyTorch model
 * Shigemichi Matsuzaki
 *
 */

#include<ros/ros.h>
#include<pytorch_ros/pytorch_seg_trav_path_ros.h>

int main(int argc, char* argv[]) {
  // Initialize the node
  ros::init(argc, argv, "pytorch_seg_trav_path");

  ros::NodeHandle nh("~");

  // Initialize the class
  PyTorchSegTravPathROS pytorch_ros(nh);

  ROS_INFO("[PyTorchENetROS] The node has been initialized");

  ros::spin();

}
