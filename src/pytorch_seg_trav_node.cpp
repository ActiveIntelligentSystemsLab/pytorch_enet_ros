/*
 * A wrapper class of PyTorch C++ to use PyTorch model
 * Shigemichi Matsuzaki
 *
 */

#include<ros/ros.h>
#include<pytorch_ros/pytorch_seg_trav_ros.h>

int main(int argc, char* argv[]) {
  // Initialize the node
  ros::init(argc, argv, "pytorch_seg_trav_ros");

  ros::NodeHandle nh("~");
  //ros::NodeHandle nh;

  // Initialize the class
  PyTorchSegTravROS pytorch_seg_trav_ros(nh);

  ROS_INFO("[PyTorchSegTravROS] The node has been initialized");

  ros::spin();
  
//  ros::Rate rate(30.0);
//  while(ros::ok()) {
//    ros::spinOnce();
//
//    rate.sleep();
//  }
}
