#!/usr/bin/python                                             
# -*- coding: utf-8 -*- 

# ============================================
__author__ = "ShigemichiMatsuzaki"
__maintainer__ = "ShigemichiMatsuzaki"
# ============================================

import rospy
import cv2
import message_filters
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
import copy

class Visualizer:
    """ """
    def __init__(self):
        """ Constructor of Visualizer class

        - define the class variables (ROS publisher and subscribers etc.)

        """
        # Subscribers
        self.image_sub = message_filters.Subscriber('image', Image)
        self.start_point_sub = message_filters.Subscriber('start_point', PointStamped)
        self.end_point_sub = message_filters.Subscriber('end_point', PointStamped)
        # Time synchronizer
        self.ts = message_filters.TimeSynchronizer([self.image_sub, self.start_point_sub, self.end_point_sub], 100)
        self.ts.registerCallback(self.image_points_callback)

        # Publisher
        self.image_pub = rospy.Publisher('image_and_path', Image, queue_size=100)

        self.bridge = CvBridge()

    def image_points_callback(self, img_msg, start_point_msg, end_point_msg):
        """Callback of image and point messages

        Args:
            img_msg(sensor_msgs/Image)
            start_point_msg(geometry_msgs/PointStamped)
            end_point_msg(geometry_msgs/PointStamped)
        """
        # Convert the image message to 
        cv_image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='passthrough')

        cv_image_with_line = self.draw_line(cv_image,
            (int(start_point_msg.point.x), int(start_point_msg.point.y)),
            (int(end_point_msg.point.x), int(end_point_msg.point.y)))

        vis_img_msg = self.bridge.cv2_to_imgmsg(cv_image_with_line)

        self.image_pub.publish(vis_img_msg)

    def draw_line(self, cv_image, start_point, end_point):
        """Draw a line, whose start and end points are given as PointStamped messages, on the image

        Args:
            cv_image(OpenCV image)
            start_point_msg(geometry_msgs/PointStamped)
            end_point_msg(geometry_msgs/PointStamped)
        
        Return:
            OpenCV image with a line drawn
        """
        ret_image = copy.deepcopy(cv_image)

        cv2.line(ret_image, start_point, end_point, color=(0, 0, 255), thickness=10)

        return ret_image

def main():
    """Main function to initialize the ROS node"""
    rospy.init_node("visualizer")

    visualizer = Visualizer()

    rospy.loginfo('visualizer is initialized')
    
    rospy.spin()  


if __name__=='__main__':
    main()