#!/usr/bin/python3
from model.segmentation.espnet_ue_cosine import espnetue_seg
import torch.nn.functional as F
import torch
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import message_filters
import sys
from distutils.util import strtobool
import argparse
import rospy

RESTORE_FROM = '/root/catkin_ws/src/pytorch_enet_ros/models/espnet_ue_cosine_best_6.pth'
MODEL = 'espnetue_cosine'
GPU = 0


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    # shared by train & val
    # data
    parser.add_argument("--weights", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    # gpu
    parser.add_argument("--gpu", type=int, default=GPU,
                        help="choose gpu device.")
    # model related params
    parser.add_argument('--s', type=float, default=2.0,
                        help='Factor by which channels will be scaled')
    parser.add_argument('--model', default='espnetue_cosine',
                        help='Which model? basic= basic CNN model, res=resnet style)')
    parser.add_argument('--dataset', default='greenhouse', help='Dataset')
    parser.add_argument('--classes', default=3, help='The number of classes')
    # Parameters related to cosine-based softmax
    parser.add_argument('--cos-margin', default=0.5, type=float,
                        help='Angle margin')
    parser.add_argument('--cos-logit-scale', default=30.0, type=float,
                        help='Scale factor for the final logits')
    parser.add_argument('--is-easy-margin', default=False, type=strtobool,
                        help='Whether to use an easy margin')

    return parser.parse_args()


class MyObject():
    def __init__(self):
        pass


class pytorch_enet_ros:
    def __init__(self, is_cuda=True):
        """

        Parameters
        ----------
        args: argparse.ArgumentParser
            A dictionary of arguments

        """
        #
        # Get params
        #
        args = MyObject()
        setattr(args, 'model', MODEL)
        setattr(args, 'weights', RESTORE_FROM)
        setattr(args, 'classes', 3)
        setattr(args, 'dataset', 'greenhouse')
        setattr(args, 's', 2.0)
        rospy.get_param("model", args.model)
        rospy.get_param("weights", args.weights)
        rospy.get_param("classes", args.classes)

        rospy.loginfo("model: " + args.model)
        rospy.loginfo("weights: " + args.weights)
        rospy.loginfo("classes: " + str(args.classes))

        #
        # Get the model
        #
        self.device = 'cuda' if is_cuda else 'cpu'
        self.model = espnetue_seg(
            args, load_entire_weights=True, fix_pyr_plane_proj=True, is_wide=False)

        self.model.to(self.device)

        #
        # Define a subscriber
        #
        self.image_sub = rospy.Subscriber(
            "~image", Image, self.image_callback, queue_size=1)

        self.bridge = CvBridge()

        rospy.loginfo("=== Ready ===")

    def image_callback(self, img_msg):
        """Callback for image message

        Parameters
        ----------
        img_msg: sensor_msgs.Image
            Image message

        """
        # Image message to OpenCV image
        cv_img = self.bridge.imgmsg_to_cv2(
            img_msg, desired_encoding='passthrough')
        cv_img = cv2.resize(cv_img, (256, 480))
        cv_img = cv_img.transpose(2, 1, 0)
        cv_img = cv_img[:3, :, :]

        # OpenCV to torch.Tensor
        tensor_img = torch.from_numpy(
            cv_img.astype(np.float32)).to(self.device)
        tensor_img = torch.unsqueeze(tensor_img, dim=0)

        # Get output
        output = self.model(tensor_img)


def main():
    """Main function"""

#    args = get_arguments()
    rospy.init_node('pytorch_seg_online', anonymous=True)
    obj = pytorch_enet_ros()

    rospy.spin()


if __name__ == '__main__':
    main()
