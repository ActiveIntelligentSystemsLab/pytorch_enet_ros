#!/usr/bin/python3
from model.segmentation.espnet_ue_cosine import espnetue_seg
import torch.nn.functional as F
import torch
import sensor_msgs
import PIL
from cv_bridge import CvBridge
import cv2
from distutils.util import strtobool
import argparse
import rospy
from rospy_pytorch_util.inference_node_base import InferenceNodeBase
from rospy_pytorch_util.utils import imgmsg_to_pil, pil_to_imgmsg

RESTORE_FROM = '/root/catkin_ws/src/pytorch_enet_ros/models/espnet_ue_cosine_best_6.pth'
RESTORE_FROM = '/root/catkin_ws/src/pytorch_enet_ros/models/espdnet_ue_trav_20210115-151110.pt'
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


class pytorch_enet_ros(InferenceNodeBase):
    def __init__(self, args, is_cuda=True):
        """

        Parameters
        ----------
        args: argparse.ArgumentParser
            A dictionary of arguments

        """
        super().__init__(size=(256, 480))

        args.model = rospy.get_param("~model", args.model)
        args.weights = rospy.get_param("~weights", args.weights)
        args.classes = rospy.get_param("~classes", args.classes)
        is_jit = rospy.get_param("~is_jit", True)

        rospy.loginfo("model:   " + args.model)
        rospy.loginfo("weights: " + args.weights)
        rospy.loginfo("classes: " + str(args.classes))
        rospy.loginfo("is_jit:  " + str(is_jit))

        #
        # Get the model
        #
        # self.device = 'cuda' if is_cuda else 'cpu'

        if is_jit:
            self.model = torch.jit.load(
                args.weights, map_location=torch.device(self.device))
        else:
            self.model = espnetue_seg(
                args,
                load_entire_weights=True,
                fix_pyr_plane_proj=True,
                is_wide=False
            )

            self.model.to(self.device)

        self.model.eval()

        #
        # Definition of a subscriber
        #
        self.image_sub = rospy.Subscriber(
            "~image", sensor_msgs.msg.Image, self.image_callback, queue_size=1)

        #
        # Definition of publishers
        self.label_pub = rospy.Publisher(
            "~label",
            sensor_msgs.msg.Image,
            queue_size=10,
        )
        self.color_label_pub = rospy.Publisher(
            "~color_label",
            sensor_msgs.msg.Image,
            queue_size=10,
        )
        self.trav_pub = rospy.Publisher(
            "~prob",
            sensor_msgs.msg.Image,
            queue_size=10,
        )

        self.bridge = CvBridge()

        self.colors = [0, 255, 0, 0, 255, 255, 255, 0, 0, 255, 255, 0, 0, 0, 0]

        rospy.loginfo("=== Ready ===")

    def image_callback(self, img_msg):
        """Callback for image message

        Parameters
        ----------
        img_msg: sensor_msgs.msg.Image
            Image message

        """
        # Image message to OpenCV image
        # cv_img = self.bridge.imgmsg_to_cv2(
        #     img_msg, desired_encoding='passthrough')
        # cv_img = cv2.resize(cv_img, (256, 480))
        # cv_img = cv_img.transpose(2, 1, 0)
        # cv_img = cv_img[:3, :, :]

        # # OpenCV to torch.Tensor
        # tensor_img = torch.from_numpy(
        #     cv_img.astype(np.float32)).to(self.device)
        # tensor_img = torch.unsqueeze(tensor_img, dim=0)

        pil_image, _, _ = imgmsg_to_pil(img_msg)
        tensor_img = self.transforms(pil_image).unsqueeze(0).to(self.device)

        # Get output

        with torch.no_grad():
            output = self.model(tensor_img)

            # Segmentation
            label_tensor = torch.argmax(
                output[0] + 0.5 * output[1], dim=1).squeeze()
            label_pil = PIL.Image.fromarray(
                label_tensor.byte().cpu().numpy()).resize(pil_image.size)
            self.label_pub.publish(pil_to_imgmsg(label_pil))

            label_pil.putpalette(self.colors)
            label_pil = label_pil.convert('RGB')
            self.color_label_pub.publish(pil_to_imgmsg(label_pil))

            # Traversability
            # trav_pil = PIL.Image.fromarray(
            #     output[2].squeeze().cpu().numpy()).resize(pil_image.size)
            trav_msg = self.bridge.cv2_to_imgmsg(
                output[2].squeeze().cpu().numpy(), "32FC1")
            self.trav_pub.publish(trav_msg)


def main():
    """Main function"""
    #
    # Get params
    #
    args = MyObject()
    setattr(args, 'model', MODEL)
    setattr(args, 'weights', RESTORE_FROM)
    setattr(args, 'classes', 3)
    setattr(args, 'dataset', 'greenhouse')
    setattr(args, 's', 2.0)

    # args = get_arguments()
    rospy.init_node('pytorch_seg_online', anonymous=False)
    obj = pytorch_enet_ros(
        args, is_cuda=torch.cuda.is_available())

    rospy.spin()


if __name__ == '__main__':
    main()
