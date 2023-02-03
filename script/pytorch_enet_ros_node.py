#!/usr/bin/python3
import os
import time
from distutils.util import strtobool
import argparse
import glob
#from tqdm import tqdm

from torchvision import transforms
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode
import torch
from model.segmentation.espnet_ue_cosine import espnetue_seg
from utilities.pseudo_label_generator import get_output
import PIL
import numpy as np

import sensor_msgs
from std_srvs.srv import Trigger, TriggerResponse
from cv_bridge import CvBridge
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


def get_image_and_mask(data_path: str):
    """Get paths to training images and masks

    Parameters
    ----------
    data_path : `str`
        Path to the directory where the images are stored

    Returns
    -------
    `list`
        List of lists of paths to images and masks
    """
    image_list = sorted(glob.glob(os.path.join(data_path, "rgb_*.png")))
    mask_list = sorted(glob.glob(os.path.join(data_path, "mask_*.png")))

    return image_list, mask_list


class MyObject():
    def __init__(self):
        pass


class PyTorchENetROS(InferenceNodeBase):
    def __init__(self, args, is_cuda=True):
        """

        Parameters
        ----------
        args: `argparse.ArgumentParser`
            A dictionary of arguments

        """
        super().__init__(size=(256, 480))

        self.ready = False

        args.model = rospy.get_param("~model", args.model)
        args.weights = rospy.get_param("~weights", args.weights)
        args.classes = rospy.get_param("~classes", args.classes)
        self.is_jit = rospy.get_param("~is_jit", True)

        rospy.loginfo("model:   " + args.model)
        rospy.loginfo("weights: " + args.weights)
        rospy.loginfo("classes: " + str(args.classes))
        rospy.loginfo("is_jit:  " + str(self.is_jit))

        #
        # Get the model
        #
        # self.device = 'cuda' if is_cuda else 'cpu'
        if self.is_jit:
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

        # Service for imprinting
        self.train_data_path = rospy.get_param("~train_data_path", "/tmp/")
        self.imprinting_srv = rospy.Service(
            '~imprint',
            Trigger,
            self.imprinting_srv_callback)

        self.bridge = CvBridge()

        # self.colors = [0, 255, 0, 0, 255, 255, 255, 0, 0, 255, 255, 0, 0, 0, 0]
        self.colors = [0, 255, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255, ]

        #
        # Definition of publishers
        #
        self.label_pub = rospy.Publisher(
            "~label",
            sensor_msgs.msg.Image,
            queue_size=1,
        )
        self.color_label_pub = rospy.Publisher(
            "~color_label",
            sensor_msgs.msg.Image,
            queue_size=1,
        )
        self.trav_pub = rospy.Publisher(
            "~prob",
            sensor_msgs.msg.Image,
            queue_size=1,
        )

        #
        # Definition of a subscriber
        #
        self.image_sub = rospy.Subscriber(
            "~image", sensor_msgs.msg.Image, self.image_callback, queue_size=1)

        rospy.loginfo("=== Ready ===")
        self.ready = True

    def image_callback(self, img_msg):
        """Callback for image message

        Parameters
        ----------
        img_msg: sensor_msgs.msg.Image
            Image message

        """
        if not self.ready:
            return

        pil_image, _, _ = imgmsg_to_pil(img_msg)
        tensor_img = self.transforms(pil_image).unsqueeze(0).to(self.device)

        # Get output
        with torch.no_grad():
            output = self.model(tensor_img)

            print("Class num: {}".format(output["main"].size(1)))

            # Segmentation
            if self.is_jit:
                label_tensor = torch.argmax(
                    output[0] + 0.5 * output[1], dim=1).squeeze()
            else:
                label_tensor = torch.argmax(
                    output["main"] + 0.5 * output["aux"], dim=1).squeeze()

            label_pil = PIL.Image.fromarray(
                label_tensor.byte().cpu().numpy()).resize(pil_image.size)
            self.label_pub.publish(pil_to_imgmsg(label_pil))

            label_pil.putpalette(self.colors)
            label_pil = label_pil.convert('RGB')
            self.color_label_pub.publish(pil_to_imgmsg(label_pil))

            # Traversability
            # trav_pil = PIL.Image.fromarray(
            #     output[2].squeeze().cpu().numpy()).resize(pil_image.size)

            # trav_msg = self.bridge.cv2_to_imgmsg(
            #     output[2].squeeze().cpu().numpy(), "32FC1")
            # self.trav_pub.publish(trav_msg)

    def imprinting_srv_callback(self, req: Trigger):
        """Callback function for 

        Parameters
        ----------
        req : `Trigger`
            Request
        """
        self.model.eval()
        try:
            with torch.no_grad():
                #
                # Train
                #
                train_image_list, train_mask_list = get_image_and_mask(
                    self.train_data_path)

                image_list = []
                mask_list = []
                for image_path, mask_path in zip(train_image_list, train_mask_list):

                    # image = batch["rgb"].to(self.device)
                    # mask = batch["trav_mask"].to(self.device)
                    pil_image = PIL.Image.open(image_path)
                    pil_mask = PIL.Image.open(mask_path)

                    tensor_image = self.transforms(
                        pil_image).unsqueeze(0).to(self.device)
                    tensor_mask = torch.LongTensor(np.array(pil_mask).astype(
                        np.int64)).unsqueeze(dim=0).to(self.device)
                    tensor_mask = F.resize(
                        tensor_mask, size=self.size, interpolation=InterpolationMode.NEAREST)

                    # Output: tensor, KLD: tensor, feature: tensor
                    output_prob, _, _ = get_output(
                        self.model, tensor_image, is_numpy=False)
                    argmax_output = torch.argmax(output_prob, dim=1)

                    # Extract MISCLASSIFIED (argmax != 0) plant regions
                    tensor_mask[argmax_output == 0] = 0
                    tensor_mask = tensor_mask // 255

                    image_list.append(tensor_image)
                    mask_list.append(tensor_mask)

                    print(tensor_mask.max())
        except Exception as e:
            rospy.logerr(e)
            return TriggerResponse(False, "Failed feature accumlation: '{}'".format(e))

        #
        # Imprinting
        #
        # try:
        self.model.imprint(image_list, mask_list, alpha=0.26)
        # except Exception as e:
        #    rospy.logerr(e)
        #    return TriggerResponse(False, "Failed imprinting: '{}'".format(e))

        return TriggerResponse(True, "succeeded")


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
    obj = PyTorchENetROS(
        args, is_cuda=torch.cuda.is_available())

    rospy.spin()


if __name__ == '__main__':
    main()
