# pytorch_scene_recognition_ros

## 1. Overview

A ROS package to use [LibTorch](https://pytorch.org/cppdocs/), a PyTorch C++ API, for inference with our scene recognition models.

A Docker environment for running this package is [here](https://github.com/ActiveIntelligentSystemsLab/pytorch-enet-docker).
This package is **only tested in the virtual environment**.

## 2. Requirements

- PyTorch with LibTorch built
  - Tested with 1.5.0
- [semantic_segmentation_srvs/GetLabelAndProbability](https://github.com/ActiveIntelligentSystemsLab/aisl_utils/blob/master/aisl_srvs/semantic_segmentation_srv/srv/GetLabelAndProbability.srv)

## 3. Nodes

### 3.1 `pytorch_seg_trav_path_node`

A node to use a multi-task model for semantic segmentation, traversability estimation, and path estimation.

#### **3.1.1 Subscribed topics**

- `image` ([sensor_msgs/Image](http://docs.ros.org/melodic/api/sensor_msgs/html/msg/Image.html))

    An input image

#### **3.1.2 Published topics**

- `label` ([sensor_msgs/Image](http://docs.ros.org/melodic/api/sensor_msgs/html/msg/Image.html))

    Image that stores label indices of each pixel

- `color_label` ([sensor_msgs/Image](http://docs.ros.org/melodic/api/sensor_msgs/html/msg/Image.html))

    Image that stores color labels of each pixel (for visualization)

- `prob` ([sensor_msgs/Image](http://docs.ros.org/melodic/api/sensor_msgs/html/msg/Image.html))

    Image that stores *traversability* of each pixel

- `start_point` ([geometry_msgs/PointStamped](http://docs.ros.org/en/melodic/api/geometry_msgs/html/msg/PointStamped.html))

    Start point of the estimated path line

- `end_point` ([geometry_msgs/PointStamped](http://docs.ros.org/en/melodic/api/geometry_msgs/html/msg/PointStamped.html))

    End point of the estimated path line

#### **3.1.3 Service**

- `get_label_image` ([semantic_segmentation_srvs/GetLabelAndProbability](https://github.com/ActiveIntelligentSystemsLab/aisl_utils/blob/master/aisl_srvs/semantic_segmentation_srv/srv/GetLabelAndProbability.srv))

    Return inference results (segmentation and traversability) for a given image.

### **3.2 visualizer.py**

#### **3.2.1 Subscribed topics**

- `image` ([sensor_msgs/Image](http://docs.ros.org/melodic/api/sensor_msgs/html/msg/Image.html))

    An input image

- `start_point` ([geometry_msgs/PointStamped](http://docs.ros.org/en/melodic/api/geometry_msgs/html/msg/PointStamped.html))

    Start point of the estimated path line from the inference node

- `end_point` ([geometry_msgs/PointStamped](http://docs.ros.org/en/melodic/api/geometry_msgs/html/msg/PointStamped.html))

    End point of the estimated path line from the inference node

#### **3.2.2 Published topics**

- `image_with_path` ([sensor_msgs/Image](http://docs.ros.org/melodic/api/sensor_msgs/html/msg/Image.html))

    An image with the path overlaid

## 4. How to run the node

```
roslaunch pytorch_enet_ros.launch image:=<image topic name> model_name:=<model name>
```

## 5. Weight files

The ROS nodes in this package use models saved as a serialized Torch Script file.

At this moment, we don't provide a script to generate the weight files.

Refer to [this page](https://pytorch.org/tutorials/advanced/cpp_export.html) to get the weight file.

### CAUTION
If the version of PyTorch that runs this ROS package and that you generate your weight file (serialized Torch Script) do not match, the ROS node may fail to import the weights.

For example, if you use [our Docker environment](https://github.com/ActiveIntelligentSystemsLab/pytorch-enet-docker), the weights should be generated using PyTorch 1.5.0.

## 6. Color map

For visualization of semantic segmentation, we use a color map image.

It is a 1xC PNG image file (C: The number of classes), where 
the color of class i is stored in the pixel at (1, i).