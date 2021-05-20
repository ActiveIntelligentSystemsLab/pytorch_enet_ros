cmake_minimum_required(VERSION 2.8.3)
project(pytorch_ros)

add_compile_options(-std=c++14)

# Locate the cmake file of torchlib
set(Torch_DIR "/opt/pytorch/pytorch/torch/share/cmake/Torch/")
#set(Torch_DIR "$ENV{TORCH_ROOT}/torch/share/cmake/Torch/")

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  image_transport
  cv_bridge
  semantic_segmentation_srvs
  tf2
  tf2_ros
)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
find_package(OpenCV REQUIRED)

catkin_package(
 INCLUDE_DIRS include
 CATKIN_DEPENDS roscpp rospy std_msgs
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

## Declare a C++ library
add_library(${PROJECT_NAME}
  src/pytorch_seg_trav_path_ros.cpp
  src/pytorch_cpp_wrapper_seg_trav.cpp
  src/pytorch_cpp_wrapper_seg_trav_path.cpp
  src/pytorch_cpp_wrapper_base.cpp
)

target_link_libraries(${PROJECT_NAME}
  ${catkin_LIBRARIES}
  ${TORCH_LIBRARIES}
  ${Open_CV_LIBS}
  opencv_core opencv_highgui opencv_imgcodecs
)
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 14)

## Declare a C++ executable
## With catkin_make all packages are built within a single CMake context
## The recommended prefix ensures that target names across packages don't collide
add_executable(pytorch_seg_trav_path_node src/pytorch_seg_trav_path_node.cpp)

## Specify libraries to link a library or executable target against
target_link_libraries(pytorch_seg_trav_path_node
  ${catkin_LIBRARIES}
  ${PROJECT_NAME}
  ${TORCH_LIBRARIES}
)
