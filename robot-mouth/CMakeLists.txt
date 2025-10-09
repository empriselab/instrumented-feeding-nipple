cmake_minimum_required(VERSION 3.0.2)
project(robot_mouth)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -mavx -O3")

find_package(Eigen3 REQUIRED)
message(STATUS "Eigen3_INCLUDE_DIRS = ${Eigen3_INCLUDE_DIRS}")

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
)

## Generate added messages and services with any dependencies listed here

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES robot_mouth
  CATKIN_DEPENDS roscpp rospy std_msgs
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${Eigen3_INCLUDE_DIRS}
  /usr/include/eigen3
)

add_executable(robot_mouth_perception src/RobotMouthPerception.cpp src/main.cpp)
target_link_libraries(robot_mouth_perception
    ${catkin_LIBRARIES}
)
