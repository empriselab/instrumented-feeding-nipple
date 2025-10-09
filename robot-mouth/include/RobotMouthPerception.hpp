#ifndef MOUTHPERCEPTION_HPP_
#define MOUTHPERCEPTION_HPP_

#include <ros/ros.h>
#include <std_msgs/Float64.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <fstream>
#include <iostream>

#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

namespace robot_mouth{

class RobotMouthPerception
{

public:

	RobotMouthPerception(ros::NodeHandle nodeHandle);
	void run(); 

private:

	void publishRobotMouthMarker(const std_msgs::Float64::ConstPtr& msg);

	ros::NodeHandle mNodeHandle;

	ros::Subscriber mMouthCmdSubscriber;
	ros::Publisher mMouthModelArrayPublisher;

}; // namespace robot_mouth

}

#endif