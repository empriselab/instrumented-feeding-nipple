#include <RobotMouthPerception.hpp>

int main(int argc, char** argv)
{
	ros::init(argc, argv, "robot_mouth");
	ros::NodeHandle n;
	robot_mouth::RobotMouthPerception robotMouthPerception(n);
	robotMouthPerception.run(); 
}