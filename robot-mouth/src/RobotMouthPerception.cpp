#include <RobotMouthPerception.hpp>

#include <chrono>
using namespace std::chrono;

namespace robot_mouth{

RobotMouthPerception::RobotMouthPerception(ros::NodeHandle nodeHandle): mNodeHandle(nodeHandle)
{
	mMouthCmdSubscriber = mNodeHandle.subscribe("/mouth_cmd", 10, &RobotMouthPerception::publishRobotMouthMarker, this);
	// mSubInfo = mNodeHandle.subscribe(info_topic_name, 10, &FacePerception::cameraInfoCallback, this);
	mMouthModelArrayPublisher = mNodeHandle.advertise<visualization_msgs::MarkerArray>("/mouth_model_detector/marker_array", 1);
}

void RobotMouthPerception::run()
{
	ros::Rate rate(30); // ROS Rate at 30Hz
	std::cout<<"Started Robot Mouth Perception!"<<std::endl;
	while (ros::ok()) 
	{
		ros::spinOnce();
	}
}

void RobotMouthPerception::publishRobotMouthMarker(const std_msgs::Float64::ConstPtr& msg)
{
	std::cout<<"Command Received:"<<msg->data<<std::endl;

	double closing_angle = (0.35/1.5)*(1.5 - msg->data); //figure out closing angle from command recieved

	Eigen::Isometry3d mouth_base_frame = Eigen::Isometry3d::Identity();

	mouth_base_frame.translation()[0] = 0.685;
	mouth_base_frame.translation()[1] = -0.31;
	mouth_base_frame.translation()[2] = 0.52;

  	mouth_base_frame.linear() = mouth_base_frame.linear() * Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitZ());

  	Eigen::Isometry3d goal_mouth_frame = Eigen::Isometry3d::Identity();
  	goal_mouth_frame.translation()[0] = 0.08;
	goal_mouth_frame.translation()[2] = 0.03883;

	Eigen::Isometry3d upper_mouth_base_frame = mouth_base_frame; // upper_mouth is aligned same as base
	
	Eigen::Isometry3d lower_mouth_base_frame = mouth_base_frame;
	lower_mouth_base_frame.linear() = lower_mouth_base_frame.linear() * Eigen::AngleAxisd(-closing_angle, Eigen::Vector3d::UnitY());

	Eigen::Isometry3d goal_base_frame;
	Eigen::Isometry3d upper_mouth_goal_frame;
	Eigen::Isometry3d lower_mouth_goal_frame;

	goal_base_frame = mouth_base_frame * goal_mouth_frame;
	upper_mouth_goal_frame = goal_base_frame.inverse() * upper_mouth_base_frame;
	lower_mouth_goal_frame = goal_base_frame.inverse() * lower_mouth_base_frame;

	Eigen::Vector3d upper_mouth_euler = upper_mouth_goal_frame.linear().eulerAngles(0, 1, 2);
	double upper_mouth_x = upper_mouth_goal_frame.translation()[0];
	double upper_mouth_y = upper_mouth_goal_frame.translation()[1];
	double upper_mouth_z = upper_mouth_goal_frame.translation()[2];
	double upper_mouth_roll = upper_mouth_euler[0];
	double upper_mouth_pitch = upper_mouth_euler[1]; 
	double upper_mouth_yaw = upper_mouth_euler[2];

	Eigen::Vector3d lower_mouth_euler = lower_mouth_goal_frame.linear().eulerAngles(0, 1, 2);
	double lower_mouth_x = lower_mouth_goal_frame.translation()[0];
	double lower_mouth_y = lower_mouth_goal_frame.translation()[1];
	double lower_mouth_z = lower_mouth_goal_frame.translation()[2];
	double lower_mouth_roll = lower_mouth_euler[0]; 
	double lower_mouth_pitch = lower_mouth_euler[1]; 
	double lower_mouth_yaw = lower_mouth_euler[2];

	std::string urdf_path = "/home/emprise/bite_transfer_ws/src/pr_assets/data/objects/robot_mouth.urdf";

	std::ofstream urdf_file;
	urdf_file.open(urdf_path, std::ios::out);
	std::cout<<"Writing to file "<<urdf_path<<std::endl;
	urdf_file<<"<robot name=\"mouth\">\n  <link name=\"mouth\">\n";

	// upper mouth model
	urdf_file<<"    <visual>\n";
	urdf_file<<"      <origin xyz=\""+std::to_string(upper_mouth_x)+" "+std::to_string(upper_mouth_y)+" "+std::to_string(upper_mouth_z)+
					"\" rpy=\""+std::to_string(upper_mouth_roll)+" "+std::to_string(upper_mouth_pitch)+" "+std::to_string(upper_mouth_yaw)+"\"/>\n";
	urdf_file<<"      <geometry>\n";
	urdf_file<<"        <mesh filename=\"package://pr_assets/data/objects/mouth_model_upper.obj\" scale=\" 1.0 1.0 1.0\"/>\n";
	urdf_file<<"      </geometry>\n";
	urdf_file<<"    </visual>\n";
	urdf_file<<"    <collision>\n";
	urdf_file<<"      <origin xyz=\""+std::to_string(upper_mouth_x)+" "+std::to_string(upper_mouth_y)+" "+std::to_string(upper_mouth_z)+
					"\" rpy=\""+std::to_string(upper_mouth_roll)+" "+std::to_string(upper_mouth_pitch)+" "+std::to_string(upper_mouth_yaw)+"\"/>\n";
	urdf_file<<"      <geometry>\n";
	urdf_file<<"        <mesh filename=\"package://pr_assets/data/objects/mouth_model_upper.obj\" scale=\" 1.0 1.0 1.0\"/>\n";
	urdf_file<<"      </geometry>\n";
	urdf_file<<"    </collision>\n";

	// lower mouth model
	urdf_file<<"    <visual>\n";
	urdf_file<<"      <origin xyz=\""+std::to_string(lower_mouth_x)+" "+std::to_string(lower_mouth_y)+" "+std::to_string(lower_mouth_z)+
				  		"\" rpy=\""+std::to_string(lower_mouth_roll)+" "+std::to_string(lower_mouth_pitch)+" "+std::to_string(lower_mouth_yaw)+"\"/>\n";
	urdf_file<<"      <geometry>\n";
	urdf_file<<"        <mesh filename=\"package://pr_assets/data/objects/mouth_model_lower.obj\" scale=\" 1.0 1.0 1.0\"/>\n";
	urdf_file<<"      </geometry>\n";
	urdf_file<<"    </visual>\n";
	urdf_file<<"    <collision>\n";
	urdf_file<<"    <origin xyz=\""+std::to_string(lower_mouth_x)+" "+std::to_string(lower_mouth_y)+" "+std::to_string(lower_mouth_z)+
					"\" rpy=\""+std::to_string(lower_mouth_roll)+" "+std::to_string(lower_mouth_pitch)+" "+std::to_string(lower_mouth_yaw)+"\"/>\n";
	urdf_file<<"      <geometry>\n";
	urdf_file<<"        <mesh filename=\"package://pr_assets/data/objects/mouth_model_lower.obj\" scale=\" 1.0 1.0 1.0\"/>\n";
	urdf_file<<"      </geometry>\n";
	urdf_file<<"    </collision>\n";
	
	urdf_file<<"  </link>\n</robot>\n";
	urdf_file.close();

	visualization_msgs::Marker robot_mouth_marker;
	robot_mouth_marker.pose.position.x = goal_base_frame.translation()[0];
	robot_mouth_marker.pose.position.y = goal_base_frame.translation()[1];
	robot_mouth_marker.pose.position.z = goal_base_frame.translation()[2];

	Eigen::Quaterniond quat(goal_base_frame.linear());
	robot_mouth_marker.pose.orientation.x = quat.x();
	robot_mouth_marker.pose.orientation.y = quat.y();
	robot_mouth_marker.pose.orientation.z = quat.z();
	robot_mouth_marker.pose.orientation.w = quat.w();

	// Additional Visualization
	robot_mouth_marker.scale.x = 1.0;
	robot_mouth_marker.scale.y = 1.0;
	robot_mouth_marker.scale.z = 1.0;

	robot_mouth_marker.type = visualization_msgs::Marker::MESH_RESOURCE;
	robot_mouth_marker.mesh_resource = "package://pr_assets/data/objects/robot_mouth.urdf";
	robot_mouth_marker.mesh_use_embedded_materials = true;
	robot_mouth_marker.action = visualization_msgs::Marker::ADD;
	robot_mouth_marker.id = 1;
	robot_mouth_marker.header.stamp = ros::Time();
	robot_mouth_marker.color.a = 1.0;
	robot_mouth_marker.color.r = 0.5;
	robot_mouth_marker.color.g = 0.5;
	robot_mouth_marker.color.b = 0.5;

	// mouth status display

	bool mIsMouthOpen = true;
	if (mIsMouthOpen == true) 
	{
		robot_mouth_marker.text = "{\"db_key\": \"robot_mouth\", \"mouth-status\": \"open\"}";
		robot_mouth_marker.ns = "robot_mouth";
		std::cout << "OPEN" << std::endl;
	} 
	else 
	{
		robot_mouth_marker.text = "{\"db_key\": \"robot_mouth\", \"mouth-status\": \"closed\"}";
		robot_mouth_marker.ns = "robot_mouth";
		std::cout << "CLOSED" << std::endl;
	}

	robot_mouth_marker.header.frame_id = "base_link";

	visualization_msgs::MarkerArray marker_arr;
	marker_arr.markers.push_back(robot_mouth_marker);
	mMouthModelArrayPublisher.publish(marker_arr);

	std::cout<<"In publishing robot_mouth marker!"<<std::endl;
}

}