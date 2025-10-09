#!/usr/bin/env python

import rospy
from std_msgs.msg import Float64MultiArray
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
import matplotlib.pyplot as plt

# Initialize colormap (e.g., Reds, Blues, etc.)
cmap = plt.cm.Reds

def visualize_callback(msg):
    marker_array = MarkerArray()

    # 4 sensors, placed along x-axis for visualization
    sensor_positions = [
        (0.0, 0.0, 0.0),
        (0.05, 0.0, 0.0),
        (0.10, 0.0, 0.0),
        (0.15, 0.0, 0.0),
    ]

    for i, value in enumerate(msg.data):
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.id = i

        # normalize value for color/size scaling
        normalized = np.clip(abs(value) / 1023.0, 0.0, 1.0)
        color = cmap(normalized)

        # color and transparency
        marker.color.a = 1.0
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]

        # size depends on reading
        marker.scale.x = 0.01 + 0.02 * normalized
        marker.scale.y = 0.01 + 0.02 * normalized
        marker.scale.z = 0.01 + 0.02 * normalized

        # position of this sensor
        marker.pose.position.x = sensor_positions[i][0]
        marker.pose.position.y = sensor_positions[i][1]
        marker.pose.position.z = sensor_positions[i][2]

        marker_array.markers.append(marker)

    pub.publish(marker_array)


if __name__ == '__main__':
    rospy.init_node('mouth_visualizer')
    pub = rospy.Publisher('mouth_markers', MarkerArray, queue_size=10)
    rospy.Subscriber('/mouth_contact', Float64MultiArray, visualize_callback)

    rospy.loginfo("Mouth visualization node started.")
    rospy.spin()