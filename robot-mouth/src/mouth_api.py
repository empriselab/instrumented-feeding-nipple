#!/usr/bin/env python

import rospy
from std_msgs.msg import Float64

CMD_TOPIC = "mouth_cmd"  # The topic to publish the mouth cmd to


class MouthClient:
    def __init__(self, rate=1, mouth_range=[0.2, 1.0], cmd_topic=CMD_TOPIC):
        rospy.init_node("mouth_client", anonymous=True)

        self.rate = rospy.Rate(rate)
        self.cmd_pub = rospy.Publisher(cmd_topic, Float64, queue_size=1)
        self.range = mouth_range

    def move(self, cmd):
        msg = Float64()
        msg.data = cmd
        self.cmd_pub.publish(msg)
        self.rate.sleep()

    def open(self):
        self.move(self.range[1])

    def close(self):
        self.move(self.range[0])


# Test script to toggle back and forth between opening and closing the mouth
if __name__ == "__main__":
    mouth = MouthClient()
    opened = False

    while not rospy.is_shutdown():
        mouth.close() if opened else mouth.open()
        opened = not opened
