#!/usr/bin/env python

import rospy
import numpy as np

from mouth_api import MouthClient

# Simulate the mouth chewing by opening and closing
if __name__ == "__main__":
    CMD_OPEN = 1.0
    CMD_CLOSE = 0.2
    mouth = MouthClient(rate=100)

    while not rospy.is_shutdown():
        cmd = np.linspace(CMD_CLOSE, CMD_OPEN, num=50)
        for mouth_pos in reversed(cmd):
            mouth.move(mouth_pos)
        for mouth_pos in cmd[1:-1]:
            mouth.move(mouth_pos)
