#!/usr/bin/env python
import rospy
import numpy as np
import matplotlib.pyplot as plt
from std_msgs.msg import String
from std_msgs.msg import Int16
from geometry_msgs.msg import WrenchStamped
from pyrcareworld.utils.skeleton_visualizer import SkeletonVisualizer

class ForceVisualizer:

    def __init__(self):
        self.ft_sub = rospy.Subscriber("/ft_sensor/netft_data", WrenchStamped, self.ft_callback)
        
        self.skeleton_visualizer = SkeletonVisualizer()
        self.skeleton_visualizer.show()

        self.current_body_id = 0
        self.body_ids = np.array(["upper left arm","left forearm", "left thigh", "abdomen",
                                  "upper right arm","right forearm", "right thigh"])
        
        input("Begin Phase 1 of User Study. Press Enter to Continue")
        
    def ft_callback(self, data):
        return
    
    def change_body_part(self):
        self.current_body_id += 1

    def get_body_part(self):
        if(self.current_body_id < 0):
            print("Not yet!")
            return
        return self.body_ids[self.current_body_id]
    
    def get_body_id(self):
        return self.current_body_id

if __name__ == '__main__':

    viz = ForceVisualizer()
    rospy.init_node('force_visualizer_node')
    try:
        while not rospy.is_shutdown():
            body_part = viz.get_body_part()
            A = input("Press Enter once you are done calibrating forces for the " + body_part)
            if(viz.get_body_id() == 6):
                print("Phase 1 done. Shutting down")
                plt.close('all')
                rospy.signal_shutdown("Phase 1 done. Shutting down")
            else:
                viz.change_body_part()
                print("Now begin calibration for the " + viz.get_body_part())

    except KeyboardInterrupt:
        print("Shutting down")