#!/usr/bin/env python

import cv2
import rospy
from std_msgs.msg import String

def commander():
  rospy.init_node('talker', anonymous=True)

  pub = rospy.Publisher('chatter', String, queue_size=10)

  rospy.loginfo("$ Command Panel $")

  while not rospy.is_shutdown():
    char_in = raw_input()

    cmd_str = "Command: %s" % char_in
    rospy.loginfo(cmd_str)

    pub.publish(char_in)


if __name__ == '__main__':

  try:
    commander()
  except rospy.ROSInterruptException:
    pass
