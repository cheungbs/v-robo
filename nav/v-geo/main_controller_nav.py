#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
from math              import radians
from std_msgs.msg      import String
import vision_engine
import actuator_engine
import nav_engine
import sys_setup
import data_engine

global cmd_stat
global vid_stat
global b_stop

rospy.init_node('listener', anonymous=False)  

cmd_stat        = 'none'
vid_stat        = 'none'
b_stop          = False

config          = sys_setup.config()
actuator_engine = actuator_engine.actuator_engine(config)
vision_engine   = vision_engine.vision_engine(config)

data_engine     = data_engine.data_engine(config, vision_engine_p=vision_engine, actuator_engine_p=actuator_engine)
nav_engine      = nav_engine.nav_engine(config, vision_engine_p=vision_engine, actuator_engine_p=actuator_engine)

def command_processing(cmd_str, cmd_param=None):
  global cmd_stat
  global vid_stat
  global b_stop

  if cmd_str is None:
    return

  if cmd_str == 'exit':
    cmd_stat = 'exit'
  elif cmd_str == 'train':
    cmd_stat = 'train'
    data_engine.eposode_start()
  elif cmd_str == 'untrain':
    cmd_stat = 'none'
    data_engine.eposode_end()
  elif cmd_str == 'START-Button':
    if not cmd_stat=='video' and vid_stat == 'none':
      cmd_stat = 'video'
      vid_stat = 'start'
  elif cmd_str == 'BACK-Button':
    if cmd_stat=='video' and vid_stat=='recording':
      b_stop = True
  elif cmd_stat == 'train':
    data_engine.excute_cmds(cmd_str, cmd_param)
  else:
    nav_engine.excute_cmds(cmd_str, cmd_param)

def callback(data):
  cmd_str = data.data

  rospy.loginfo(rospy.get_caller_id() + " command to: %s", data.data)

  cmds    = cmd_str.split()
  cmd_len = len(cmds)

  cmd_param = None

  if cmd_len == 1:
    cmd_str   = cmds[0]
  elif cmd_len >= 2:
    cmd_str   = cmds[0]
    cmd_param = cmds[1]
  else:
    cmd_str   = None

  command_processing(cmd_str, cmd_param)

def shutdown():
  rospy.loginfo("Stop Listener!")
  actuator_engine.shut_down()

rospy.on_shutdown(shutdown)
rospy.Subscriber('chatter', String, callback)

rospy.loginfo(">>> Main Controller center: waiting commands <<<")
r = rospy.Rate(10)
 
while not rospy.is_shutdown():
  if cmd_stat == 'video':
    if vid_stat == 'start':
      data_engine.eposode_start_vid()
      vid_stat = 'recording'
      print('start video recording >>>>>>')
    elif vid_stat == 'recording':
      if b_stop:
        vid_stat = "stop"
        b_stop   = False
      data_engine.record_one_frame()
    elif vid_stat == 'stop':
      data_engine.eposode_end_vid()
      cmd_stat = 'none'
      vid_stat = 'none'
      print('end video recording <<<<<<')
  elif cmd_stat == 'exit':
    print('Exit the system.')
    break

  r.sleep() 
