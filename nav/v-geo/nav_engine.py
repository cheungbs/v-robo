import rospy
from std_msgs.msg import String

import numpy as np

import geo_map
import geo_engine
import actuator_engine
import vision_engine

class nav_engine(object):
  
  def __init__(self, config, map_file=None, vision_engine_p=None, geo_engine_p=None, actuator_engine_p=None):
    super(nav_engine, self).__init__()

    print("start nav engine loading ...")

    self.config          = config

    if geo_engine_p is None:
      self.geo_engine = geo_engine.geo_engine(config)
    else:
      self.geo_engine = geo_engine_p

    if actuator_engine_p is None:  
      self.actuator_engine = actuator_engine.actuator_engine(config)
    else:
      self.actuator_engine = actuator_engine_p

    if vision_engine_p is None:
      self.vision_engine = vision_engine.vision_engine(config)
    else:
      self.vision_engine = vision_engine_p

    if map_file is None:
      self.ori_map = geo_map.ori_map()
    else:
      self.ori_map = np.load(map_file)

    self.control = rospy.Publisher('control', String, queue_size=100)

    self.pre_pos     = None
    self.pre_ori     = None
    self.pos         = None
    self.ori         = None
    self.pos_v       = None
    self.ori_v       = None
    self.pos_confi   = 0.0
    self.ori_confi   = 0.0
    self.pos_changed = True
    self.ori_changed = True

    self.stop_sign   = False

    print("complete nav engine loading")

  def near_ori(self, curr_ori, goal_ori, d_ori):
    bNear = False
    dd_ori = np.abs(goal_ori - curr_ori)
    if dd_ori > 5.1:
      dd_ori = 12 - dd_ori
    if dd_ori <= d_ori:
      bNear = True

    return bNear

  def set_stop_sign(self):
    if self.stop_sign == False:
      self.stop_sign = True

  def check_stop_sign(self):
    ret = False
    if self.stop_sign == True:
      self.stop_sign = False
      ret = True

    return ret

  def calc_loc(self, pos_v):
    idx = np.argmax(pos_v)
    return idx, pos_v[idx]

  def calc_ori(self, ori_v):
    idx = np.argmax(ori_v)
    return idx, ori_v[idx]

  def plan_next_forward_ori_dist(self, curr_loc, goal_loc):
    ori = self.ori_map[goal_loc, curr_loc]
    print('the oritation is: ' + str(ori))
    return self.ori_map[goal_loc, curr_loc], 10

  def plan_next_turn_ori_angle(self, curr_ori, goal_ori):
    if goal_ori > curr_ori:
      if goal_ori - curr_ori >= 5.5:
        ori = 'left'
      else:
        ori = 'right'
    else:
      if curr_ori - goal_ori >= 5.5:
        ori = 'right'
      else:
        ori = 'left'

    return 'left', 10

  def locate_me(self):
    print("The process of locating robot's location")

    ret = True
    if not self.vision_engine.check_sense():
      print("ERROR: No camera for vision!")
      return False

    for i in range(6):
      print('preparing for locating: ' + str(i))
      ret, _ = self.vision_engine.sense_image()
      if not ret:
        print("Camera frame capture error.")
        return False

      ret, self.ori_changed = self.actuator_engine.loc_step()
      self.loop_rate.sleep()

    self.geo_engine.reset_loc()
    for i in range(36):
      ret, vgg16_feat = self.vision_engine.sense_vgg16()
      if not ret:
        print("ERROR: sensing vgg16 feature error.")
        return False

      pos_v = self.geo_engine.loc(vgg16_feat)
      pos_t, pos_confi_t = self.calc_loc(pos_v)
      self.control.publish(str(pos_t))
      # print("current position is: " + str(pos_t) + " confidence is: " + str(pos_confi_t))
      ret, self.ori_changed = self.actuator_engine.loc_step()
      self.loop_rate.sleep()
      
    self.pos_v = pos_v 
    self.pos, self.pos_confi = self.calc_loc(self.pos_v)
    self.control.publish(str(self.pos))
    print("--------------------------------------------------------------------------------")
    print("current position is: " + str(self.pos) + " confidence is: " + str(self.pos_confi))
    print("--------------------------------------------------------------------------------")

    self.pos_changed = False
    self.ori_changed = True

    return ret

  def facing_ori_v(self):
    print("The process of determining robot's current facing oritation")
    
    ret = True
    if self.pos == None or self.pos_changed == True:
      ret = self.locate_me()
    if not ret:
      print('There are errors for locating robot ...')
      return False

    if not self.vision_engine.check_sense():
      print("ERROR: No camera for vision!")
      return False

    for i in range(6):
      ret, _ = self.vision_engine.sense_image()
      if not ret:
        print("Camera frame capture error.")
        return False

    ret, vgg16_feat = self.vision_engine.sense_vgg16()
    if not ret:
      print("ERROR: sensing vgg16 feature error.")
      return False

    ori_v = self.geo_engine.ori_v(vgg16_feat, self.pos_v)
    self.ori_v = ori_v
    self.ori, self.ori_confi = self.calc_ori(ori_v)


    self.ori_changed = False
    return ret

  def facing_ori(self):
    print("The process of determining robot's current facing oritation")
    
    ret = True
    if self.pos == None or self.pos_changed == True:
      ret = self.locate_me()
    if not ret:
      print('There are errors for locating the robot ...')
      return False

    if not self.vision_engine.check_sense():
      print("ERROR: No camera for vision!")
      return False

    for i in range(6):
      ret, _ = self.vision_engine.sense_image()
      if not ret:
        print("Camera frame capture error.")
        return False

    ret, vgg16_feat = self.vision_engine.sense_vgg16()
    if not ret:
      print("ERROR: sensing vgg16 feature error.")
      return False
    ori_v = self.geo_engine.ori(vgg16_feat, self.pos)
    self.ori_v = ori_v
    self.ori, self.ori_confi = self.calc_ori(self.ori_v)

    print("current orientation is: " + str(self.ori) + " confidence is: " + str(self.ori_confi))

    self.ori_changed = False
    return ret
  
  def face_to(self, goal_ori):
    print('The process of rotatint to face to: ' + str(goal_ori))

    ret = True
    if self.ori_changed == True:
      ret = self.facing_ori()
    if not ret:
      print('There are errors for determine the orientation.')
      return False

    loop = 0
    while ret and not self.near_ori(self.ori, goal_ori, loop//30):
      if loop//30 > 3:
        print('Cannot find the near dirction to go to the goal location.')
        return False
      ori, angle = self.plan_next_turn_ori_angle(self.ori, goal_ori)
      if ori is None:
        ret = False
        break

      ret, self.ori_changed = self.actuator_engine.turn_steps(ori, angle)
      if not ret:
        print('There are errors for robot turning ...')
        break

      if not self.facing_ori():
        ret = False
        break

      if self.check_stop_sign():
        print('stop face_to process')
        ret = False
        break

      loop = loop + 1

    return ret

  def nav_to(self, goal_loc):
    print("The process of navigate to location: " + str(goal_loc))

    ret = True
    if self.pos is None or self.pos_changed==True:
      ret = self.locate_me()
    if not ret:
      print('There are errors for locating robot ...')
      return False

    while ret and not self.pos == goal_loc:
      goal_ori, dist = self.plan_next_forward_ori_dist(self.pos, goal_loc)
      print('************************************************************')
      print('* current position is        : ' + str(self.pos))
      print('* the goal position is       : ' + str(goal_loc))
      print('* the forward orientation is : ' + str(goal_ori))
      print('************************************************************')
      if goal_ori is None:
        ret = False
        break

      if not self.face_to(goal_ori):
        ret = False
        break

      ret, self.pos_changed = self.actuator_engine.forward_steps(dist)
      if not ret:
        print('There are errors for robot forwarding ... ')
        break

      if not self.locate_me():
        ret = False
        break
      
      if self.check_stop_sign():
        print("Stop the nav_to process.")
        ret = False
        break

    return ret

  def nav_and_face_to(self, goal_loc, goal_ori):
    print("The process of navigating to: " + str(goal_loc) + " and facing to: " + str(goal_ori))

    if not self.nav_to(goal_loc):
      return False
    if not self.face_to(goal_ori):
      return False

  def excute_cmds(self, cmd_str, cmd_param=None):
    if cmd_str == 'nav':
      if cmd_param.isdigit():
        self.nav_to(int(cmd_param))
      else:
        print('Goal location id must be integer from 0 to 24.') 
    elif cmd_str == 'face':
      if cmd_param.isdigit():
        self.face_to(int(cmd_param))
      else:
        print('Goal orientation id must be integer from 0 to 11')
    elif cmd_str in ['gf', 'gff', 'gfff', 'gn', 'gnn', 'gnnn']:
      self.actuator_engine.forward_to(cmd_str)
    elif cmd_str in ['fl', 'fr', 'lf', 'rf', 'l', 'r', 'lb', 'rb', 'bl', 'br', 'b']:
      self.actuator_engine.turn_to(cmd_str)
    elif cmd_str in ['X-Button', 'Arrow-Button-Left']:
      self.actuator_engine.a_turn_left()
    elif cmd_str in ['B-Button', 'Arrow-Button-Right']:
      self.actuator_engine.a_turn_right()
    elif cmd_str in ['A-Button', 'Arrow-Button-Down']:
      self.actuator_engine.a_turn_left()
    elif cmd_str in ['LT-Button', 'RT-Button']:
      self.actuator_engine.a_forward()
    elif cmd_str == 'loc':
      self.locate_me()
    elif cmd_str == 'ori':
      self.facing_ori()
    elif cmd_str == 'stop':
      self.set_stop_sign()