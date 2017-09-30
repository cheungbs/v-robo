import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg      import String
from math              import radians



class actuator_engine(object):

    def __init__(self, config):
        super(actuator_engine, self).__init__()

        print("start actuator engine loading ...")

        self.config              = config

        self.cmd_vel             = rospy.Publisher('cmd_vel_mux/input/navi', Twist, queue_size=10)

        self.turn_oris           = {
            'f' : 0,
            'fl': 0,
            'lf': 0,
            'l' : 0,
            'lb': 0,
            'bl': 0,
            'b' : 0,
            'br': 1,
            'rb': 1,
            'r' : 1,
            'rf': 1,
            'fr': 1
        }
        self.turn_angles         = {
            'f' : 0.0,
            'fl': 30.0,
            'fr': 30.0,
            'lf': 60.0,
            'rf': 60.0,
            'l' : 90.0,
            'r' : 90.0,
            'lb': 120.0,
            'rb': 120.0,
            'bl': 150.0,
            'br': 150.0,
            'b' : 180.0
        }

        self.turn_fps_rate       = 10
        self.turn_loop_rate      = rospy.Rate(self.turn_fps_rate)
        self.turn_velocity       = 36.0
        self.turn_velocity_coef  = 0.8

        self.turn_fps_rate_dict = {
            '+++': 60,
            '++' : 30,
            '+'  : 15,
            '0'  : 10,
            '-'  : 5,
            '--' : 3,
            '---': 1
        }
        self.turn_loop_rate_dict = {
            '+++': rospy.Rate(self.turn_fps_rate_dict['+++']),
            '++' : rospy.Rate(self.turn_fps_rate_dict['++']),
            '+'  : rospy.Rate(self.turn_fps_rate_dict['+']),
            '0'  : rospy.Rate(self.turn_fps_rate_dict['0']),
            '-'  : rospy.Rate(self.turn_fps_rate_dict['-']),
            '--' : rospy.Rate(self.turn_fps_rate_dict['--']),
            '---': rospy.Rate(self.turn_fps_rate_dict['---'])
        }

        self.forward_loop_rate = rospy.Rate(10)
        self.forwar_velocity   = 0.16

        self.forward_dist_N_loop_dict = {
            'gfff' : 64,
            'gff'  : 32,
            'gf'   : 16,
            'g'    : 8,
            'gn'   : 4,
            'gnn'  : 2,
            'gnnn' : 1
        }

        self.move_cmd            = Twist()
        self.move_cmd.linear.x   = self.forwar_velocity

        self.right_cmd           = Twist()
        self.right_cmd.linear.x  = 0
        self.right_cmd.angular.z = -radians(self.turn_velocity)

        self.left_cmd            = Twist()
        self.left_cmd.linear.x   = 0
        self.left_cmd.angular.z  = radians(self.turn_velocity)

        self.loc_cmd             = Twist()
        self.loc_cmd.linear.x    = 0
        self.loc_cmd.angular.z   = -radians(self.turn_velocity)

        print("complete actuator engine loading")

    def shut_down(self):
        self.cmd_vel.publish(Twist())
        rospy.sleep(1)

    def a_turn_left(self):
        self.cmd_vel.publish(self.left_cmd)

    def a_turn_right(self):
        self.cmd_vel.publish(self.right_cmd)

    def a_forward(self):
        self.cmd_vel.publish(self.move_cmd)

    def get_turn_loop_rate(self, speed='0', angle=360.0):

        loop_rate = self.turn_loop_rate_dict.get(speed, self.turn_loop_rate)
        fps_rate  = self.turn_fps_rate_dict.get(speed, self.turn_fps_rate)

        w_theta   = self.turn_velocity * self.turn_velocity_coef
        delta_t   = 1.0 / fps_rate
        t_360     = angle / w_theta
        N_loop    = int(t_360 / delta_t + 5.5)

        return loop_rate, N_loop

    def get_ori_turn_loop_rate(self, ori='f'):

        loop_rate = self.turn_loop_rate
        ori_dir = self.turn_oris.get(ori, 0)
        if ori_dir == 0:
            action = self.a_turn_left
        else:
            action = self.a_turn_right

        angle   = self.turn_angles.get(ori, 0)
        w_theta = self.turn_velocity * self.turn_velocity_coef
        delta_t = 1.0 / self.turn_fps_rate
        t_angle = angle / w_theta
        N_loop  = int(t_angle / delta_t)

        return loop_rate, action, N_loop, ori_dir

    def get_forward_loop_rate(self, dist='g'):
        loop_rate = self.forward_loop_rate
        N_loop    = self.forward_dist_N_loop_dict.get(dist, 0)

        return loop_rate, N_loop

    def forward_to(self, dist='g'):
        if dist is None:
            dist = 'g'

        loop_rate, N_loop = self.get_forward_loop_rate(dist)
        for i in range(N_loop):
            self.a_forward()
            loop_rate.sleep()

        return True, True # ret, pos_changed, angle_changed

    def turn_to(self, angle=None):
        if angle is None:
            return

        loop_rate, action, N_loop, _ = self.get_ori_turn_loop_rate(angle)

        for _ in range(N_loop):
            action()
            loop_rate.sleep()

        return True, True # ret, angle_changed
    
    def forward_steps(self, loop):
        print("The process of go forward with loop: " + str(loop))

        ret         = True
        pos_changed = True
        for _ in range(loop):
          self.a_forward()
          self.turn_loop_rate.sleep()

        return ret, pos_changed

    def turn_left_steps(self, loop):
        ret         = True
        ori_changed = True

        for _ in range(loop):
            self.a_turn_left()
            self.turn_loop_rate.sleep()

        return ret, ori_changed

    def turn_right_steps(self, loop):
        ret         = True
        ori_changed = True

        for _ in range(loop):
            self.a_turn_right()
            self.turn_loop_rate.sleep()

        return ret, ori_changed

    def turn_steps(self, ori, loop):
        print("The process of turning with ori: " + ori + " loop: " + str(loop))

        ret         = True
        ori_changed = True
        if ori == 'left':
          self.turn_left_steps(loop)
        elif ori == 'right':
          self.turn_right_steps(loop)

        return ret, ori_changed

    def loc_step(self):
        ret         = True
        ori_changed = True
        self.cmd_vel.publish(self.loc_cmd)

        return ret, ori_changed