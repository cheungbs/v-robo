import rospy
from std_msgs.msg    import String
from sensor_msgs.msg import Joy

global pubcmd
global cmd_str
pubcmd  = None
cmd_str = None

def joycallback(joydata):
    global pubcmd
    global cmd_str

    # left <- o -> right :: 1.0 <- 0 -> -1.0
    if joydata.axes[0] > 0:
        cmd_str = 'Left-Left'
        rospy.loginfo(cmd_str)
    elif joydata.axes[0] < 0:
        cmd_str = 'Left-Right'
        rospy.loginfo(cmd_str)

    # up <- o -> down :: 1.0 <- 0 -> -1.0
    if joydata.axes[1] > 0:
        cmd_str = 'Left-Up'
        rospy.loginfo(cmd_str)
    elif joydata.axes[1] < 0:
        cmd_str = 'Left-Down'
        rospy.loginfo(cmd_str)

    # 1.0 -> -1.0 :: LT button
    if joydata.axes[2] < 0:
        cmd_str = 'LT-Button'
        rospy.loginfo(cmd_str)

    # left <- o -> right :: 1.0 <- 0 -> -1.0
    if joydata.axes[3] > 0:
        cmd_str = 'Right-Left'
        rospy.loginfo(cmd_str)
    elif joydata.axes[3] < 0:
        cmd_str = 'Right-Right'
        rospy.loginfo(cmd_str)

    # up <- o -> down :: 1.0 <- 0 -> -1.0
    if joydata.axes[4] > 0:
        cmd_str = 'Right-Up'
        rospy.loginfo(cmd_str)
    elif joydata.axes[4] < 0:
        cmd_str = 'Right-Down'
        rospy.loginfo(cmd_str)

    # 1.0 -> -1.0 :: RT button
    if joydata.axes[5] < 0:
        cmd_str = 'RT-Button'
        rospy.loginfo(cmd_str)

    # left <- o -> right :: 1.0 <- 0 -> -1.0
    if joydata.axes[6] > 0:
        cmd_str = 'Arrow-Button-Left'
        rospy.loginfo(cmd_str)
    elif joydata.axes[6] < 0:
        cmd_str = 'Arrow-Button-Right'
        rospy.loginfo(cmd_str)

    # up <- o -> down :: 1.0 <- 0 -> -1.0
    if joydata.axes[7] > 0:
        cmd_str = 'Arrow-Button-Up'
        rospy.loginfo(cmd_str)
    elif joydata.axes[7] < 0:
        cmd_str = 'Arrow-Button-Down'
        rospy.loginfo(cmd_str)

    if joydata.buttons[0] == 1:
        cmd_str = 'A-Button'
        rospy.loginfo(cmd_str)

    if joydata.buttons[1] == 1:
        cmd_str = 'B-Button'
        rospy.loginfo(cmd_str)

    if joydata.buttons[2] == 1:
        cmd_str = 'X-Button'
        rospy.loginfo(cmd_str)

    if joydata.buttons[3] == 1:
        cmd_str = 'Y-Button'
        rospy.loginfo(cmd_str)

    if joydata.buttons[4] == 1:
        cmd_str = 'LB-Button'
        rospy.loginfo(cmd_str)

    if joydata.buttons[5] == 1:
        cmd_str = 'RB-Button'
        rospy.loginfo(cmd_str)

    if joydata.buttons[6] == 1:
        cmd_str = 'BACK-Button'
        rospy.loginfo(cmd_str)

    if joydata.buttons[7] == 1:
        cmd_str = 'START-Button'
        rospy.loginfo(cmd_str)

    if joydata.buttons[8] == 1:
        cmd_str = 'POWER-Button'
        rospy.loginfo(cmd_str)

    if joydata.buttons[9] == 1:
        cmd_str = 'UNKNOWN-Button-9'
        rospy.loginfo(cmd_str)

    if joydata.buttons[10] == 1:
        cmd_str = 'UNKNOWN-Button-10'
        rospy.loginfo(cmd_str)

    pubcmd.publish(cmd_str)


if __name__ == '__main__':
    rospy.init_node('rosjoy')
    rospy.Subscriber('/joy', Joy, joycallback, queue_size=1)

    pubcmd = rospy.Publisher('chatter', String, queue_size = 1)

    rospy.loginfo("$ Joy Stick Command Panel $")

    rospy.spin()
