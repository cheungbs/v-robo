import cv2

cap = cv2.VideoCapture(0)

# 0 CAP_PROP_POS_MSEC
# 1 CAP_PROP_POS_FRAMES
# 2 CAP_PROP_AVI_RATIO
# 3 CAP_PROP_FRAME_WIDTH
# 4 CAP_PROP_FRAME_HEIGHT
# 5 CAP_PROP_FPS
# 6 CAP_PROP_FOURCC
# 7 CAP_PROP_FRAME_COUNT
# 8 CAP_PROP_FORMAT
# 9 CAP_PROP_MODE
# 10 CAP_PROP_BRIGHTNESS
# 11 CAP_PROP_CONTRAST
# 12 CAP_PROP_SATUATION
# 13 CAP_PROP_HUE
# 14 CAP_PROP_GAIN
# 15 CAP_PROP_EXPOSURE
# 16 CAP_PROP_CONVERT_RGB
# 17 CAP_PROP_WHITE_BALANCE
# 18 CAP_PROP_RECTIFICATION

for i in range(100):
  print(str(i) + " : " + str(cap.get(i)))