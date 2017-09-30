import time
import cv2

def get_camera(id):
  cap = cv2.VideoCapture(id)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, 432)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
  
  return cap

def sense_image(cap, t_pre, max_delta_t, delta_t):
  d_t = min(time.time() - t_pre, max_delta_t)
  while d_t >= delta_t:
    cap.grab()
    d_t -= delta_t
  ret, frame = cap.read()

  return ret, frame, time.time()

def sense_2_images(cap1, cap2, t_pre, max_delta_t, delta_t):
  d_t = min(time.time() - t_pre, max_delta_t)
  while d_t >= delta_t:
    cap1.grab()
    cap2.grab()
    d_t -= delta_t
  cap1.grab()
  cap2.grab()

  ret1, frame1 = cap1.retrieve()
  ret2, frame2 = cap2.retrieve()

  return ret1 and ret2, frame1, frame2, time.time()

max_delta_t = 0.165
delta_t = 0.033

t_pre = time.time()
cap0 = get_camera(0)
cap1 = get_camera(1) 

for _ in range(1000):
  ret, frame0, frame1, t_pre = sense_2_images(cap0, cap1, t_pre, max_delta_t, delta_t)
  
  cv2.imshow("frame0", frame0)
  cv2.imshow("frame1", frame1)
  cv2.waitKey(100)
