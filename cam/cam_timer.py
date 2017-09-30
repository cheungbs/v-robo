import threading
import cv2
import time

cap = cv2.VideoCapture(0)

def sayhello():
  global t
  global cap

  s = time.time()
  ret = cap.grab()
  # ret, frame = cap.read()
  dt = time.time() - s
  print(dt)
  print(ret)
  # print(type(frame))
  t = threading.Timer(0.01, sayhello)
  t.start()

t = threading.Timer(0.01, sayhello)
t.start()
