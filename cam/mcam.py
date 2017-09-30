import numpy as np
import cv2

cap0 = cv2.VideoCapture(0)
cap0.set(cv2.CAP_PROP_FRAME_WIDTH, 432)
cap0.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

cap1 = cv2.VideoCapture(1)
cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 432)
cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

cap2 = cv2.VideoCapture(2)
cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 432)
cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

for _ in xrange(1000):
  ret0, frame0 = cap0.read()
  ret1, frame1 = cap1.read()
  ret2, frame2 = cap2.read()
  cv2.imshow('frame0', frame0)
  cv2.imshow('frame1', frame1)
  cv2.imshow('frame2', frame2)
  cv2.waitKey(10)
