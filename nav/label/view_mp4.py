import cv2

fname = './tmp/20170930-065954_1.mp4'
cap = cv2.VideoCapture(fname)

ret, frame = cap.read()

while ret:
  cv2.imshow('frame', frame)
  cv2.waitKey(100)
  ret, frame = cap.read()

print('mp4 video file viewing completed.')
