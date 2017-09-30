import cv2

for thing in dir(cv2):
  if thing.find("CAP_") > -1:
    print(thing)
