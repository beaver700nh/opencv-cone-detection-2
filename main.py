import cv2 as cv
import numpy as np

HSV_LOWER = np.array([ 11,   0, 150])
HSV_UPPER = np.array([ 39, 255, 255])

KERNEL_A = np.ones(( 5,  5), np.uint8)
KERNEL_B = np.ones((10, 10), np.uint8)

cap = cv.VideoCapture(0)

while True:
  _, img = cap.read()
  cv.imshow("Image #1", img)

  hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
  mask = cv.inRange(hsv, HSV_LOWER, HSV_UPPER)
  img = cv.bitwise_and(img, img, mask=mask)
  cv.imshow("Image #2", img)

  _, _, img = cv.split(img)
  _, img = cv.threshold(img, 100, 255, cv.THRESH_BINARY)
  cv.imshow("Image #3", img)

  img = cv.morphologyEx(img, cv.MORPH_OPEN, KERNEL_A)
  img = cv.morphologyEx(img, cv.MORPH_OPEN, KERNEL_B)
  cv.imshow("Image #4", img)

  if (key := cv.waitKey(0)) & 0xFF == ord("q"):
    break

cap.release()
cv.destroyAllWindows()
