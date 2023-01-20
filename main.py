import cv2 as cv
import numpy as np
import math

HSV_LOWER = np.array([ 11,   0, 150])
HSV_UPPER = np.array([ 39, 255, 255])

KERNEL_A = np.ones(( 5,  5), np.uint8)
KERNEL_B = np.ones((10, 10), np.uint8)

cap = cv.VideoCapture(0)

def triangleify(contour):
  multiplier = 0.01
  while True:
    epsilon = multiplier * cv.arcLength(contour, True)
    approx = cv.approxPolyDP(contour, epsilon, True)

    if len(approx) == 3:
      return approx

    if len(approx) > 3:
      multiplier += 0.025

def find_lengths(contour):
  return [
    round(
      math.sqrt(
        (contour[n][0][0] - contour[(n + 1) % 3][0][0]) ** 2 +
        (contour[n][0][1] - contour[(n + 1) % 3][0][1]) ** 2
      )
    )
    for n in range(len(contour))
  ]

def find_smallest_idx(iterable):
  idx = 0
  for i in range(len(iterable)):
    if iterable[i] < iterable[idx]:
      idx = i

  return idx

def find_orientation(vertices, top):
  other1 = vertices[(top + 1) % 3][0]
  other2 = vertices[(top + 2) % 3][0]
  center = [(other1[d] + other2[d]) // 2 for d in range(2)]
  point = vertices[top][0]
  angle = math.atan2(point[1] - center[1], point[0] - center[0])
  return 0 - (angle * 180 / math.pi)

while True:
  _, img = cap.read()
  cv.imshow("Image #1", img)

  hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
  mask = cv.inRange(hsv, HSV_LOWER, HSV_UPPER)
  img = cv.bitwise_and(img, img, mask=mask)
  # cv.imshow("Image #2", img)

  _, _, img = cv.split(img)
  _, img = cv.threshold(img, 100, 255, cv.THRESH_BINARY)
  # cv.imshow("Image #3", img)

  img = cv.morphologyEx(img, cv.MORPH_OPEN, KERNEL_A)
  img = cv.morphologyEx(img, cv.MORPH_OPEN, KERNEL_B)
  # cv.imshow("Image #4", img)

  contours, _ = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

  if len(contours) == 0:
    continue

  approx = triangleify(cv.convexHull(max(contours, key=cv.contourArea)))
  lengths = find_lengths(approx)
  top = (find_smallest_idx(lengths) + 2) % 3
  angle = find_orientation(approx, top)
  print(f"The cone has an orientation of {angle} degrees.")
  cv.drawContours(hsv, [approx], 0, (0, 255, 0), 3)
  cv.circle(hsv, approx[top][0], 3, (0, 0, 255), 3)
  cv.imshow("Image #5", hsv)

  if (key := cv.waitKey(0)) & 0xFF == ord("q"):
    break

cap.release()
cv.destroyAllWindows()
