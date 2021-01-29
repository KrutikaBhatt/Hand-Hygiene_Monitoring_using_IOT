# Question2 :Trackbar for definin hsv color range in live webcam

import cv2
import numpy as np

cam = cv2.VideoCapture(0)
cv2.namedWindow('Frame')


def nothing(x):
    pass


cv2.namedWindow('Frame')

# Creating Trackbar
cv2.createTrackbar('H', 'Frame', 0, 179, nothing)
cv2.createTrackbar('S', 'Frame', 0, 255, nothing)
cv2.createTrackbar('V', 'Frame', 0, 255, nothing)

while True:
    x, image = cam.read()  # Taking Inputs from Live web cam
    # Convert color images to HSV
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Get Trackbar position
    h = cv2.getTrackbarPos('H', 'Frame')
    s = cv2.getTrackbarPos('S', 'Frame')
    v = cv2.getTrackbarPos('V', 'Frame')

    lower_bound = np.array([h, s, v])  # Input current Track as Lower-Bound as mask

    upper_bound = np.array([179, 255, 255])  # HSV ranges from(0,0,0)->(179,255,255)

    mask1 = cv2.inRange(hsv_img, lower_bound, upper_bound)
    result = cv2.bitwise_and(image, image, mask=mask1)

    cv2.imshow('Frame', result)
    cv2.imshow('Binary_Image', mask1)

    if cv2.waitKey(5) & 0xFF == ord('a'):
        break