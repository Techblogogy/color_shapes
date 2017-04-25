# USAGE
# python detect_shapes.py --image shapes_and_colors.png

# import the necessary packages
from pyimagesearch.shapedetector import ShapeDetector
import argparse
import imutils
import cv2

import numpy as np

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
args = vars(ap.parse_args())

# load the image and resize it to a smaller factor so that
# the shapes can be approximated better
image = cv2.imread(args["image"])
resized = imutils.resize(image, width=300)
ratio = image.shape[0] / float(resized.shape[0])

# detect by color
hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

cv2.imwrite("hsv.jpg", hsv)

r_low = np.array([0, 100, 100])
r_up = np.array([10, 255, 255])

p_low = np.array([160, 100, 100])
p_up = np.array([179, 255, 255])

mask1 = cv2.inRange(hsv, r_low, r_up)
mask2 = cv2.inRange(hsv, p_low, p_up)

cv2.imwrite("mask1.jpg", mask1)
cv2.imwrite("mask2.jpg", mask2)

# convert the resized image to grayscale, blur it slightly,
# and threshold it
# gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
mask = cv2.GaussianBlur(mask1, (5, 5), 0)
# thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)[1]
#
# cv2.bitwise_not(thresh, thresh)
#
# cv2.imwrite("gray.jpg", gray)
# cv2.imwrite("blurred.jpg", blurred)
# cv2.imwrite("thresh.jpg", thresh)


# find contours in the thresholded image and initialize the
# shape detector
# cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
sd = ShapeDetector()

print cnts

# loop over the contours
for c in cnts:
	# compute the center of the contour, then detect the name of the
	# shape using only the contour
	M = cv2.moments(c)
	cX = int((M["m10"] / M["m00"]) * ratio)
	cY = int((M["m01"] / M["m00"]) * ratio)
	shape = sd.detect(c)

	# multiply the contour (x, y)-coordinates by the resize ratio,
	# then draw the contours and the name of the shape on the image
	c = c.astype("float")
	c *= ratio
	c = c.astype("int")
	cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
	cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
		0.5, (255, 255, 255), 2)

	# show the output image
	# cv2.imshow("Image", image)
	cv2.imwrite("image.jpg", image)
	# cv2.waitKey(0)
