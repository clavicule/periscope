import cv2
from skimage.filters import threshold_adaptive
from skimage import measure
import numpy as np


class Scissors(object):
	@staticmethod
	def cut(image):
		V = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))[2]
		thresh = threshold_adaptive(V, 29, offset=15).astype("uint8") * 255
		thresh = cv2.bitwise_not(thresh)

		# perform a connected components analysis and initialize the mask to store the locations
		# of the character candidates
		labels = measure.label(thresh, neighbors=8, background=0)

		boxes = []

		# loop over the unique components
		for label in np.unique(labels):
			# if this is the background label, ignore it
			if label == 0:
				continue

			# otherwise, construct the label mask to display only connected components for the
			# current label, then find contours in the label mask
			labelMask = np.zeros(thresh.shape, dtype="uint8")
			labelMask[labels == label] = 255
			(_, cnts, _) = cv2.findContours(labelMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

			# ensure at least one contour was found in the mask

			if len(cnts) > 0:
				# grab the largest contour which corresponds to the component in the mask, then
				# grab the bounding box for the contour
				c = max(cnts, key=cv2.contourArea)
				(boxX, boxY, boxW, boxH) = cv2.boundingRect(c)

				# compute the aspect ratio, solidity, and height ratio for the component
				aspectRatio = boxW / float(boxH)
				solidity = cv2.contourArea(c) / float(boxW * boxH)
				heightRatio = boxH / float(thresh.shape[0])

				# determine if the aspect ratio, solidity, and height of the contour pass
				# the rules tests
				keepAspectRatio = aspectRatio < 1.0
				keepSolidity = solidity > 0.15
				keepHeight = heightRatio > 0.4 and heightRatio < 0.95

				# check to see if the component passes all the tests
				if keepAspectRatio and keepSolidity and keepHeight:
					# compute the convex hull of the contour and draw it on the character
					# candidates mask
					# hull = cv2.convexHull(c)
					# cv2.drawContours(charCandidates, [hull], -1, 255, -1)
					#
					dX = min(35, 35 - boxW) // 2
					boxX -= dX
					boxW += (dX * 2)
					boxes.append((boxX, boxY, boxX + boxW, boxY + boxH))

		# sort the bounding boxes from left to right
		boxes = sorted(boxes, key=lambda b: b[0])

		return boxes, thresh
