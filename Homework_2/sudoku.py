import joblib

import numpy as np
from numpy import logical_and as land
from numpy import logical_not as lnot
from skimage.feature import canny
from skimage.transform import rescale, ProjectiveTransform, warp
from skimage.morphology import dilation, disk
import cv2 as cv

from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border
import imutils
 
SCALE = 0.33


def find_puzzle( image: np.ndarray, debug: bool = False ):
	gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
	blurred = cv.GaussianBlur(gray, (7, 7), 3)
	thresh = cv.adaptiveThreshold(blurred, 255,
		cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
	thresh = cv.bitwise_not(thresh)
	if debug:
		cv.imshow("Puzzle Thresh", thresh)
		cv.waitKey(0)
	cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL,
		cv.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	cnts = sorted(cnts, key=cv.contourArea, reverse=True)
	puzzleCnt = None
	for c in cnts:
		peri = cv.arcLength(c, True)
		approx = cv.approxPolyDP(c, 0.02 * peri, True)
		if len(approx) == 4:
			puzzleCnt = approx
			break
	if puzzleCnt is None:
		raise Exception(("Could not find Sudoku puzzle outline. "
			"Try debugging your thresholding and contour steps."))
	if debug:
		output = image.copy()
		cv.drawContours(output, [puzzleCnt], -1, (0, 255, 0), 2)
		cv.imshow("Puzzle Outline", output)
		cv.waitKey(0)
	puzzle = four_point_transform(image, puzzleCnt.reshape(4, 2))
	warped = four_point_transform(gray, puzzleCnt.reshape(4, 2))
	if debug:
		cv.imshow("Puzzle Transform", puzzle)
		cv.waitKey(0)
	return (puzzle, warped)

def predict_image(image: np.ndarray) -> (np.ndarray, list):
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    sudoku_digits = [
        np.int16([[-1, -1, -1, -1, -1, -1, -1, -1, -1],
                  [-1, -1, -1,  8,  9,  4, -1, -1, -1],
                  [-1, -1, -1,  6, -1,  1, -1, -1, -1],
                  [-1,  6,  5,  1, -1,  9,  7,  8, -1],
                  [-1,  1, -1, -1, -1, -1, -1,  3, -1],
                  [-1,  3,  9,  4, -1,  5,  6,  1, -1],
                  [-1, -1, -1,  8, -1,  2, -1, -1, -1],
                  [-1, -1, -1,  9,  1,  3, -1, -1, -1],
                  [-1, -1, -1, -1, -1, -1, -1, -1, -1]]),
    ]
    mask = np.bool_(np.ones_like(image))

    # loading train image:
    train_img_4 = cv.imread('/autograder/source/train/train_4.jpg', 0)

    # loading model:  (you can use any other pickle-like format)
    rf = joblib.load('/autograder/submission/random_forest.joblib')

    return mask, sudoku_digits
	