import joblib
from pathlib import Path
import glob
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

PROJ_PATH = Path('/home/dendav/projects/Intro2CV_course/Homework_2')
IMAGES_PATH = PROJ_PATH/'images'
TRAIN_PATH = PROJ_PATH/'train'
TRAIN_IMG_NAMES = glob.glob(str(TRAIN_PATH/'*.jpg')) 

SCALE = 0.33


def find_puzzle( image: np.ndarray, debug: bool = False ):
	area_thr = 5000
	
	gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
	blurred = cv.GaussianBlur(gray, (7, 7), 5)
	thresh = cv.adaptiveThreshold(blurred, 255,
		cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
	thresh = cv.bitwise_not(thresh)
	open_kernel = np.ones((3,3))
	open_iters = 2
	thresh = cv.morphologyEx(thresh, cv.MORPH_OPEN, open_kernel, iterations=open_iters)
	# To eliminate several contours in thick sudoku borders
	dilate_kernel = np.ones((3,3))
	dilate_iters = 3
	thresh = cv.morphologyEx(thresh, cv.MORPH_DILATE, dilate_kernel, iterations=dilate_iters)

	if debug:
		cv.namedWindow("Puzzle_Thresh", cv.WINDOW_NORMAL)
		cv.resizeWindow("Puzzle_Thresh", 600, 400)
		cv.imshow("Puzzle_Thresh", thresh)
	cnts, hierarchy = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL,
		cv.CHAIN_APPROX_SIMPLE)
	print(len(cnts), len(hierarchy[0]))
	cnts, hierarchy = zip(*((c, h) for c,h in zip(cnts,hierarchy[0]) if cv.contourArea(c) > area_thr))
	hierarchy = [hierarchy]
	if debug:
		det_cnts_img = img.copy()
		det_cnts_img = cv.drawContours(det_cnts_img, cnts, -1, color=(0,150,255), thickness=3)
		cv.namedWindow("detected_contours", cv.WINDOW_NORMAL)
		cv.resizeWindow("detected_contours", 600, 400)		
		cv.imshow("detected_contours", det_cnts_img)

	# cnts_imu = imutils.grab_contours(cnts)
	cnts_imu = sorted(cnts, key=cv.contourArea, reverse=True)
	puzzle_cnt = None
	
	for c in cnts_imu:
		peri = cv.arcLength(c, True)
		approx = cv.approxPolyDP(c, 0.02 * peri, closed=True)
		if len(approx) == 4:
			puzzle_cnt = approx
			break

	if debug:
		if puzzle_cnt is not None:
			output = image.copy()
			cv.namedWindow("puzzle_outline", cv.WINDOW_NORMAL)
			cv.resizeWindow("puzzle_outline", 600, 400)
			cv.drawContours(output, [puzzle_cnt], -1, (0, 255, 0), 2)
			cv.imshow("puzzle_outline", output)
		else:
			print('I have not found puzzle contour!')

	puzzle = None
	warped = None

	if puzzle_cnt is None:
		return (puzzle, warped) 

	puzzl
	puzzle = None
	warped = None

	if puzzle_cnt is None:
		return (puzzle, warped) 

	puzzle = None
	warped = None

	if puzzle_cnt is None:
		return (puzzle, warped) 

	puzzle = None
	warped = None

	if puzzle_cnt is None:
		return (puzzle, warped) 

	puzzle = 
	puzzle = four_point_transform(image, puzzle_cnt.reshape(4, 2))
	warped = four_point_transform(gray, puzzle_cnt.reshape(4, 2))

	if debug:
		cv.namedWindow("puzzle_transform", cv.WINDOW_NORMAL)
		cv.resizeWindow("puzzle_transform", 600, 400)
		cv.imshow("puzzle_transform", puzzle)
	# return a 2-tuple of puzzle in both RGB and grayscale
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


if __name__ == '__main__':
	
	img = cv.imread(TRAIN_IMG_NAMES[2])
	puzzle, warped = find_puzzle(img, debug=True)

	while True:
		key = cv.waitKey(1) & 0xFF
		if key == ord('q'):
			break
	cv.destroyAllWindows()