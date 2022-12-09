import joblib
from pathlib import Path
import glob
import re

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

PROJ_PATH = Path('/home/dendenmushi/projects/Intro2CV_course/Homework_2')
IMAGES_PATH = PROJ_PATH/'images'
TRAIN_PATH = PROJ_PATH/'train'
TRAIN_IMG_NAMES = glob.glob(str(TRAIN_PATH/'*.jpg'))
TRAIN_IMG_NAMES.sort(key = lambda x: re.findall('\d+', x)[-1])

SCALE = 0.33


def find_puzzle_mask( image: np.ndarray, debug: bool = False ):
	area_thr = 700*700
	rect_ratio_thresh = 0.7
	puzzle_areas_ratio_min = 0.7

	gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
	blurred = cv.GaussianBlur(gray, (7, 7), 5)
	thresh = cv.adaptiveThreshold(blurred, 255,
		cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
	thresh = cv.bitwise_not(thresh)
	open_kernel = np.ones((3,3))
	open_iters = 2
	
	thresh = cv.morphologyEx(thresh, cv.MORPH_OPEN, open_kernel, iterations=open_iters, borderType=cv.BORDER_CONSTANT)
	# To eliminate several contours in thick sudoku borders
	dilate_kernel = np.ones((3,3))
	dilate_iters = 5
	# thresh = cv.morphologyEx(thresh, cv.MORPH_DILATE, dilate_kernel, iterations=dilate_iters, borderType=cv.BORDER_CONSTANT)

	if debug:
		cv.namedWindow("Puzzle_Thresh", cv.WINDOW_NORMAL)
		cv.resizeWindow("Puzzle_Thresh", 600, 400)
		cv.imshow("Puzzle_Thresh", thresh)
	cnts, hierarchy = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL,
		cv.CHAIN_APPROX_SIMPLE)
	
	cnts, hierarchy = zip(*((c, h) for c,h in zip(cnts,hierarchy[0]) if cv.contourArea(c) > area_thr))
	hierarchy = [hierarchy]

	if debug:
		det_cnts_img = img.copy()
		hulls = [cv.convexHull(c) for c in cnts]
		det_cnts_img = cv.drawContours(det_cnts_img, hulls, -1, color=(0,150,255), thickness=3)
		det_cnts_img = cv.drawContours(det_cnts_img, cnts, -1, color=(0,255,255), thickness=3)
		cv.namedWindow("detected_contours", cv.WINDOW_NORMAL)
		cv.resizeWindow("detected_contours", 600, 400)		
		cv.imshow("detected_contours", det_cnts_img)

	# cnts_imu = imutils.grab_contours(cnts)
	cnts_imu = sorted(cnts, key=cv.contourArea, reverse=True)
	puzzle_cnts = []
	puzzle_mask = None
	
	max_puzzle_area = cv.contourArea(cnts_imu[0])
	cnts_imu = list(filter(lambda c: cv.contourArea(c)/max_puzzle_area > puzzle_areas_ratio_min, cnts_imu) )

	for c in cnts_imu:
		peri = cv.arcLength(c, True)
		hull = cv.convexHull(c)
		approx = cv.approxPolyDP(hull, 0.02 * peri, closed=True)
		if len(approx) > 8:
			continue	
		
		rect = cv.minAreaRect(c)
		center, dims, angle = rect
		box = cv.boxPoints(rect)
		box = np.int0(box)
		if min(dims)/max(dims) < rect_ratio_thresh:
			continue

		puzzle_cnts.append(hull)

	if debug:
		if len(puzzle_cnts) > 0:
			output = image.copy()
			cv.namedWindow("puzzle_outline", cv.WINDOW_NORMAL)
			cv.resizeWindow("puzzle_outline", 600, 400)
			cv.drawContours(output, puzzle_cnts, -1, (0, 255, 0), 2)
			cv.imshow("puzzle_outline", output)
		else:
			print('I have not found puzzle contour!')

	if len(puzzle_cnts) == 0 :
		return puzzle_mask, puzzle_cnts

	puzzle_mask = np.zeros_like(thresh)
	puzzle_mask = cv.fillPoly(puzzle_mask, puzzle_cnts, 255)
		

	# puzzle = four_point_transform(image, puzzle_cnts.reshape(4, 2))
	# warped = four_point_transform(gray, puzzle_cnts.reshape(4, 2))

	if debug:
		cv.namedWindow("puzzle_mask", cv.WINDOW_NORMAL)
		cv.resizeWindow("puzzle_mask", 600, 400)
		cv.imshow("puzzle_mask", puzzle_mask/255)
	# return a 2-tuple of puzzle in both RGB and grayscale
	return puzzle_mask/255, puzzle_cnts


def warp_puzzles( puzzle_contours, img_orig, debug=False):

	warped_imgs = []
	
	for c in puzzle_contours:
		if len(c) > 4:
			rect = cv.minAreaRect(c)

			box = cv.boxPoints(rect)
			# box = np.int0(box)
			center, dims, angle = rect
			width = int(max(dims))
		else :
			box = c
		dst = np.array([[0, 0],[width, 0],[width, width],[0, width]], dtype = "float32")
		box = box.reshape((4,2))
		M = cv.getPerspectiveTransform(box, dst)
		warped = cv.warpPerspective(img_orig, M, (width, width))
		warped_imgs.append(warped)
	
	if debug:
		cv.namedWindow("warped", cv.WINDOW_NORMAL)
		cv.resizeWindow("warped", 600, 400)
		cv.imshow('warped',warped_imgs[0])
	return warped_imgs


def warp_puzzles_denis( puzzle_contours, img_orig ):
	warped_imgs = []
	
	for c in puzzle_contours:
		rect = cv.minAreaRect(c)

		box = cv.boxPoints(rect)
		box = np.int0(box)
		perimeter = cv.arcLength(c, True) 
		epsilon = 0.01 * perimeter
		approx = cv.approxPolyDP(c,epsilon,True)
		if approx.shape[0] == 4:
			box = approx.reshape(4,-1)
		center, dims, angle = rect
		# dst = np.array([[0, 0],[max(dims) - 1, 0],[max(dims) - 1, max(dims) - 1],[0, maxHeight - 1]], dtype = "float32")
		# M = cv.getPerspectiveTransform(rect, dst)
		# warped = cv.warpPerspective(img_orig, M, (maxWidth, maxHeight))
		# warped_imgs.append(warped)
		
	return warped_imgs


def warp_puzzles_old( puzzle_contours, img_orig):

	warped_imgs = []
	
	for c in puzzle_contours:
		rect = cv.minAreaRect(c)

		box = cv.boxPoints(rect)
		box = np.int0(box)
		perimeter = cv.arcLength(c, True) 
		epsilon = 0.01 * perimeter
		approx = cv.approxPolyDP(c,epsilon,True)
		if approx.shape[0] == 4:
			box = approx.reshape(4,-1)
		rect = np.zeros((4, 2), dtype = "float32")
		s = box.sum(axis = 1)
		rect[0] = box[np.argmin(s)]
		rect[2] = box[np.argmax(s)]
		diff = np.diff(box, axis = 1)
		rect[1] = box[np.argmin(diff)]
		rect[3] = box[np.argmax(diff)]
		(tl, tr, br, bl) = rect
		widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
		widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
		maxWidth = max(int(widthA), int(widthB))
		heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
		heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
		maxHeight = max(int(heightA), int(heightB))
		dst = np.array([[0, 0],[maxWidth - 1, 0],[maxWidth - 1, maxHeight - 1],[0, maxHeight - 1]], dtype = "float32")
		M = cv.getPerspectiveTransform(rect, dst)
		warped = cv.warpPerspective(img_orig, M, (maxWidth, maxHeight))
		warped_imgs.append(warped)
		
	return warped_imgs

def find_puzzle( img, debug=False ):
	puzzle_mask, puzzle_cnts = find_puzzle_mask(img, debug)
	return puzzle_mask, warp_puzzles_denis( puzzle_cnts, img)
	
	

def predict_image(image: np.ndarray) -> (np.ndarray, list):
    # image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
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
    puzzles_mask = find_puzzle(img)


    # loading train image:
    train_img_4 = cv.imread('/autograder/source/train/train_4.jpg', 0)

    # loading model:  (you can use any other pickle-like format)
    rf = joblib.load('/autograder/submission/random_forest.joblib')

    return mask, sudoku_digits


if __name__ == '__main__':
	
	img = cv.imread(TRAIN_IMG_NAMES[1])
	puzzle_mask, warp_puzzles = find_puzzle(img, debug=True)
	print(warp_puzzles)
	for i, warped in enumerate(warp_puzzles):
		cv.namedWindow(f"warped_puzze_{i}", cv.WINDOW_NORMAL)
		cv.resizeWindow(f"warped_puzze_{i}", 600, 400)
		cv.imshow(f"warped_puzze_{i}", warped)


	while True:
		key = cv.waitKey(1) & 0xFF
		if key == ord('q'):
			break
	cv.destroyAllWindows()