import glob
from pathlib import Path
import random
import re

import numpy as np
import cv2 as cv
import torch
from torchvision import transforms


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True

RELEASE = True

PROJ_PATH = Path('/home/dendav/projects/Intro2CV_course/Homework_2')
# PROJ_PATH = Path('/home/dendenmushi/projects/Intro2CV_course/Homework_2')
IMAGES_PATH = PROJ_PATH/'images'
TRAIN_PATH = PROJ_PATH/'train'
TRAIN_IMG_NAMES = glob.glob(str(TRAIN_PATH/'*.jpg'))
TRAIN_IMG_NAMES.sort(key = lambda x: re.findall(r'\d+', x)[-1])
WEIGHTS_PATH = '/autograder/submission/mnist_net.pt' if RELEASE else '/home/dendav/projects/Intro2CV_course/Homework_2/mnist_net.pt'

LOWER_THR, UPPER_THR = (0,255)
SCALE = 0.33
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
trans = transforms.ToTensor()


class LeNet5(torch.nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()

        self.conv1 = torch.nn.Conv2d(
            in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.act1  = torch.nn.Tanh()
        self.pool1 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv2 = torch.nn.Conv2d(
            in_channels=6, out_channels=16, kernel_size=5, padding=0)
        self.act2  = torch.nn.Tanh()
        self.pool2 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.fc1   = torch.nn.Linear(5 * 5 * 16, 120)
        self.act3  = torch.nn.Tanh()

        self.fc2   = torch.nn.Linear(120, 84)
        self.act4  = torch.nn.Tanh()

        self.fc3   = torch.nn.Linear(84, 10)

    def forward(self, x):

        x = self.conv1(x)
        x = self.act1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.act2(x)
        x = self.pool2(x)

        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))

        x = self.fc1(x)
        x = self.act3(x)
        x = self.fc2(x)
        x = self.act4(x)
        x = self.fc3(x)

        return x

MODEL = LeNet5()
MODEL = MODEL.to(DEVICE)
MODEL.load_state_dict(torch.load(WEIGHTS_PATH, map_location=torch.device('cpu')))


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
        approx = cv.approxPolyDP(hull, 0.01 * peri, closed=True)
        if len(approx) > 8:
            continue

        rect = cv.minAreaRect(c)
        center, dims, angle = rect
        box = cv.boxPoints(rect)
        box = np.int0(box)
        if min(dims)/max(dims) < rect_ratio_thresh:
            continue

        puzzle_cnts.append(approx)

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
            center, dims, angle = rect
            width = int(max(dims))
        else :
            box = c
            width = int(np.linalg.norm(box[0]-box[1]))
        dst = np.array([[0, 0],[width, 0],[width, width],[0, width]], dtype = "float32")
        box = box.reshape((4,2)).astype(np.float32)

        M = cv.getPerspectiveTransform(box, dst)
        warped = cv.warpPerspective(img_orig, M, (width, width))
        warped_imgs.append(warped)

    if debug:
        cv.namedWindow("warped", cv.WINDOW_NORMAL)
        cv.resizeWindow("warped", 600, 400)
        cv.imshow('warped',warped_imgs[0])
    return warped_imgs


def warp_puzzles_old( puzzle_contours, img_orig, debug=False):

    warped_imgs = []

    for c in puzzle_contours:
        if c.shape[0] == 4:
            box = c.reshape(4,-1)
        else:
            rect = cv.minAreaRect(c)
            box = cv.boxPoints(rect)
            box = np.int0(box)

        rect = np.zeros((4, 2), dtype = "float32")

        #define order of points
        summ = box.sum(axis = 1)
        rect[0] = box[np.argmin(summ)]
        rect[2] = box[np.argmax(summ)]
        diff = np.diff(box, axis = 1)
        rect[1] = box[np.argmin(diff)]
        rect[3] = box[np.argmax(diff)]

        (tl, tr, br, bl) = rect
        width1 = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        width2 = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        max_width = max(int(width1), int(width2))
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

        maxHeight = max(int(heightA), int(heightB))
        dst = np.array([[0, 0],[max_width, 0],[max_width, maxHeight],[0, maxHeight]], dtype = "float32")
        M = cv.getPerspectiveTransform(rect, dst)
        warped = cv.warpPerspective(img_orig, M, (max_width, maxHeight))
        warped_imgs.append(warped)

    if debug:
        cv.namedWindow("warped", cv.WINDOW_NORMAL)
        cv.resizeWindow("warped", 600, 400)
        cv.imshow('warped',warped_imgs[0])
    return warped_imgs


def find_puzzle( img, debug=False ):
    puzzle_mask, puzzle_cnts = find_puzzle_mask(img, debug=False)
    # return puzzle_mask, warp_puzzles_old( puzzle_cnts, img, debug=True)
    return puzzle_mask, warp_puzzles_old( puzzle_cnts, img, debug=debug)


def find_cells(img, debug=False):
    """
    Find the cells of a sudoku grid
    """
    img_area = img.shape[0] * img.shape[1]

    contours, _ = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # Array containing the cropped cell image and its position in the grid
    cells = []
    for c in contours:
        area = cv.contourArea(c)

        # Approximate the contour in order to determine whether the contour is a quadrilateral
        peri = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, 0.017 * peri, True)

        # We are looking for a contour of a specific area in relation to the grid size
        # and that is roughly quadrilateral
        # We filter for areas that are too small or too large in relation to the whole image
        if area / img_area > 0.0001 and area / img_area < 0.02 and len(approx) == 4:
            # Using masking, we crop the cell into its own 28 by 28 pixel image
            mask = np.zeros_like(img)
            cv.drawContours(mask, [c], -1, 255, -1)

            (y, x) = np.where(mask == 255)

            (top_y, top_x) = (np.min(y), np.min(x))
            (bottom_y, bottom_x) = (np.max(y), np.max(x))
            cell = img[top_y : bottom_y + 1, top_x : bottom_x + 1].copy()

            cell = cv.resize(cell, (28, 28))

            # We also find the centroid of the cell in relation
            # to the grid
            M = cv.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            cells.append(({"img": cell, "pos": (cX, cY)}))

    return cells


def find_cells_manual( warped_gray, debug=False ):
    assert len(warped_gray.shape) == 2

    h,w = warped_gray.shape
    h_cell = int(h / 9)
    w_cell = int(w / 9)
    win = 10
    cells = []
    for i in range(9):
        cells.append([])
        for j in range(9):
            begin = [h_cell*i, w_cell*j]
            end = [h_cell*(i+1), w_cell*(j+1)]
            cell_img = warped_gray[begin[0]+win:end[0]-win, begin[1]+win:end[1]-win].copy()
            cells[i].append(cell_img)

    return cells


def find_cells_digits(cells, debug=False):

    cell_area = np.prod(cells[0][0].shape)
    cont_area_thr = 0.03
    max_area_thr = 0.9
    digit_width = 40

    digit_cells = np.full((9, 9), -1)
    pred_vals = np.zeros((9, 9))

    for i in range(9):
        for j in range(9):
            cell = cells[i][j].copy()
            # find_thresh_for_digits(cell)

            blurred = cv.GaussianBlur(cell, (7, 7), 5)
            thresh = cv.adaptiveThreshold(blurred, 255,
                cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
            thresh = cv.bitwise_not(thresh)
            open_kernel = np.ones((3,3))
            open_iters = 2
            thresh = cv.morphologyEx(thresh, cv.MORPH_OPEN, open_kernel, iterations=open_iters, borderType=cv.BORDER_CONSTANT)
            thresh[:,:10] = 0
            thresh[:,-10:] = 0
            thresh[:10,:] = 0
            thresh[-10:,:] = 0

            h, w = cell.shape
            contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

            if debug:
                cv.namedWindow("contours in cell", cv.WINDOW_NORMAL)
                cv.resizeWindow("contours in cell", 600, 400)

                # cv.namedWindow("cell", cv.WINDOW_NORMAL)
                # cv.resizeWindow("cell", 600, 400)
                det_img = thresh.copy()
                det_img = cv.drawContours(det_img, contours, -1, (0,255,0))
                cv.imshow('contours in cell', det_img)

                # cell_filtered = cv.resize(cell, (28, 28))
                # cell_filtered = cv.normalize(cell_filtered, cell_filtered, 0, 255, cv.NORM_MINMAX)
                # cell_filtered = cv.bitwise_not(cell_filtered)
                # cv.imshow('cell', cell_filtered)
                while True:
                    key = cv.waitKey(100) & 0xFF
                    if key == ord('q'):
                        break


            if debug:
                if i == 1 and j == 3:
                    debug_cnts = contours
            area = 0
            if len(contours):
                areas = [cv.contourArea(c) for c in contours]/cell_area
                contours, areas = zip(*((c, ar) for c,ar in zip(contours,areas) if ar < max_area_thr))
                if len(areas):
                    max_ind = np.argmax(areas)
                    area = areas[max_ind]

            if debug:
                print(f'area: {area}')
            if  area < cont_area_thr or area > max_area_thr:
                digit_cells[i,j] = -1
                continue

            # area = len(np.where(cv.threshold(cell, empty_thr, 255,
            #            cv.THRESH_BINARY)[1] > empty_thr)[0])/(h*w)
            # area = (cv.adaptiveThreshold(cell, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2).sum()/255)/(h*w)

            cell = cv.GaussianBlur(cell, (7, 7), 5)
            cell = cv.normalize(cell, cell, 0, 255, cv.NORM_MINMAX)
            cell = cv.bitwise_not(cell)
            if len(areas):
                dig_cnt = contours[max_ind]
                dig_rect = cv.boundingRect(dig_cnt)
                buff = cell.copy()
                cell[:,:] = 0;
                cell[dig_rect[1]-10:dig_rect[1]+dig_rect[3]+10, dig_rect[0]-10:dig_rect[0]+dig_rect[2]+10] = buff[dig_rect[1]-10:dig_rect[1]+dig_rect[3]+10, dig_rect[0]-10:dig_rect[0]+dig_rect[2]+10].copy()
            cell = cv.resize(cell, (28, 28))


            if area < cont_area_thr*1.5:
                cell = cv.morphologyEx(cell, cv.MORPH_DILATE, np.ones((5,5)), iterations=open_iters+10, borderType=cv.BORDER_CONSTANT)


            if debug:
                cv.namedWindow("cell", cv.WINDOW_NORMAL)
                cv.resizeWindow("cell", 600, 400)
                cv.imshow('cell', cell)
                while True:
                    key = cv.waitKey(100) & 0xFF
                    if key == ord('q'):
                        break


            cell = trans(cell)
            cell = cell.to(DEVICE)
            pred = MODEL.forward(cell[np.newaxis, ...])
            maximum = pred.max()
            dig = torch.where(pred == maximum)[1][0]
            if dig == 0:
                dig = 6
            if debug:
                print(f'I found {dig} digit, pred_vals: {pred}')
            digit_cells[i, j] = dig
            pred_vals[i,j] = maximum
            pred_vals[i, j] = str(round(area, 2))

    return digit_cells, pred_vals


def predict_image(image: np.ndarray, debug=False) -> (np.ndarray, list):

    mask, warp_puzzles = find_puzzle(image, debug=debug)

    sudoku_digits = []
    for warp in warp_puzzles:
        puzzle_cells = find_cells_manual(cv.cvtColor(warp, cv.COLOR_BGR2GRAY))
        digit_cells, pred_vals = find_cells_digits(puzzle_cells, debug=debug)
        sudoku_digits.append(digit_cells)
    return mask, sudoku_digits

def empty(a):
        pass

def find_thresh_for_digits(img):
    global LOWER_THR, UPPER_THR

    cv.namedWindow('Calibrate')
    cv.createTrackbar("LOWER_THR", 'Calibrate', LOWER_THR, 255, empty)
    cv.createTrackbar("UPPER_THR", 'Calibrate', UPPER_THR, 255, empty)

    # img = cv.normalize(img, img, 0, 255, cv.NORM_MINMAX)
    # img = cv.GaussianBlur(img, (11,11), 10)
    # img = cv.bitwise_not(img)

    # This part is from find mask
    blurred = cv.GaussianBlur(img, (7, 7), 5)
    thresh = cv.adaptiveThreshold(blurred, 255,
        cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
    thresh = cv.bitwise_not(thresh)
    open_kernel = np.ones((3,3))
    open_iters = 2
    thresh = cv.morphologyEx(thresh, cv.MORPH_OPEN, open_kernel, iterations=open_iters, borderType=cv.BORDER_CONSTANT)

    print(img.min(), img.max())
    cv.namedWindow("contours in cell", cv.WINDOW_NORMAL)
    cv.resizeWindow("contours in cell", 600, 400)

    cv.namedWindow("cell", cv.WINDOW_NORMAL)
    cv.resizeWindow("cell", 600, 400)
    while True:
        LOWER_THR = cv.getTrackbarPos("LOWER_THR", "Calibrate")
        UPPER_THR = cv.getTrackbarPos("UPPER_THR", "Calibrate")
        # thr = img.copy()
        # thr = cv.inRange(img, LOWER_THR, UPPER_THR)
        contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        det_img = thresh.copy()
        det_img = cv.drawContours(det_img, contours, -1, (0,255,0))
        cv.imshow('contours in cell', det_img)
        cv.imshow('cell', img)

        key = cv.waitKey(100) & 0xFF
        if key == ord('q'):
            break
    return LOWER_THR, UPPER_THR

if __name__ == '__main__':

    debug=False
    img = cv.imread(TRAIN_IMG_NAMES[2])
    mask, sudoku_digits = predict_image(img, debug=debug)

    cv.namedWindow("puzzle_mask", cv.WINDOW_NORMAL)
    cv.resizeWindow("puzzle_mask", 600, 400)
    cv.imshow('puzzle_mask',mask)

    cv.namedWindow("img", cv.WINDOW_NORMAL)
    cv.resizeWindow("img", 600, 400)
    cv.imshow('img',img)
    print(sudoku_digits)
    while True:
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    cv.destroyAllWindows()

