from typing import Union

from collections import defaultdict
from itertools import combinations

import numpy as np
import cv2
from skimage.transform import rescale
from skimage.measure import label, find_contours
from skimage.filters import gaussian
from scipy.spatial.distance import cdist
import scipy.stats as st
import json


COLORS = ('blue', 'green', 'black', 'yellow', 'red')
TRAINS2SCORE = {1: 1, 2: 2, 3: 4, 4: 7, 6: 15, 8: 21}

img = cv2.imread('train_all.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
CITY_TEMPL = img[197:260, 622:685]
CITY_TEMPL_GRAY = cv2.cvtColor(CITY_TEMPL, cv2.COLOR_BGR2GRAY)


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

APPROX_CENTERS = load_json(f'all_centers.json')

def find_cities_centers_opencv( img_gray, method=cv2.TM_CCOEFF_NORMED ):
    
    roi_win = 120
    match_centers = []
    w_templ = CITY_TEMPL.shape[1]
    h_templ = CITY_TEMPL.shape[0]
    for pt in approx_centers:
        roi = img_gray[pt[0]-roi_win:pt[0]+roi_win, pt[1]-roi_win:pt[1]+roi_win]
        match = cv2.matchTemplate(roi, CITY_TEMPL_GRAY, method  )
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match)
        x = max_loc[0] + w_templ // 2 + pt[1] - roi_win
        y = max_loc[1] + h_templ // 2 + pt[0] - roi_win
        i = y; j = x
        match_centers.append([i, j])

    return np.int64(match_centers)

def find_blue_trains_mask( img_bgr ):

    img = img_bgr.copy()
    roi_win = 70
    k_size = 3
    kernel = np.ones((k_size, k_size))
    min_area = 1000
    min_len = 100
    lower_bound = (90, 159, 90)
    upper_bound = (112, 255, 180)
    # 1 hsv filter
    HSV = cv2.cvtColor( img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange( HSV, lower_bound, upper_bound).astype(np.uint8)
    # 2 erosion etc
   

def find_blue_trains_mask( img_bgr ):

    img = img_bgr.copy()
    roi_win = 70
    k_size = 3
    kernel = np.ones((k_size, k_size))
    min_area = 1000
    min_len = 100
    lower_bound = (90, 159, 90)
    upper_bound = (112, 255, 180)
    # 1 hsv filter
    HSV = cv2.cvtColor( img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange( HSV, lower_bound, upper_bound).astype(np.uint8)
    # 2 erosion etc
    mask = cv2.morphologyEx( mask, cv2.MORPH_ERODE, kernel)
    # 3 find contours -> oriented rectangles
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # 4 filters areas and dims
    filtered_contours = []
    trains_mask = np.zeros_like(mask, dtype=np.uint8)
    # Maybe I shoud firstly remove holes
    labelled, nlabels = label(mask, connectivity=2, return_num=True)
    train_centers = []
    for cnt in contours:
        hull = cv2.convexHull(cnt)
        area = cv2.contourArea(hull)
        if area < min_area:
            continue
        rect = cv2.minAreaRect(hull)
        center, dims, angle = rect
        center = (int(center[0]), int(center[1]))
        if max(dims) < min_len:
            continue
        label_id = labelled[center[1], center[0]]
        if label_id != 0:
            train_mask = labelled == label_id
            trains_mask = np.logical_or(trains_mask, train_mask )
    
    trains_mask = 255 * trains_mask.astype(np.uint8)
    print(trains_mask.min(), trains_mask.max())
    erode_ksize=3
    erode_kernel=np.ones((erode_ksize,erode_ksize))
    sticked_trains_mask = cv2.morphologyEx(trains_mask, cv2.MORPH_DILATE, erode_kernel, iterations=10)
    # sticked_trains_label = label(sticked_trains_mask, connectivity=2, return_num=True)
    

    # 5 create roi for each possible train and rotate template
        # roi = img[center[1]-roi_win:center[1]-roi_win, center[0]-roi_win:center[0]+roi_win].copy()
        # blue_rotated = cv2.imread('images/blue_rotated.jpg')
        # height, width = blue_rotated.shape[:2]
        # center = (width/2, height/2)
        # rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=35, scale=1)
        # rotated_image = cv2.warpAffine(src=blue_rotated, M=rotate_matrix, dsize=(width, height))

        # plot_img(rotated_image)
        # plot_img(blue_rotated)

        # filtered_contours.append(hull)

    # return filtered_contours
    return train_centers, sticked_trains_mask


def count_color_trains( mask, single_area=500 ):
    # TODO
    pass
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        

def count_color_trains_and_score(mask, single_area=500):
    max_train_len = 8
    max_train_area = single_area*max_train_len
    scores = []
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(cnt) for cnt in contours]
    # score
    n_trains = 0
    score = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        count = area/single_area
        if count < 1.5:
            n_train += 1
            score += 1
        elif 1.5 <= count < 2.5:
            n_train += 2
            score += 2
        elif 2.5 <= count < 3.5:
            n_train += 3
            score += 4
        elif 3.5 <= count < 4.5:
            n_train += 4
            score += 7
        elif 4.5 <= count < 7:
            n_train += 6
            score += 15
        else:
            n_train += 8
            score += 21
    return n_trains, score


def count_color_trains( mask, single_area=500 ):
    # TODO
    pass
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        
    

def count_color_trains_and_score(mask, single_area=500):
    max_train_len = 8
    max_train_area = single_area*max_train_len
    scores = []
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(cnt) for cnt in contours]
    # score
    n_trains = 0
    score = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        count = area/single_area
        if count < 1.5:
            n_train += 1
            score += 1
        elif 1.5 <= count < 2.5:
            n_train += 2
            score += 2
        elif 2.5 <= count < 3.5:
            n_train += 3
            score += 4
        elif 3.5 <= count < 4.5:
            n_train += 4
            score += 7
        elif 4.5 <= count < 7:
            n_train += 6
            score += 15
        else:
            n_train += 8
            score += 21
    return n_trains, score

def predict_image(img: np.ndarray) -> (Union[np.ndarray, list], dict, dict):
    # raise NotImplementedError
    
    city_centers = find_cities_centers_opencv(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    # city_centers = np.int64([[1000, 2000], [1500, 3000], [1204, 3251]])
    n_trains = {'blue': 20, 'green': 30, 'black': 0, 'yellow': 30, 'red': 0}
    scores = {'blue': 60, 'green': 90, 'black': 0, 'yellow': 45, 'red': 0}
    return city_centers, n_trains, scores

