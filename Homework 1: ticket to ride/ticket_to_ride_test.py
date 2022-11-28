from typing import Union

from collections import defaultdict
from itertools import combinations

import numpy as np
import cv2 as cv
from skimage.transform import rescale
from skimage.measure import label, find_contours
from skimage.filters import gaussian
from scipy.spatial.distance import cdist
import scipy.stats as st
import json


COLORS = ('blue', 'green', 'black', 'yellow', 'red')
TRAINS2SCORE = {1: 1, 2: 2, 3: 4, 4: 7, 6: 15, 8: 21}

img = cv.imread('train_all.jpg')
# img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
CITY_TEMPL = img[197:260, 622:685]
CITY_TEMPL_GRAY = cv.cvtColor(CITY_TEMPL, cv.COLOR_BGR2GRAY)
# BLUE_TEMPLATE = img[2365:2395, 1995:2125]
BLUE_TEMPLATE = cv.imread('blue_template.jpg')
BLUE_TEMPLATE_GRAY = cv.cvtColor(BLUE_TEMPLATE, cv.COLOR_BGR2GRAY)
BLUE_ROAD_TEMPLATE1 = cv.imread('blue_road1.jpg')
BLUE_ROAD_TEMPLATE_GRAY1 = cv.cvtColor(BLUE_ROAD_TEMPLATE1, cv.COLOR_BGR2GRAY)
BLUE_ROAD_TEMPLATE2 = cv.imread('blue_road2.jpg')
BLUE_ROAD_TEMPLATE_GRAY2 = cv.cvtColor(BLUE_ROAD_TEMPLATE2, cv.COLOR_BGR2GRAY)
BLUE_ROAD_LABEL = cv.imread('blue_label.jpg')
BLUE_ROAD_LABEL_GRAY = cv.cvtColor(BLUE_ROAD_LABEL, cv.COLOR_BGR2GRAY)
print(BLUE_TEMPLATE.shape)

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

APPROX_CENTERS = load_json('train/all_centers.json')

def find_cities_centers_opencv( img_gray, method=cv.TM_CCOEFF_NORMED ):
    
    roi_win = 120
    match_centers = []
    w_templ = CITY_TEMPL.shape[1]
    h_templ = CITY_TEMPL.shape[0]
    for pt in APPROX_CENTERS:
        roi = img_gray[pt[0]-roi_win:pt[0]+roi_win, pt[1]-roi_win:pt[1]+roi_win]
        match = cv.matchTemplate(roi, CITY_TEMPL_GRAY, method  )
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(match)
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
    HSV = cv.cvtColor( img, cv.COLOR_BGR2HSV)
    mask = cv.inRange( HSV, lower_bound, upper_bound).astype(np.uint8)
    # 2 erosion etc
   

def find_blue_trains_mask( img_bgr, match_method=cv.TM_CCOEFF_NORMED, debug=False ):

    img = img_bgr.copy()
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    lower_bound = (90, 159, 90)
    upper_bound = (112, 255, 180)
    min_area = 800
    min_len = 80
    k_size = 3
    kernel = np.ones((k_size, k_size))
    roi_win = 80
    match_thresh = 0.17
    match_road_thresh = 0.5
    match_road_label_thresh = 0.45

    HSV = cv.cvtColor( img, cv.COLOR_BGR2HSV)
    mask = cv.inRange( HSV, lower_bound, upper_bound).astype(np.uint8)
    
    # mask = cv.morphologyEx( mask, cv.MORPH_CLOSE, kernel, iterations=3)
    mask = cv.morphologyEx( mask, cv.MORPH_ERODE, kernel, iterations=3)
    
    contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    filtered_contours = []
    trains_mask = np.zeros_like(mask, dtype=np.uint8)
    
    # Maybe I shoud firstly remove holes
    labelled, nlabels = label(mask, connectivity=2, return_num=True)
    if debug:
        cv.namedWindow('labelled', cv.WINDOW_NORMAL)
        cv.resizeWindow("labelled", 600, 500)
        cv.imshow('labelled', (labelled * 30).astype(np.uint8))
    train_centers = []
    if debug:
        img_hull = img_bgr.copy()
        cv.namedWindow('hull', cv.WINDOW_NORMAL)
        cv.resizeWindow("hull", 600, 500)
    if debug:
        det_rois = np.zeros_like(img_gray)

    for i, cnt in enumerate(contours):
        hull = cv.convexHull(cnt)
        area = cv.contourArea(hull)
        if area < min_area:
            continue
        rect = cv.minAreaRect(hull)
        center, dims, angle = rect
        rect_pts = np.int0(cv.boxPoints(rect))
        center = (int(center[0]), int(center[1]))
        if max(dims) < min_len:
            if debug:
                img_hull = cv.drawContours(img_hull, [rect_pts], 0, (0,150,255), 3)
            continue

        roi = img_gray[center[1]-roi_win:center[1]+roi_win, center[0]-roi_win:center[0]+roi_win].copy()
        height, width = roi.shape[:2]
        roi_center = (width/2, height/2)
        rect_pt1 = 
        rect_pt2 = 
        rect_angle = 
        rotate_matrix = cv.getRotationMatrix2D(center=roi_center, angle=90+angle, scale=1)
        roi_rotated = roi.copy()
        roi_rotated = cv.warpAffine(src=roi_rotated, M=rotate_matrix, dsize=(width, height))
        match = cv.matchTemplate(roi_rotated, BLUE_TEMPLATE_GRAY, method=match_method )
        _, max_val1, _, _ = cv.minMaxLoc(match)
        match_road = cv.matchTemplate(roi_rotated, BLUE_ROAD_TEMPLATE_GRAY1, method=match_method )
        _, max_val2, _, _ = cv.minMaxLoc(match_road)
        match_label = cv.matchTemplate(roi_rotated, BLUE_ROAD_LABEL_GRAY, method=match_method )
        _, max_val3, _, _ = cv.minMaxLoc(match_label)
        print(max_val1, max_val2, max_val3)
        if max_val1 < match_thresh:
            if debug:
                # cv.imshow(f'{i}', match)
                img_hull = cv.drawContours(img_hull, [rect_pts], 0, (0,150,255), 3)
                img_hull = cv.putText(img_hull, '0', center, cv.FONT_HERSHEY_SIMPLEX, 1,(0,0,255), 3)
                det_rois[center[1]-roi_win:center[1]+roi_win, center[0]-roi_win:center[0]+roi_win] = roi_rotated
            continue
        if max_val2 > match_road_thresh:
            if debug:
                # cv.imshow(f'{i}', match)
                img_hull = cv.drawContours(img_hull, [rect_pts], 0, (0,150,255), 3)
                img_hull = cv.putText(img_hull, '1', center, cv.FONT_HERSHEY_SIMPLEX, 1,(0,0,255), 3)
                det_rois[center[1]-roi_win:center[1]+roi_win, center[0]-roi_win:center[0]+roi_win] = roi_rotated
            continue
        if max_val3 > match_road_label_thresh:
            if debug:
                # cv.imshow(f'{i}', match)
                img_hull = cv.drawContours(img_hull, [rect_pts], 0, (0,150,255), 3)
                img_hull = cv.putText(img_hull, '2', center, cv.FONT_HERSHEY_SIMPLEX, 1,(0,0,255), 3)
                det_rois[center[1]-roi_win:center[1]+roi_win, center[0]-roi_win:center[0]+roi_win] = roi_rotated
            continue
        
        if debug:
            det_rois[center[1]-roi_win:center[1]+roi_win, center[0]-roi_win:center[0]+roi_win] = roi_rotated
        if debug:
            img_hull = cv.drawContours(img_hull, [hull], 0, (255,0,0), 3)
            img_hull = cv.drawContours(img_hull, [rect_pts], 0, (0,255,0), 3)
        trains_mask = cv.fillConvexPoly( trains_mask, hull, (255,255,255) );
        
        # label_id = labelled[center[1], center[0]]
        # if label_id != 0:
        #     train_mask = labelled == label_id
        #     trains_mask = np.logical_or(trains_mask, train_mask )
    
    cv.imshow('hull', img_hull)
    # trains_mask = 255 * trains_mask.astype(np.uint8)
    if debug:
        cv.namedWindow('train_blue_mask', cv.WINDOW_NORMAL)
        cv.resizeWindow('train_blue_mask', 600, 500)
        cv.imshow('train_blue_mask', trains_mask)
    if debug:
        cv.namedWindow('det_rois', cv.WINDOW_NORMAL)
        cv.resizeWindow('det_rois', 600, 500)
        cv.imshow('det_rois', det_rois)

    erode_ksize=3
    erode_kernel=np.ones((erode_ksize,erode_ksize))
    sticked_trains_mask = cv.morphologyEx(trains_mask, cv.MORPH_DILATE, erode_kernel, iterations=10)
    # sticked_trains_label = label(sticked_trains_mask, connectivity=2, return_num=True)
    

    # 5 create roi for each possible train and rotate template
        # roi = img[center[1]-roi_win:center[1]-roi_win, center[0]-roi_win:center[0]+roi_win].copy()
        # blue_rotated = cv.imread('images/blue_rotated.jpg')
        # height, width = blue_rotated.shape[:2]
        # center = (width/2, height/2)
        # rotate_matrix = cv.getRotationMatrix2D(center=center, angle=35, scale=1)
        # rotated_image = cv.warpAffine(src=blue_rotated, M=rotate_matrix, dsize=(width, height))

        # plot_img(rotated_image)
        # plot_img(blue_rotated)

        # filtered_contours.append(hull)

    # return filtered_contours
    return train_centers, sticked_trains_mask


def count_color_trains( mask, single_area=500 ):
    # TODO
    pass
    contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    count = 0
    for cnt in contours:
        area = cv.contourArea(cnt)
        

def count_color_trains_and_score(mask, single_area=500):
    max_train_len = 8
    max_train_area = single_area*max_train_len
    scores = []
    contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    areas = [cv.contourArea(cnt) for cnt in contours]
    # score
    n_trains = 0
    score = 0
    for cnt in contours:
        area = cv.contourArea(cnt)
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
    contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    count = 0
    for cnt in contours:
        area = cv.contourArea(cnt)
        
    

def count_color_trains_and_score(mask, single_area=500):
    max_train_len = 8
    max_train_area = single_area*max_train_len
    scores = []
    contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    areas = [cv.contourArea(cnt) for cnt in contours]
    # score
    n_trains = 0
    score = 0
    for cnt in contours:
        area = cv.contourArea(cnt)
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
    
    city_centers = find_cities_centers_opencv(cv.cvtColor(img, cv.COLOR_BGR2GRAY))
    # city_centers = np.int64([[1000, 2000], [1500, 3000], [1204, 3251]])
    n_trains = {'blue': 20, 'green': 30, 'black': 0, 'yellow': 30, 'red': 0}
    scores = {'blue': 60, 'green': 90, 'black': 0, 'yellow': 45, 'red': 0}
    return city_centers, n_trains, scores


if __name__ == '__main__':
    img_path = 'train_all.jpg'
    img = cv.imread(img_path)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    train_centers, sticked_trains_mask = find_blue_trains_mask(img, debug=True)

    cv.namedWindow('blue_templ', cv.WINDOW_NORMAL)
    cv.resizeWindow('blue_templ', 600, 500)
    cv.imshow('blue_templ', BLUE_TEMPLATE)

    cv.namedWindow('sticked_blue_mask', cv.WINDOW_NORMAL)
    cv.resizeWindow('sticked_blue_mask', 600, 500)
    cv.imshow('sticked_blue_mask', sticked_trains_mask)
    
    while True:
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    cv.destroyAllWindows()