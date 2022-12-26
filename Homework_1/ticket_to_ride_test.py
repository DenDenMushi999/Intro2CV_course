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
TRAIN_CASES = {i: fname for i, fname in enumerate(('all', 'black_blue_green', 'black_red_yellow',
                                                   'red_green_blue_inaccurate', 'red_green_blue'))}

img = cv.imread(f'train/{TRAIN_CASES[2]}.jpg')
# img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
# CITY_TEMPL = img[197:260, 622:685]
CITY_TEMPL = cv.imread('images/city_template.jpg')
CITY_TEMPL_GRAY = cv.cvtColor(CITY_TEMPL, cv.COLOR_BGR2GRAY)
# BLUE_TEMPLATE = img[2365:2395, 1995:2125]
TRAIN_TEMPLATES = { c : cv.imread(f'images/{c}_template.jpg') for c in COLORS }
TRAIN_TEMPLATES_GRAY = {c : cv.cvtColor(img, cv.COLOR_BGR2GRAY) for c, img in TRAIN_TEMPLATES.items()}
LABEL_TEMPLATES = { c : cv.imread(f'images/{c}_label_template.jpg') for c in COLORS}
LABEL_TEMPLATES_GRAY = {c : cv.cvtColor(img, cv.COLOR_BGR2GRAY) for c, img in LABEL_TEMPLATES.items()}

BLUE_TEMPLATE = cv.imread('images/blue_template.jpg')
BLUE_TEMPLATE_GRAY = cv.cvtColor(BLUE_TEMPLATE, cv.COLOR_BGR2GRAY)
BLUE_ROAD_TEMPLATE1 = cv.imread('images/blue_road1.jpg')
BLUE_ROAD_TEMPLATE_GRAY1 = cv.cvtColor(BLUE_ROAD_TEMPLATE1, cv.COLOR_BGR2GRAY)
BLUE_ROAD_TEMPLATE2 = cv.imread('images/blue_road2.jpg')
BLUE_ROAD_TEMPLATE_GRAY2 = cv.cvtColor(BLUE_ROAD_TEMPLATE2, cv.COLOR_BGR2GRAY)
# BLUE_ROAD_LABEL = cv.imread('images/blue_label.jpg')
# BLUE_ROAD_LABEL_GRAY = cv.cvtColor(BLUE_ROAD_LABEL, cv.COLOR_BGR2GRAY)

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

APPROX_CENTERS = load_json('train/all_centers.json')

# TODO
# fix padding issues

def filter_cities( centers ):
    # TODO
    pass


def find_city_centers_opencv( img_gray, method=cv.TM_CCOEFF_NORMED ):
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


def find_city_centers_wo_roi_opencv(img_gray, method=cv.TM_CCOEFF_NORMED, debug=False):
    threshold = 0.55

    match_centers = []
    w_templ = CITY_TEMPL.shape[1]
    h_templ = CITY_TEMPL.shape[0]
    match = cv.matchTemplate(img_gray, CITY_TEMPL_GRAY, method=method )

    loc = np.where( match > threshold )
    if debug:
        img_det = img_gray.copy()
    for pt in zip(*loc[::-1]):
        x = pt[0] + w_templ//2
        y = pt[1] + h_templ//2
        i = y; j = x
        match_centers.append([i, j])
        if debug:
            cv.circle(img_det, (x,y), w_templ, (0,0,255), 4)
    if debug:
        cv.namedWindow('det_centers', cv.WINDOW_NORMAL)
        cv.resizeWindow("det_centers", 600, 500)
        cv.imshow('det_centers', img_det)
    return np.int64(match_centers)


def color_filter( img_bgr, color ):
    assert color in COLORS
    img = img_bgr.copy()

    k_size = 3
    if color == 'black':
        k_size = 3
    kernel = np.ones((k_size, k_size))

    img = cv.medianBlur(img, k_size)

    if color == 'blue':
        lower_bound = (90, 159, 90)
        upper_bound = (112, 255, 180)
    if color == 'red':
        lower_bound = (140, 160, 70)
        upper_bound = (180, 255, 255)
        # lower_bound2 = (0, 180, 84)
        # upper_bound2 = (30, 255, 255)
    if color == 'green':
        lower_bound = (32, 111, 32)
        upper_bound = (90, 255, 255)
    if color == 'black':
        lower_bound = np.array([0, 0, 0])
        upper_bound = np.array([180, 255, 30])
    if color == 'yellow':
        lower_bound = np.array([15, 59, 110])
        upper_bound = np.array([40, 255, 255])

    if color == 'yellow':
        img = cv.cvtColor( img, cv.COLOR_BGR2HLS)
    else:
        img = cv.cvtColor( img, cv.COLOR_BGR2HSV)
    mask = cv.inRange( img, lower_bound, upper_bound).astype(np.uint8)

    if color == 'blue':
        mask = cv.morphologyEx( mask, cv.MORPH_ERODE, kernel, iterations=3)
    if color == 'red':
        mask = cv.morphologyEx( mask, cv.MORPH_ERODE, kernel, iterations=2)
        mask = cv.morphologyEx( mask, cv.MORPH_CLOSE, kernel, iterations=5)
    if color == 'green':
        mask = cv.morphologyEx( mask, cv.MORPH_ERODE, kernel, iterations=5)
        mask = cv.morphologyEx( mask, cv.MORPH_CLOSE, kernel, iterations=3)
    if color == 'black':
        # to eliminate black borders
        mask = cv.morphologyEx( mask, cv.MORPH_CLOSE, kernel, iterations=2)
        mask = cv.morphologyEx( mask, cv.MORPH_OPEN, kernel, iterations=3)
        # mask = cv.morphologyEx( mask, cv.MORPH_ERODE, kernel, iterations=3)

    return mask

def make_cities_mask(centers, shape, rad=60, dtype=np.uint8 ):
    cities_masks = np.zeros(shape[:2], dtype=dtype)
    for pt in centers:
        cities_masks = cv.circle(cities_masks, pt[::-1], rad, (255,255,255), -1)
    return cities_masks


def stick_trains( mask, color, city_centers, city_rad=60 ):

    dilate_ksize=3
    if color == 'green':
        iterations=10
        dilate_ksize=5
    elif color == 'red':
        iterations=20
    else:
        iterations=15

    dilate_kernel=np.ones((dilate_ksize,dilate_ksize))
    # sticked_trains_mask = cv.morphologyEx(mask, cv.MORPH_DILATE, dilate_kernel, iterations=iterations)
    sticked_trains_mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, dilate_kernel, iterations=iterations)
    cities_mask = make_cities_mask( city_centers, sticked_trains_mask.shape, rad=city_rad )
    cities_mask = cv.bitwise_not( cities_mask )
    return cv.bitwise_and( sticked_trains_mask, cities_mask )


def cvt_train_rot_angle(rect_pts, angle):
    if np.linalg.norm((rect_pts[0] - rect_pts[3])) > np.linalg.norm((rect_pts[0] - rect_pts[1])):
        return angle
    else:
        return angle -90

def rotate_img( img, angle, scale=1 ):
    img_ = img.copy()

    height, width = img_.shape[:2]
    img_center = (width/2, height/2)

    rotate_matrix = cv.getRotationMatrix2D(center=img_center, angle=angle, scale=1)
    return cv.warpAffine(src=img_, M=rotate_matrix, dsize=(width, height))


def find_trains_mask( img_bgr, color, match_method=cv.TM_CCOEFF_NORMED, debug=False ):
    assert color in COLORS

    img = img_bgr.copy()
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    if color == 'red':
        min_area = 700
    else:
        min_area = 1500
    min_width = 80
    min_height = 15
    k_size = 3
    kernel = np.ones((k_size, k_size))
    roi_win = 80

    match_thresh = 0.4
    if color == 'green':
        match_road_thresh = 0.4
        match_road_label_thresh = 0.4
    elif color == 'black':
        match_road_thresh = 0.6
        match_road_label_thresh = 0.6
    elif color == 'yellow':
        match_thresh = 0.95
        match_road_thresh = 0.8
        match_road_label_thresh = 0.8
    else:
        match_road_thresh = 0.5
        match_road_label_thresh = 0.5

    bg_frame_size = 80
    assert bg_frame_size >= roi_win

    mask = color_filter(img, color)

    contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    contours = [cnt for cnt,hier in zip(contours, hierarchy[0]) if hier[3]==-1]
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
        epsilon = 0.005*cv.arcLength(cnt,True)
        poly = cv.approxPolyDP(cnt, epsilon, True)
        poly_area = cv.contourArea(poly)
        if area < min_area or poly_area < min_area:
            continue
        rect = cv.minAreaRect(hull)
        center, dims, angle = rect

        rect_pts = np.int0(cv.boxPoints(rect))
        center = (int(center[0]), int(center[1]))
        if (max(dims) < min_width) or (min(dims) < min_height):
            if debug:
                img_hull = cv.drawContours(img_hull, [rect_pts], 0, (0,150,255), 3)
            continue

        if min(np.abs(center[0] - img_gray.shape[1]), center[0], np.abs(center[1] - img_gray.shape[0]), center[1]) < bg_frame_size:
            continue
        roi = img_gray[center[1]-roi_win:center[1]+roi_win, center[0]-roi_win:center[0]+roi_win]
        rot_angle = cvt_train_rot_angle(rect_pts, angle)
        roi_rotated = rotate_img(roi, rot_angle)
        roi_rotated2 = rotate_img(roi, -rot_angle+180)

        match = cv.matchTemplate(roi_rotated, TRAIN_TEMPLATES_GRAY[color], method=match_method )
        _, max_val1, _, _ = cv.minMaxLoc(match)
        match_road = cv.matchTemplate(roi_rotated, BLUE_ROAD_TEMPLATE_GRAY1, method=match_method )
        _, max_val2, _, _ = cv.minMaxLoc(match_road)
        match_label = cv.matchTemplate(roi_rotated, LABEL_TEMPLATES_GRAY[color], method=match_method )
        match_label2 =  cv.matchTemplate(roi_rotated2, LABEL_TEMPLATES_GRAY[color], method=match_method )
        _, max_val3, _, _ = cv.minMaxLoc(match_label)
        _, max_val32, _, _ = cv.minMaxLoc(match_label2)
        max_val3 = max(max_val3, max_val32)
        if debug:
            print(max_val1, max_val2, max_val3)


        if color == 'red':
            is_train = True
        else:
            is_train = max_val1 > match_thresh
        is_road = max_val2 > match_road_thresh
        is_label = max_val3 > match_road_label_thresh

        if debug:
            det_rois[center[1]-roi_win:center[1]+roi_win, center[0]-roi_win:center[0]+roi_win] = roi_rotated

        if (not is_train) and (is_road or is_label):
            if debug:
                # cv.imshow(f'{i}', match)
                img_hull = cv.drawContours(img_hull, [rect_pts], 0, (0,150,255), 3)
                if not is_train:
                    img_hull = cv.putText(img_hull, '0', center, cv.FONT_HERSHEY_SIMPLEX, 1,(0,0,255), 3)
                if is_road:
                    img_hull = cv.putText(img_hull, '1', center, cv.FONT_HERSHEY_SIMPLEX, 1,(0,0,255), 3)
                if is_label:
                    img_hull = cv.putText(img_hull, '2', center, cv.FONT_HERSHEY_SIMPLEX, 1,(0,0,255), 3)
            continue


        if debug:
            img_hull = cv.drawContours(img_hull, [poly], 0, (255,0,0), 3)
            img_hull = cv.drawContours(img_hull, [rect_pts], 0, (0,255,0), 3)
            if not is_train:
                img_hull = cv.putText(img_hull, '0', center, cv.FONT_HERSHEY_SIMPLEX, 1,(0,0,255), 3)
            if is_road:
                img_hull = cv.putText(img_hull, '1', center, cv.FONT_HERSHEY_SIMPLEX, 1,(0,0,255), 3)
            if is_label:
                img_hull = cv.putText(img_hull, '2', center, cv.FONT_HERSHEY_SIMPLEX, 1,(0,0,255), 3)
        trains_mask = cv.fillConvexPoly( trains_mask, poly, (255,255,255) );

        # label_id = labelled[center[1], center[0]]
        # if label_id != 0:
        #     train_mask = labelled == label_id
        #     trains_mask = np.logical_or(trains_mask, train_mask )

    if debug:
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

    # sticked_trains_label = label(sticked_trains_mask, connectivity=2, return_num=True)

    return train_centers, trains_mask


def count_color_trains( mask, single_area=8000 ):
    # TODO
    pass
    contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    count = 0
    for cnt in contours:
        area = cv.contourArea(cnt)



def count_color_trains_and_score(mask, color, debug=False):
    max_train_len = 8

    if color == 'blue':
        single_area = 3000
    elif color == 'black':
        single_area = 5000
    elif color == 'yellow':
        single_area = 4000
    else:
        single_area = 3500

    max_train_area = single_area*max_train_len
    contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    n_trains = [None]*len(contours)
    # areas = [cv.contourArea(cnt) for cnt in contours]
    # score
    n_trains_all = 0
    score = 0
    if debug:
        det_ntrains_img = mask.copy()[:,:,np.newaxis]
        det_ntrains_img = np.tile(det_ntrains_img, (1,1,3))
    for i, cnt in enumerate(contours):
        area = cv.contourArea(cnt)
        count = area/single_area
        if count < 1.5:
            dn = 1
            ds = 1
        elif 1.5 <= count < 2.5:
            dn = 2
            ds = 2
        elif 2.5 <= count < 3.5:
            dn = 3
            ds = 4
        elif 3.5 <= count < 4.5:
            dn = 4
            ds = 7
        elif 4.5 <= count < 7:
            dn = 6
            ds = 15
        else:
            dn = 8
            ds = 21
        score += ds
        n_trains_all += dn
        n_trains[i] = dn
        if debug:
            M = cv.moments(cnt)
            cx = int(M['m10']/(M['m00']+1e-3))
            cy = int(M['m01']/(M['m00']+1e-3))
            det_ntrains_img = cv.putText(det_ntrains_img, f'{dn},{area}', (cx, cy), cv.FONT_HERSHEY_SIMPLEX, 3,(255,255,0), 3)
    if debug:
        cv.namedWindow('det_n_trains', cv.WINDOW_NORMAL)
        cv.resizeWindow("det_n_trains", 600, 500)
        cv.imshow('det_n_trains', det_ntrains_img.astype(np.uint8))
    return n_trains_all, score


def predict_image(img: np.ndarray) -> (Union[np.ndarray, list], dict, dict):
    # raise NotImplementedError

    city_centers = find_city_centers_wo_roi_opencv(cv.cvtColor(img, cv.COLOR_BGR2GRAY))
    # city_centers = np.int64([[1000, 2000], [1500, 3000], [1204, 3251]])
    scores = {}
    n_trains = {}

    for c in COLORS:
        train_centers, trains_mask = find_trains_mask(img, color=c)
        sticked_trains_mask = stick_trains(trains_mask, c, city_centers,60)
        ntrains, score = count_color_trains_and_score( sticked_trains_mask, color=c )
        n_trains[c] = ntrains
        scores[c] = score

    # n_trains = {'blue': 20, 'green': 30, 'black': 0, 'yellow': 30, 'red': 0}
    # scores = {'blue': 60, 'green': 90, 'black': 0, 'yellow': 45, 'red': 0}
    return city_centers, n_trains, scores


if __name__ == '__main__':
    img_path = f'train/{TRAIN_CASES[0]}.jpg'
    img = cv.imread(img_path)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    color = 'yellow'
    # city_centers = find_city_centers_opencv(img_gray)
    city_centers = find_city_centers_wo_roi_opencv(img_gray, debug=True)
    train_centers, trains_mask = find_trains_mask(img, color=color, debug=True)
    sticked_trains_mask = stick_trains(trains_mask, color, city_centers, 60)
    print(city_centers)

    det_centes_img = img.copy()
    for pt in city_centers:
        det_centes_img = cv.circle(det_centes_img, pt[::-1], 30, (255,255,255), 4)

    blue_ntrains, blue_score = count_color_trains_and_score( sticked_trains_mask, color=color, debug=True )
    print(blue_ntrains, blue_score)
    cv.namedWindow('city_centers', cv.WINDOW_NORMAL)
    cv.resizeWindow('city_centers', 600, 500)
    cv.imshow('city_centers', det_centes_img)

    cv.namedWindow('sticked_blue_mask', cv.WINDOW_NORMAL)
    cv.resizeWindow('sticked_blue_mask', 600, 500)
    cv.imshow('sticked_blue_mask', sticked_trains_mask)

    while True:
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    cv.destroyAllWindows()