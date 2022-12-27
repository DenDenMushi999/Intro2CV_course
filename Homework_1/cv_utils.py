from typing import Union, List
import json

import numpy as np
import cv2 as cv
from skimage.filters import gaussian
from skimage.measure import label

from utils import empty, load_json

# TODO
# fix padding issues
# create each time you need to use its functions

COLORS = ('blue', 'green', 'black', 'yellow', 'red')
TRAINS2SCORE = {1: 1, 2: 2, 3: 4, 4: 7, 6: 15, 8: 21}
TRAIN_CASES = {i: fname for i, fname in enumerate(('all', 'black_blue_green', 'black_red_yellow',
                                                   'red_green_blue_inaccurate', 'red_green_blue'))}
APPROX_CENTERS = load_json('train/all_centers.json')

# IMG = cv.imread(f'train/{TRAIN_CASES[2]}.jpg')
# IMG = cv.cvtColor(IMG, cv.COLOR_BGR2RGB)

# CITY_TEMPL = IMG[197:260, 622:685]
CITY_TEMPL = cv.imread('images/city_template.jpg')
CITY_TEMPL_GRAY = cv.cvtColor(CITY_TEMPL, cv.COLOR_BGR2GRAY)

TRAIN_TEMPLATES = { c : cv.imread(f'images/{c}_template.jpg') for c in COLORS }
TRAIN_TEMPLATES_GRAY = {c : cv.cvtColor(IMG, cv.COLOR_BGR2GRAY) for c, IMG in TRAIN_TEMPLATES.items()}

# BLUE_TEMPLATE = IMG[2365:2395, 1995:2125]
BLUE_TEMPLATE = cv.imread('images/blue_template.jpg')
BLUE_TEMPLATE_GRAY = cv.cvtColor(BLUE_TEMPLATE, cv.COLOR_BGR2GRAY)

LABEL_TEMPLATES = { c : cv.imread(f'images/{c}_label_template.jpg') for c in COLORS}
LABEL_TEMPLATES_GRAY = {c : cv.cvtColor(IMG, cv.COLOR_BGR2GRAY) for c, IMG in LABEL_TEMPLATES.items()}

BLUE_ROAD_TEMPLATE1 = cv.imread('images/blue_road1.jpg')
BLUE_ROAD_TEMPLATE_GRAY1 = cv.cvtColor(BLUE_ROAD_TEMPLATE1, cv.COLOR_BGR2GRAY)
BLUE_ROAD_TEMPLATE2 = cv.imread('images/blue_road2.jpg')
BLUE_ROAD_TEMPLATE_GRAY2 = cv.cvtColor(BLUE_ROAD_TEMPLATE2, cv.COLOR_BGR2GRAY)
# BLUE_ROAD_LABEL = cv.imread('images/blue_label.jpg')
# BLUE_ROAD_LABEL_GRAY = cv.cvtColor(BLUE_ROAD_LABEL, cv.COLOR_BGR2GRAY)

class CityDetector:

    def __init__(self):
        self.threshold = 0.57
        self.roi_win = 120
        self.min_dist = 100

    def find_city_centers_wo_roi_opencv( self, img_gray, method=cv.TM_CCOEFF_NORMED, debug=False):
        match_centers = []
        w_templ = CITY_TEMPL.shape[1]
        h_templ = CITY_TEMPL.shape[0]
        match = cv.matchTemplate(img_gray, CITY_TEMPL_GRAY, method=method )

        loc = np.where( match > self.threshold )
        if debug:
            img_det = img_gray.copy()
        for pt in zip(*loc[::-1]):
            x = pt[0] + w_templ//2
            y = pt[1] + h_templ//2
            i = y; j = x
            match_centers.append([i, j])
            # if debug:
                # cv.circle(img_det, (x,y), w_templ, (0,0,255), 4)
        match_centers = self.remove_outliers(match_centers)

        if debug:
            for c in match_centers:
                cv.circle(img_det, c[::-1], w_templ, (0,255,0), 4)
            cv.namedWindow('det_centers', cv.WINDOW_NORMAL)
            cv.resizeWindow("det_centers", 600, 500)
            cv.imshow('det_centers', img_det)
        return np.int64(match_centers)

    def find_city_centers_opencv( self, img_gray, method=cv.TM_CCOEFF_NORMED ):
        match_centers = []
        w_templ = CITY_TEMPL.shape[1]
        h_templ = CITY_TEMPL.shape[0]
        for pt in APPROX_CENTERS:
            roi = img_gray[pt[0]-self.roi_win:pt[0]+self.roi_win, pt[1]-self.roi_win:pt[1]+self.roi_win]
            match = cv.matchTemplate(roi, CITY_TEMPL_GRAY, method  )
            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(match)
            x = max_loc[0] + w_templ // 2 + pt[1] - self.roi_win
            y = max_loc[1] + h_templ // 2 + pt[0] - self.roi_win
            i = y; j = x
            match_centers.append([i, j])

        return np.int64(match_centers)

    def remove_outliers(self, centers):

        centers = np.array(centers)
        filtered_centers = centers.copy()
        index = 0
        while index != len(filtered_centers):
            val = filtered_centers[index]
            filtered_centers = filtered_centers[ np.linalg.norm(filtered_centers - val, axis=1) > self.min_dist]
            filtered_centers = np.insert(filtered_centers, index, val, axis=0)
            index += 1
        return filtered_centers

class ColorFilter():
    def __init__(self, color):
        self.colors = COLORS
        self.color = color

        self.k_size = 3
        if color == 'black':
            self.k_size = 3
        self.kernel = np.ones((self.k_size, self.k_size))

        if color == 'blue':
            self.lower_bound = (90, 159, 90)
            self.upper_bound = (112, 255, 180)
        if color == 'red':
            self.lower_bound = (140, 160, 70)
            self.upper_bound = (180, 255, 255)
            # self.lower_bound2 = (0, 180, 84)
            # self.upper_bound2 = (30, 255, 255)
        if color == 'green':
            self.lower_bound = (32, 111, 32)
            self.upper_bound = (90, 255, 255)
        if color == 'black':
            self.lower_bound = np.array([0, 0, 0])
            self.upper_bound = np.array([180, 255, 30])
        if color == 'yellow':
            self.lower_bound = np.array([15, 59, 110])
            self.upper_bound = np.array([40, 255, 255])

    def color_filter(self, img):
        img = cv.medianBlur(img, self.k_size)

        if self.color == 'yellow':
            img = cv.cvtColor( img, cv.COLOR_BGR2HLS)
        else:
            img = cv.cvtColor( img, cv.COLOR_BGR2HSV)
        mask = cv.inRange( img, self.lower_bound, self.upper_bound).astype(np.uint8)

        if self.color == 'blue':
            mask = cv.morphologyEx( mask, cv.MORPH_ERODE, self.kernel, iterations=3)
        if self.color == 'red':
            mask = cv.morphologyEx( mask, cv.MORPH_ERODE, self.kernel, iterations=2)
            mask = cv.morphologyEx( mask, cv.MORPH_CLOSE, self.kernel, iterations=5)
        if self.color == 'green':
            mask = cv.morphologyEx( mask, cv.MORPH_ERODE, self.kernel, iterations=5)
            mask = cv.morphologyEx( mask, cv.MORPH_CLOSE, self.kernel, iterations=3)
        if self.color == 'black':
            # to eliminate black borders
            mask = cv.morphologyEx( mask, cv.MORPH_CLOSE, self.kernel, iterations=2)
            mask = cv.morphologyEx( mask, cv.MORPH_OPEN, self.kernel, iterations=3)
            # mask = cv.morphologyEx( mask, cv.MORPH_ERODE, self.kernel, iterations=3)

        return mask


class TrainDetector:
    def __init__(self, color):
        assert color in COLORS

        self.color_filter = ColorFilter(color)

        self.color = color
        if color == 'red':
            self.min_area = 700
        else:
            self.min_area = 1500
        self.min_width = 80
        self.min_height = 15
        self.k_size = 3
        self.kernel = np.ones((self.k_size, self.k_size))
        self.roi_win = 80

        self.match_thresh = 0.4
        if color == 'green':
            self.match_road_thresh = 0.4
            self.match_road_label_thresh = 0.4
        elif color == 'black':
            self.match_road_thresh = 0.6
            self.match_road_label_thresh = 0.6
        elif color == 'yellow':
            self.match_thresh = 0.95
            self.match_road_thresh = 0.8
            self.match_road_label_thresh = 0.8
        else:
            self.match_road_thresh = 0.5
            self.match_road_label_thresh = 0.5

        self.bg_frame_size = 80
        assert self.bg_frame_size >= self.roi_win

    def convert_train_rot_angle(self, rect_pts, angle):
        if np.linalg.norm((rect_pts[0] - rect_pts[3])) > np.linalg.norm((rect_pts[0] - rect_pts[1])):
            return angle
        else:
            return angle -90

    def rotate_img( self, img, angle, scale=1 ):
        img_ = img.copy()

        height, width = img_.shape[:2]
        img_center = (width/2, height/2)

        rotate_matrix = cv.getRotationMatrix2D(center=img_center, angle=angle, scale=1)
        return cv.warpAffine(src=img_, M=rotate_matrix, dsize=(width, height))

    def find_one_color_trains_mask(self, img_bgr, color, match_method=cv.TM_CCOEFF_NORMED, debug=False):
        img = img_bgr.copy()
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        mask = self.color_filter.color_filter(img)

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
            if area < self.min_area or poly_area < self.min_area:
                continue
            rect = cv.minAreaRect(hull)
            center, dims, angle = rect

            rect_pts = np.int0(cv.boxPoints(rect))
            center = (int(center[0]), int(center[1]))
            if (max(dims) < self.min_width) or (min(dims) < self.min_height):
                if debug:
                    img_hull = cv.drawContours(img_hull, [rect_pts], 0, (0,150,255), 3)
                continue

            if min(np.abs(center[0] - img_gray.shape[1]), center[0], np.abs(center[1] - img_gray.shape[0]), center[1]) < self.bg_frame_size:
                continue
            roi = img_gray[center[1]-self.roi_win:center[1]+self.roi_win, center[0]-self.roi_win:center[0]+self.roi_win]
            rot_angle = self.convert_train_rot_angle(rect_pts, angle)
            roi_rotated = self.rotate_img(roi, rot_angle)
            roi_rotated2 = self.rotate_img(roi, -rot_angle+180)

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
                is_train = max_val1 > self.match_thresh
            is_road = max_val2 > self.match_road_thresh
            is_label = max_val3 > self.match_road_label_thresh

            if debug:
                det_rois[center[1]-self.roi_win:center[1]+self.roi_win, center[0]-self.roi_win:center[0]+self.roi_win] = roi_rotated

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


class TrainsSticker:
    def __init__(self,color):
        self.color = color
        self.dilate_ksize=3
        if color == 'green':
            self.iterations=10
            self.dilate_ksize=5
        elif color == 'red':
            self.iterations=20
        else:
            self.iterations=15
        self.dilate_kernel=np.ones((self.dilate_ksize,self.dilate_ksize))

    def make_cities_mask(self, centers, img_shape, rad=60, dtype=np.uint8 ):
        cities_masks = np.zeros(img_shape[:2], dtype=dtype)
        for pt in centers:
            cities_masks = cv.circle(cities_masks, pt[::-1], rad, (255,255,255), -1)
        return cities_masks

    def stick_trains( self, mask, color, city_centers, city_rad=60 ):
        # sticked_trains_mask = cv.morphologyEx(mask, cv.MORPH_DILATE, dilate_kernel, iterations=iterations)
        sticked_trains_mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, self.dilate_kernel, iterations=self.iterations)
        cities_mask = self.make_cities_mask( city_centers, img_shape=sticked_trains_mask.shape, rad=city_rad )
        cities_mask = cv.bitwise_not( cities_mask )
        return cv.bitwise_and( sticked_trains_mask, cities_mask )


class TrainsCounterOneColor:
    def __init__(self, color):
        self.max_train_len = 8
        self.color = color

        if color == 'blue':
            self.single_len = 300
        elif color == 'black':
            self.single_len = 500
        elif color == 'yellow':
            self.single_len = 350
        else:
            self.single_len = 350

    def count_color_trains( self, mask):
        pass
        # TODO
        # contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        # count = 0
        # for cnt in contours:
        #     area = cv.arcLength(cnt, closed=True)

    def count_color_trains_and_score( self, mask, debug=False):
        all_trains_max_len = self.single_len*self.max_train_len
        contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        n_trains = [None]*len(contours)
        # lens = [cv.contourlen(cnt) for cnt in contours]
        # score
        n_trains_all = 0
        score = 0
        if debug:
            det_ntrains_img = mask.copy()[:,:,np.newaxis]
            det_ntrains_img = np.tile(det_ntrains_img, (1,1,3))
        for i, cnt in enumerate(contours):
            length = cv.arcLength(cnt, closed=True)
            count = length/self.single_len
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
                det_ntrains_img = cv.putText(det_ntrains_img, f'{dn},{length:.3f}', (cx, cy), cv.FONT_HERSHEY_SIMPLEX, 3,(255,255,0), 3)
        if debug:
            cv.namedWindow('det_n_trains', cv.WINDOW_NORMAL)
            cv.resizeWindow("det_n_trains", 600, 500)
            cv.imshow('det_n_trains', det_ntrains_img.astype(np.uint8))
        return n_trains_all, score


class TicketToRideHandler:

    def __init__(self):
        pass

    def filter_cities(self,centers):
        # TODO
        pass

    def find_city_centers_wo_roi_opencv(self, img_gray, method=cv.TM_CCOEFF_NORMED, debug=False):
        city_detector = CityDetector()
        return city_detector.find_city_centers_wo_roi_opencv(img_gray, method=method, debug=debug)

    def find_city_centers_opencv( self, img_gray, method=cv.TM_CCOEFF_NORMED ):
        city_detector = CityDetector()
        return city_detector.find_city_centers_opencv(img_gray, method=method)

    def color_filter( self, img_bgr, color ):
        color_filter = ColorFilter(color)
        return color_filter.color_filter(img_bgr)

    def stick_trains( self, mask, color, city_centers, city_rad=60 ):
        trains_sticker = TrainsSticker(color)
        return trains_sticker.stick_trains(mask, color, city_centers, city_rad)

    def find_one_color_trains_mask(self, img_bgr, color, match_method=cv.TM_CCOEFF_NORMED, debug=False):
        train_detector = TrainDetector(color)
        return train_detector.find_one_color_trains_mask(img_bgr, color, match_method=cv.TM_CCOEFF_NORMED, debug=debug)

    def count_color_trains_and_score(self, mask, color, debug=False):
        trains_counter = TrainsCounterOneColor(color)
        return trains_counter.count_color_trains_and_score(mask, debug=debug)

    def predict_image(self, img: np.ndarray) -> (Union[np.ndarray, list], dict, dict):

        city_centers = self.find_city_centers_wo_roi_opencv(cv.cvtColor(img, cv.COLOR_BGR2GRAY))
        # city_centers = np.int64([[1000, 2000], [1500, 3000], [1204, 3251]])
        scores = {}
        n_trains = {}

        for c in COLORS:
            train_centers, trains_mask = self.find_one_color_trains_mask(img, color=c)
            sticked_trains_mask = self.stick_trains(trains_mask, c, city_centers,60)
            ntrains, score = self.count_color_trains_and_score( sticked_trains_mask, color=c )
            n_trains[c] = ntrains
            scores[c] = score

        # n_trains = {'blue': 20, 'green': 30, 'black': 0, 'yellow': 30, 'red': 0}
        # scores = {'blue': 60, 'green': 90, 'black': 0, 'yellow': 45, 'red': 0}
        return city_centers, n_trains, scores
