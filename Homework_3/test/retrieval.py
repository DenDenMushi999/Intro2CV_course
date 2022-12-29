import json
from typing import List

import cv2 as cv
import numpy as np

# from utils import CalibrateParametersGUI
# from product_detector import FeatureMatcher, ProductDetectorOnFeatures
# from product_detector import TRAIN_IMAGES, TEMPLATE_IMAGES

TEMPLATE_FNAME_PREFIXES = ['0_0', '0_1', '1', '2', '3', 'extreme']
TRAIN_FNAME_PREFIXES = ['0', '1', '2', '3', 'extreme']

PATH = '/home/dendav/projects/Intro2CV_course/Homework_3'
TEMPLATE_IMAGES = [cv.imread(f'{PATH}/train/template_{prefx}.jpg') for prefx in TEMPLATE_FNAME_PREFIXES]
TRAIN_IMAGES = [cv.imread(f'{PATH}/train/train_{prefx}.jpg') for prefx in TRAIN_FNAME_PREFIXES]

def empty(a):
    pass

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def imshow(win_name, img):
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    cv.resizeWindow(win_name, 800, 600)
    cv.imshow(win_name, img)

def create_win(win_name):
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    cv.resizeWindow(win_name, 800, 600)

def CalibrateParametersGUI( cls, init_params: List, func_name, var_names, var_coeffs, max_var_vals, *args, **kwargs ):
    # print(args, kwargs)
    instance = cls(*init_params)
    win_name = f'tune_{cls.__name__}.{func_name}'
    cv.namedWindow(win_name)

    for var_name, coef, max_val in zip(var_names, var_coeffs, max_var_vals):
        cv.createTrackbar(var_name, win_name, int(getattr(instance, var_name)*coef), max_val, empty)

    old_vals = [None]*len(var_names)
    new_vals = [None]*len(var_names)
    while True:
        old_vals = new_vals.copy()
        key = cv.waitKey(1000) & 0xFF
        if key == ord('q'):
            break
        for i, (var_name, coef) in enumerate(zip(var_names,var_coeffs)):
            new_vals[i] = cv.getTrackbarPos(var_name, win_name)/coef
        if new_vals == old_vals:
            continue
        else:
            for var_name, val in zip(var_names, new_vals):
                if var_name in dir(instance):
                    # for case of complex class fields that changes other fields
                    try:
                        getattr(instance, f'set_{var_name}')(val)
                        # print(f'found setter for {var_name}')
                    except AttributeError:
                        setattr(instance, var_name, val)
                        # print(f'not found setter for {var_name}')
                else :
                    raise AttributeError(f'Obejct of class \'{cls.__name__}\' has no attribute \'{var_name}\'')

        getattr(instance, func_name)(*args, **kwargs)

    # print(f'Final values of {cls.__name__}.{func_name}():')
    for var_name in var_names:
        # print(f'{var_name}: {getattr(instance, var_name)}')
        pass

def remove_object_from_img(img, bbox, scaled=False):
    if scaled:
        bbox = scale_bbox_inv(bbox, img.shape[:2])
    x_min, y_min, w, h = bbox
    img[y_min:y_min+h, x_min:x_min+w] = 0
    return img

def scale_bbox(bbox, img_shape):
    bbox = list(bbox)
    bbox[0] /= img_shape[1]
    bbox[1] /= img_shape[0]
    bbox[2] /= img_shape[1]
    bbox[3] /= img_shape[0]
    return tuple(bbox)

def scale_bbox_inv(bbox, img_shape):
    bbox = list(bbox)
    bbox[0] *= img_shape[1]
    bbox[1] *= img_shape[0]
    bbox[2] *= img_shape[1]
    bbox[3] *= img_shape[0]
    return bbox


class ROI:
    def __init__(self, img, bbox):
        self.bbox = bbox
        x_min, y_min, w, h = bbox
        self.img = img[y_min:y_min+h, x_min:x_min+w]


class ROIHandler:
    def __init__(self, img, w_roi, h_roi, stride_x, stride_y, min_w_roi, min_h_roi):
        self.img = img
        self.w_roi = w_roi
        self.h_roi = h_roi
        self.stride_x = stride_x
        self.stride_y = stride_y

        assert min_w_roi < w_roi and min_h_roi < h_roi
        self.min_w_roi = min_w_roi
        self.min_h_roi = min_h_roi
        self.cur_x = 0
        self.cur_y = 0
        self.roi = ROI(img, (0,0, w_roi, h_roi))
        self.cur_roi_num = 0

    @property
    def is_end(self):
        return self.is_y_end

    @property
    def is_x_end(self):
        return self.roi.bbox[0] + self.roi.bbox[2] >= self.img.shape[1] \
            or self.roi.img.shape[1] < self.min_w_roi

    @property
    def is_y_end(self):
        return self.roi.bbox[1] + self.roi.bbox[3] >= self.img.shape[0] \
            or self.roi.img.shape[0] < self.min_h_roi

    def cvt_roi_bbox_to_img(self, bbox):
        # bbox = (x_min, y_min, w, h)
        bbox = list(bbox)
        bbox[0] += self.roi.bbox[0]
        bbox[1] += self.roi.bbox[1]
        return tuple(bbox)

    def next_roi(self):
        if self.cur_roi_num != 0 and  not self.is_end:
            new_bbox = list(self.roi.bbox)
            if self.is_x_end:
                # print('roi_handler: end of x')
                new_bbox[0] = 0
                new_bbox[1] += self.stride_y
            else:
                new_bbox[0] += self.stride_x
            self.roi = ROI(self.img, tuple(new_bbox))
        self.cur_roi_num += 1
        return self.roi

class FeatureMatcher:

    def __init__(self):
        self.k_neighbours = 2
        self.FLANN_INDEX_KDTREE = 0
        self.index_trees = 5
        self.index_params = dict(algorithm = self.FLANN_INDEX_KDTREE, trees=self.index_trees)
        self.search_checks = 50
        self.search_params = dict(checks=self.search_checks)   # or pass empty dictionary
        self.matcher = cv.FlannBasedMatcher(self.index_params, self.search_params)

        self.lowe = 0.75
        self.min_match_count = 10

    def set_FLANN_INDEX_KDTREE(self, val):
        val = int(val) if val > 0 else 1
        self.FLANN_INDEX_KDTREE = val
        self.index_params = dict(algorithm = self.FLANN_INDEX_KDTREE, trees = val)
        self.matcher = cv.FlannBasedMatcher(self.index_params, self.search_params)

    def set_index_trees(self, trees):
        trees = int(trees) if trees > 0 else 1
        self.index_trees = trees
        self.index_params = dict(algorithm = self.FLANN_INDEX_KDTREE, trees = trees)
        self.matcher = cv.FlannBasedMatcher(self.index_params, self.search_params)

    def set_search_checks(self, checks):
        checks = int(checks) if checks > 0 else 1
        self.search_checks = checks
        self.search_params = dict(checks=checks)   # or pass empty dictionary
        self.matcher = cv.FlannBasedMatcher(self.index_params, self.search_params)

    def match(self, descr1, descr2, debug=False, img=None, kp_img=None, query=None, kp_query=None ):
        if not np.any(descr1) or not np.any(descr2):
            return []
        if len(descr1) < self.k_neighbours or len(descr2) < self.k_neighbours:
            return []
        matches = self.matcher.knnMatch(descr1, descr2, self.k_neighbours)
        if not debug:
            return matches
        else:
            assert (img is not None) and (kp_img is not None) and (query is not None) and (kp_query is not None)
            matches_mask = [[1,0] for i in range(len(matches))]
            draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matches_mask,
                   flags = cv.DrawMatchesFlags_DEFAULT)
            img_matches = cv.drawMatchesKnn(query, kp_query, img, kp_img, matches, None, **draw_params)
            imshow('img_matches', img_matches)
            return matches

    def filter_matches(self, matches, debug=False, img=None, kp_img=None, query=None, kp_query=None):
        filtered_matches = []
        if not matches:
            return filtered_matches, []
        matches_mask = [[0,0] for i in range(len(matches))]
        for i,(m, n) in enumerate(matches):
            if m.distance < self.lowe*n.distance:
                matches_mask[i]=[1,0]
                filtered_matches.append(m)
        if not debug:
            return filtered_matches, matches_mask
        else:
            assert (img is not None) and (kp_img is not None) and (query is not None) and (kp_query is not None)
            draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matches_mask,
                   flags = cv.DrawMatchesFlags_DEFAULT)
            img_matches = cv.drawMatchesKnn(query, kp_query, img, kp_img, matches, None, **draw_params)
            imshow('img_filtered_matches', img_matches)
            return filtered_matches, matches_mask


class ProductDetectorOnFeatures:
    def __init__(self, img, query):
        self.img_orig = img
        self.query = query

        self.detector = cv.SIFT_create()
        self.matcher = FeatureMatcher()

        self.stride_x = int(2/3*query.shape[1])
        self.stride_y = int(2/3*query.shape[0])
        self.h_roi = query.shape[0]*3
        self.w_roi = query.shape[1]*3
        self.roi_handler = ROIHandler(img, self.w_roi, self.h_roi, self.stride_x, self.stride_y, int(query.shape[1]/3), int(query.shape[0]/3) )

        self.drect_ratio = 0.2
        self.dims_ratio = 0.3
        self.found_obj_dims = None
        # w/h
        self.query_rect_ratio = self.query.shape[1]/self.query.shape[0]
        self.ratio_query_to_img_dims = np.array([query.shape[0]/img.shape[0], query.shape[1]/img.shape[1]])
        # self.ratio_query_to_img_dims = np.array([query.shape[0]/img.shape[0], query.shape[1]/img.shape[1]])
        # self.min_obj_dist_sqr_scaled = np.linalg.norm(self.ratio_query_to_img_dims)/4
        self.obj_min_x_dist = self.ratio_query_to_img_dims[1]/2
        self.obj_min_y_dist = self.ratio_query_to_img_dims[0]/2

    def scale_bbox(self, bbox):
        bbox = list(bbox)
        bbox[0] /= self.img_orig.shape[1]
        bbox[1] /= self.img_orig.shape[0]
        bbox[2] /= self.img_orig.shape[1]
        bbox[3] /= self.img_orig.shape[0]
        return tuple(bbox)

    def scale_bbox_inv(self, bbox):
        bbox = list(bbox)
        bbox[0] *= self.img_orig.shape[1]
        bbox[1] *= self.img_orig.shape[0]
        bbox[2] *= self.img_orig.shape[1]
        bbox[3] *= self.img_orig.shape[0]
        return bbox

    def verify_bbox(self, bbox):
        if not bbox:
            return False
        # print(f'self.found_obj_dims: {self.found_obj_dims}')
        # print(f'bbox: {bbox}')
        rect_ratio = np.abs(bbox[2]/bbox[3] - self.query_rect_ratio)/self.query_rect_ratio
        is_ok = rect_ratio < self.drect_ratio
        if not is_ok:
            # print(f'object bbox has bad rect dimensions ratio!: {rect_ratio}')
            pass
        is_ok = True
        if self.found_obj_dims:
            dimx_ratio = np.abs(bbox[2]-self.found_obj_dims[0])/self.found_obj_dims[0]
            dimy_ratio = np.abs(bbox[3]-self.found_obj_dims[1])/self.found_obj_dims[1]
            is_ok = is_ok and  dimx_ratio < self.dims_ratio and dimy_ratio < self.dims_ratio
            if not is_ok:
                # print(f'object bbox has bad dimensions: {dimx_ratio}, {dimy_ratio}')
                pass
        return is_ok

    def detect_strongest_object(self, filtered_matches, kp_src, kp_dst ,query_shape, img_shape, debug=False):
        if len(filtered_matches)>self.matcher.min_match_count:
            # TODO:: check if I'm using right query and train indeces
            #kp_query[m.queryIdx] will be kp of kp_query in match m
            #kp_img[m.queryIdx] will be kp of kp_img in match m
            src_pts = np.float32([kp_src[m.queryIdx].pt for m in filtered_matches]).reshape(-1,1,2)
            dst_pts = np.float32([kp_dst[m.trainIdx].pt for m in filtered_matches]).reshape(-1,1,2)
            Homography, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
            matches_mask = mask.ravel().tolist()
            h_query, w_query = query_shape

            # 0 -- 4
            # -    -
            # 1 -- 2
            pts = np.float32([ [0,0],[0,h_query-1],[w_query-1,h_query-1],[w_query-1,0] ]).reshape(-1,1,2)
            try:
                dst = cv.perspectiveTransform(pts,Homography)
            except:
                return []
            # print(f'dst: {dst}')

            # x_min, y_min, w, h = bbox
            bbox = list(cv.boundingRect(dst))
            # print("found object!")
        else:
            # print("Not enough matches to find object")
            return []
        return tuple(bbox)

    def scan_img_once(self, img, debug=False):
        bboxes_list = []
        self.roi_handler = ROIHandler(img, self.w_roi, self.h_roi, self.stride_x, self.stride_y, self.query.shape[1]/3, self.query.shape[0]/3)

        # while not self.roi_handler.is_end or not obj_bbox:
        while not self.roi_handler.is_end:
            roi_obj = self.roi_handler.next_roi()
            roi = roi_obj.img
            kp_img, descr_img = self.detector.detectAndCompute(roi, mask=None)
            kp_query, descr_query = self.detector.detectAndCompute(self.query, mask=None)

            matches = self.matcher.match(descr_query, descr_img, debug, roi, kp_img, self.query, kp_query)
            filtered_matches, matches_mask = self.matcher.filter_matches(matches, debug, roi, kp_img, self.query, kp_query)

            obj_bbox = self.detect_strongest_object(filtered_matches, kp_query, kp_img, self.query.shape[:2], roi.shape[:2], debug=debug)

            if not obj_bbox:
                continue
            if not self.verify_bbox(self.scale_bbox(obj_bbox)):
                continue

            obj_bbox = self.roi_handler.cvt_roi_bbox_to_img(obj_bbox)
            self.roi_handler.img = remove_object_from_img(self.roi_handler.img, obj_bbox)

            obj_bbox = self.scale_bbox(obj_bbox)
            bboxes_list.append(obj_bbox)
            self.found_obj_dims = obj_bbox[2:]
            if debug:
                # print(self.scale_bbox(obj_bbox))
                cv.imshow('current_img',self.roi_handler.img)
        # print('scanned all image once')
        return bboxes_list, self.roi_handler.img

    def remove_outliers(self, bboxes_list):
        print(f'bboxes before remove outliers:\n{bboxes_list}')
        mask_arr = np.zeros(len(bboxes_list))
        filtered_bboxes = np.array(bboxes_list)
        index = 0
        while index != len(filtered_bboxes):
            bbox = filtered_bboxes[index]
            filtered_bboxes = filtered_bboxes[ np.logical_or(np.abs(filtered_bboxes[:,0] - bbox[0]) > self.obj_min_x_dist, \
                                                              np.abs(filtered_bboxes[:,1] - bbox[1]) > self.obj_min_y_dist)]
            filtered_bboxes = np.insert(filtered_bboxes, index, bbox, axis=0)
            index += 1
        filtered_bboxes = filtered_bboxes.tolist()
        filtered_bboxes = list(map(tuple, filtered_bboxes))

        filtered_bboxes = list(filter(self.is_bbox_outside, filtered_bboxes))

        return filtered_bboxes

    def is_bbox_outside(self, bbox):
        # scaled bboxe
        x_min = bbox[0]
        y_min = bbox[1]
        x_max = bbox[0]+bbox[2]
        y_max = bbox[1]+bbox[3]

        if  x_min < -bbox[2]*2/3 \
            or y_min < -bbox[3]*2/3 \
            or (x_max - 1) > bbox[2]*2/3 \
            or (y_max - 1) > bbox[3]*2/3:
            return False
        return True

    def detect_all_objects(self, debug=False):
        bboxes_list = []
        if debug:
            create_win('current_img')
        img_wo_objects = self.img_orig.copy()

        # while True:
        #     bboxes_list_one_iter, img_wo_objects = self.scan_img_once(img_wo_objects, debug=debug)
        #     if not bboxes_list_one_iter:
        #         break
        #     bboxes_list.extend(bboxes_list_one_iter)

        bboxes_list_one_iter, img_wo_objects = self.scan_img_once(img_wo_objects, debug=debug)

        bboxes_list.extend(bboxes_list_one_iter)

        bboxes_list = self.remove_outliers(bboxes_list)
        if not bboxes_list:
            bboxes_list = [(0,0,1,1)]
        if debug:
            imshow('img_wo_objects', self.roi_handler.img)
        print(f'Found all objects: {len(bboxes_list)} count')
        print(bboxes_list)
        return bboxes_list

def predict_image(img: np.ndarray, query: np.ndarray) -> list:
    # list_of_bboxes = [(0, 0, 1, 1), ]
    # return list_of_bboxes

    product_detector = ProductDetectorOnFeatures(img, query)
    bboxes_list = product_detector.detect_all_objects()
    if not bboxes_list:
        return [(0, 0, 1, 1)]
    return bboxes_list

if __name__ == '__main__':

    debug=True

    img = TRAIN_IMAGES[2]
    query = TEMPLATE_IMAGES[3]
    # img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # query = cv.cvtColor(query, cv.COLOR_BGR2GRAY)

    feature_detector = cv.SIFT_create()
    feature_matcher = FeatureMatcher()
    kp_img, descr_img = feature_detector.detectAndCompute(img, mask=None)
    kp_query, descr_query = feature_detector.detectAndCompute(query, mask=None)

    # init_params = []
    # func_name = 'match'
    # var_names = ['FLANN_INDEX_KDTREE', 'index_trees', 'search_checks']
    # var_coeffs = [1,1,1, 100]
    # max_var_vals = [5, 20, 100, 100]
    # CalibrateParametersGUI( FeatureMatcher, init_params, func_name, var_names, var_coeffs, max_var_vals, descr_query, descr_img, debug=True, img=img, kp_img=kp_img, query=query, kp_query=kp_query )
    # matches = feature_matcher.match(descr_img, descr_query)
    # init_params = []
    # func_name = 'filter_matches'
    # var_names = ['FLANN_INDEX_KDTREE', 'index_trees', 'search_checks', 'lowe']
    # var_coeffs = [1,1,1, 100]
    # max_var_vals = [20, 20, 100, 100]
    # CalibrateParametersGUI( FeatureMatcher, init_params, func_name, var_names, var_coeffs, max_var_vals, matches, debug=True, img=img, kp_img=kp_img, query=query, kp_query=kp_query )

    product_detector = ProductDetectorOnFeatures(img, query)
    bboxes = product_detector.detect_all_objects(debug=debug)
    # product_detector.detect_objects_old(img, query, debug=debug)

    while True:
        key = cv.waitKey(100) & 0xFF
        if key == ord('q'):
            break
    cv.destroyAllWindows()
