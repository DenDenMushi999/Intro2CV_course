from typing import Tuple

import numpy as np
import cv2 as cv
from utils import CalibrateParametersGUI, imshow, create_win
from roi_utils import ROI, ROIHandler, remove_object_from_img
from feature_matcher import FeatureMatcher

TEMPLATE_FNAME_PREFIXES = ['0_0', '0_1', '1', '2', '3']
TRAIN_FNAME_PREFIXES = ['0', '1', '2', '3', 'extreme']

TEMPLATE_IMAGES = [cv.imread(f'train/template_{prefx}.jpg') for prefx in TEMPLATE_FNAME_PREFIXES]
TRAIN_IMAGES = [cv.imread(f'train/train_{prefx}.jpg') for prefx in TRAIN_FNAME_PREFIXES]


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
        self.dims_ratio = 0.2
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
        print(f'self.found_obj_dims: {self.found_obj_dims}')
        print(f'bbox: {bbox}')
        rect_ratio = np.abs(bbox[2]/bbox[3] - self.query_rect_ratio)/self.query_rect_ratio
        is_ok = rect_ratio < self.drect_ratio
        if not is_ok:
            print(f'object bbox has bad rect dimensions ratio!: {rect_ratio}')
        is_ok = True
        if self.found_obj_dims:
            dimx_ratio = np.abs(bbox[2]-self.found_obj_dims[0])/self.found_obj_dims[0]
            dimy_ratio = np.abs(bbox[3]-self.found_obj_dims[1])/self.found_obj_dims[1]
            is_ok = is_ok and  dimx_ratio < self.dims_ratio and dimy_ratio < self.dims_ratio
            if not is_ok:
                print(f'object bbox has bad dimensions: {dimx_ratio}, {dimy_ratio}')
        return is_ok

    def detect_strongest_object(self, filtered_matches, kp_src, kp_dst ,query_shape, img_shape, debug=False) -> Tuple:
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
            dst = cv.perspectiveTransform(pts,Homography)
            print(f'dst: {dst}')

            # x_min, y_min, w, h = bbox
            bbox = list(cv.boundingRect(dst))
            print("found object!")
        else:
            print("Not enough matches to find object")
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
                print(self.scale_bbox(obj_bbox))
                cv.imshow('current_img',self.roi_handler.img)
                cv.waitKey(100)
        print('scanned all image once')
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
        return filtered_bboxes

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

        if debug:
            imshow('img_wo_objects', self.roi_handler.img)
        print(f'Found all objects: {len(bboxes_list)} count')
        print(bboxes_list)
        return bboxes_list

    def detect_objects_old(self, img, query, debug=False):
        bboxes_list = []

        if debug:
            create_win('current_img')
        obj_bbox = []
        while True:
            obj_bbox = []

            kp_img, descr_img = self.detector.detectAndCompute(img, mask=None)
            kp_query, descr_query = self.detector.detectAndCompute(query, mask=None)

            matches = self.matcher.match(descr_query, descr_img, debug, img, kp_img, query, kp_query)
            filtered_matches, matches_mask = self.matcher.filter_matches(matches, debug, img, kp_img, query, kp_query)

            obj_bbox = self.detect_strongest_object(filtered_matches, kp_query, kp_img, query.shape[:2], img.shape[:2], debug=debug)
            print(self.scale_bbox(obj_bbox))

            if not self.verify_bbox(obj_bbox):
                break

            img = self.remove_object_from_img(img, obj_bbox)

            # self.verify_bbox(obj_bbox)

            obj_bbox = self.scale_bbox(obj_bbox)
            self.found_obj_dims = obj_bbox[2:]
            bboxes_list.append(obj_bbox)
            if debug:
                cv.imshow('current_img',img)
        print('Found all objects')
        return bboxes_list



class ProductDetectorOnTemplateMatching:
    def __init__(self):
        pass
