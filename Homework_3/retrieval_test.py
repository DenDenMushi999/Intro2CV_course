import cv2 as cv
import numpy as np

from utils import CalibrateParametersGUI
from product_detector import FeatureMatcher, ProductDetectorOnFeatures
from product_detector import TRAIN_IMAGES, TEMPLATE_IMAGES


def predict_image(img: np.ndarray, query: np.ndarray) -> list:
    # list_of_bboxes = [(0, 0, 1, 1), ]
    # return list_of_bboxes

    product_detector = ProductDetectorOnFeatures(img, query)
    return product_detector.detect_all_objects()

if __name__ == '__main__':

    debug=False

    img = TRAIN_IMAGES[3]
    query = TEMPLATE_IMAGES[4]
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
    product_detector.detect_all_objects(debug=debug)
    # product_detector.detect_objects_old(img, query, debug=debug)

    while True:
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    cv.destroyAllWindows()
