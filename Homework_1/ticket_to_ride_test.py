from typing import Union, List
import json

import numpy as np
import cv2 as cv
from skimage.filters import gaussian
from skimage.measure import label

from cv_utils import CityDetector, TicketToRideHandler
from utils import CalibrateParametersGUI

COLORS = ('blue', 'green', 'black', 'yellow', 'red')
TRAIN_CASES = dict(enumerate(('all', 'black_blue_green', 'black_red_yellow', 'red_green_blue_inaccurate', 'red_green_blue')))

def predict_image(img: np.ndarray) -> (Union[np.ndarray, list], dict, dict):
    return TicketToRideHandler().predict_image(img)


if __name__ == '__main__':
    color = 'red'
    debug=True
    var_names = ['threshold', 'roi_win']
    var_coeffs = [100,1]
    max_var_vals = [100, 200]

    # for case in TRAIN_CASES.values():
    #     img_path = f'train/{case}.jpg'
    #     print(img_path)
    #     img = cv.imread(img_path)
    #     img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #     CalibrateParametersGUI(CityDetector, (), 'find_city_centers_wo_roi_opencv', var_names, var_coeffs, max_var_vals, img_gray, debug=debug)

    img_path = f'train/{TRAIN_CASES[0]}.jpg'
    print(img_path)
    img = cv.imread(img_path)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    task_handler = TicketToRideHandler()

    city_centers = task_handler.find_city_centers_wo_roi_opencv(img_gray, debug=debug)

    # train_centers, trains_mask = task_handler.find_one_color_trains_mask(img, color=color, debug=debug)
    # sticked_trains_mask = task_handler.stick_trains(trains_mask, color, city_centers, 60)
    # print(city_centers)

    # det_centes_img = img.copy()
    # for pt in city_centers:
    #     det_centes_img = cv.circle(det_centes_img, pt[::-1], 30, (255,255,255), 4)

    # blue_ntrains, blue_score = task_handler.count_color_trains_and_score( sticked_trains_mask, color=color, debug=True )
    # print(blue_ntrains, blue_score)
    # cv.namedWindow('city_centers', cv.WINDOW_NORMAL)
    # cv.resizeWindow('city_centers', 600, 500)
    # cv.imshow('city_centers', det_centes_img)

    # cv.namedWindow('sticked_blue_mask', cv.WINDOW_NORMAL)
    # cv.resizeWindow('sticked_blue_mask', 600, 500)
    # cv.imshow('sticked_blue_mask', sticked_trains_mask)

    while True:
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    cv.destroyAllWindows()