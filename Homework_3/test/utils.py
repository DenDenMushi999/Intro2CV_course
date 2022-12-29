import json
from typing import List

import cv2 as cv
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

    print(f'Final values of {cls.__name__}.{func_name}():')
    for var_name in var_names:
        print(f'{var_name}: {getattr(instance, var_name)}')
