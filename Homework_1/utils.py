import json
from typing import List

import cv2 as cv
def empty(a):
    pass

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def CalibrateParametersGUI( cls, init_params: List, func_name, var_names, var_coeffs, max_var_vals, *args, **kwargs ):
    instance = cls(*init_params)
    win_name = f'tune_{cls.__name__}.{func_name}'
    cv.namedWindow(win_name)

    for var_name, coef, max_val in zip(var_names, var_coeffs, max_var_vals):
        cv.createTrackbar(var_name, win_name, int(getattr(instance, var_name)*coef), max_val, empty)

    while True:
        for var_name, coef in zip(var_names,var_coeffs):
            if var_name in dir(instance):
                setattr(instance, var_name, cv.getTrackbarPos(var_name, win_name)/coef)
            else :
                raise AttributeError(f'Obejct of class \'{cls.__name__}\' has no attribute \'{var_name}\'')

        getattr(instance, func_name)(*args, **kwargs)

        key = cv.waitKey(100) & 0xFF
        if key == ord('q'):
            break

    print(f'Final values of {cls.__name__}.{func_name}():')
    for var_name in var_names:
        print(f'{var_name}: {getattr(instance, var_name)}')
