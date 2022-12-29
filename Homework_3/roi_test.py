import numpy as np

from roi_utils import ROIHandler

def roi_handler_test():
    img = np.ones((300,200))
    roi_handler = ROIHandler(img, w_roi=20, h_roi=30, stride_x=10, stride_y=10)
    roi = roi_handler.next_roi()
    print(roi.bbox)
    roi = roi_handler.next_roi()
    print(roi.bbox)
    roi = roi_handler.next_roi()
    print(roi.bbox)

roi_handler_test()
