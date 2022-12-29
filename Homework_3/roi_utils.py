
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
        self.min_w_roi = min_w_roi
        self.min_h_roi = min_h_roi
        self.cur_x = 0
        self.cur_y = 0
        self.roi = ROI(img, (0,0, w_roi, h_roi))
        self.cur_roi_num = 0

    @property
    def is_end(self):
        return self.is_x_end and self.is_y_end

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
                print('roi_handler: end of x')
                new_bbox[0] = 0
                new_bbox[1] += self.stride_y
            else:
                new_bbox[0] += self.stride_x
            self.roi = ROI(self.img, tuple(new_bbox))
        self.cur_roi_num += 1
        return self.roi
