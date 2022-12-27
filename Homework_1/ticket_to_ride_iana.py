#!/usr/bin/env python
# coding: utf-8




import numpy as np
import cv2
from skimage.transform import rescale
from skimage.measure import label, find_contours
from skimage.filters import gaussian
from scipy.spatial.distance import cdist
#import scipy.stats as st
#from operator import itemgetter
#from imutils.object_detection import non_max_suppression
#import matplotlib.pyplot as plt
#from scipy.spatial import distance as dist

PATH = '/autograder/submission/'
# PATH = ''

class TempMatch():
    def __init__(self, city_template):
        #self.city_template = cv2.imread('template.jpg', 0)
        self.city_template = city_template
        self.w = city_template.shape[::-1][0]
        self.h = city_template.shape[::-1][1]
        self.boxes = list()



    def match(self, img):
        #template matching
        count = 0
        imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        res = cv2.matchTemplate(imgray.astype(np.uint8),self.city_template.astype(np.uint8),cv2.TM_CCOEFF_NORMED)
        threshold = 0.58

        (y_points, x_points) = np.where(res >= threshold)

        # REinitialize our list of bounding boxes
        self.boxes = list()
        # store co-ordinates of each bounding box
        # we'll create a new list by looping
        # through each pair of points
        for (x, y) in zip(x_points, y_points):

            self.boxes.append((x, y, x + self.w, y + self.h))

        self.boxes = np.array(self.boxes)

        # loop over the final bounding boxes

        for (x1, y1, x2, y2) in self.boxes:

            # draw the bounding box on the image
            cv2.rectangle(img, (x1, y1), (x2, y2),(255, 0, 0), 3)
            count += 1
        obj_num = count
        return [self.boxes[:,1], self.boxes[:,0]], obj_num, img, self.boxes


def filter_cntrs(city_cntrs):
    thresh = 50
    filtered_centers = []

    for i in range (len(city_cntrs) - 1):
        if not np.all(np.sqrt((np.array(city_cntrs[i])- np.array(city_cntrs[i+1]))**2) < thresh):
            filtered_centers.append(city_cntrs[i])
    filtered_centers.append(city_cntrs[i+1])
    return filtered_centers



#Detect cities and get the coordinates of city centers
def city_coord(img):
    city_template = cv2.imread(f'{PATH}template.jpg', 0)
    match = TempMatch(city_template)
    result_match = match.match(img)
    coord = result_match[0]

    # Collect the coordinates of the center
    center_coord = []

    for i in range (len(coord[0])):
        center_x = coord[0][i] + 0.5*match.w
        center_y = coord[1][i] + 0.5*match.h
        center_coord.append((center_x, center_y))

    coord_sort = sorted(center_coord,key=lambda l:l[0])
    return coord_sort


# Filter too small contours

def contour_filter(contours, hierarchy, minArea):
    cnts1 = []
    for i in range(len(contours)):
        if hierarchy[0][i][3] == -1 and cv2.contourArea(contours[i]) > minArea:
            cnts1.append(contours[i])
    return cnts1

# Detect contours
def contours(img, minArea):
    contours, hierarchy = cv2.findContours(img,
                                            cv2.RETR_TREE,
                                            cv2.CHAIN_APPROX_SIMPLE)

    cnts = contour_filter(contours, hierarchy, minArea)

    #draw contours
    cv2.drawContours(img, cnts, -1, (255,0,0), 3)
    cv2.imwrite('cnts.png', img)
    return cnts, hierarchy, img




#filter green color

def filter_green(img, k=9, g_blur=3, m_blur=11):
    image = img

    img = cv2.GaussianBlur(img,(g_blur,g_blur),0)
    img = img[..., ::-1]
    #------------------------------------
    HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


    l_range = np.array([75, 160,50 ])
    u_range = np.array([83, 255,145 ])
    #green mask
    mask_green = cv2.inRange(HSV, l_range, u_range)
    mask_int_green = mask_green.astype(np.uint8)
    kernel = np.ones((k,k))
    mask_int_green = cv2.morphologyEx(mask_int_green, cv2.MORPH_CLOSE, kernel)
    mask_int_green = cv2.medianBlur(mask_int_green,m_blur)

    return mask_int_green




#filter red color

def filter_red(img, k=9, g_blur=3, m_blur=11):


    image = img
    rgb = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB )
    Img = cv2.GaussianBlur(image,(g_blur,g_blur),0)
    Img = Img[..., ::-1]
    #------------------------------------
    HLS = cv2.cvtColor(Img, cv2.COLOR_BGR2HLS)
    HUE = HLS[:, :, 0]              # Split attributes
    LIGHT = HLS[:, :, 1]
    SAT = HLS[:, :, 2]


    l_range = np.array([121, 86, 148])
    u_range = np.array([132, 148, 190])
    #red mask
    mask_red = cv2.inRange(HLS, l_range, u_range)
    mask_int_red = mask_red.astype(np.uint8)
    kernel = np.ones((k,k))
    mask_int_red = cv2.morphologyEx(mask_int_red, cv2.MORPH_CLOSE, kernel)
    mask_int_red = cv2.medianBlur(mask_int_red,m_blur)

    return mask_int_red


#filter blue color

def filter_blue(img, k=9, g_blur=3, m_blur=11):

    image = img
    #------------------------------------
    img = cv2.GaussianBlur(img,(g_blur,g_blur),0)
    img = img[..., ::-1]
    #------------------------------------
    HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    l_range = np.array([100, 150,110 ])
    u_range = np.array([120, 255, 180 ])
    #blue mask
    mask_blue = cv2.inRange(HSV, l_range, u_range)
    mask_blue = cv2.medianBlur(mask_blue, 3)
    mask_int_blue = mask_blue.astype(np.uint8)
    kernel = np.ones((k,k))
    mask_int_blue = cv2.morphologyEx(mask_int_blue, cv2.MORPH_CLOSE, kernel)
    mask_int_blue = cv2.medianBlur(mask_int_blue,m_blur)


    return mask_int_blue



#filter yellow color

def filter_yellow(img, k, g_blur, m_blur):

    img = cv2.GaussianBlur(img,(g_blur,g_blur),0)
    img = img[..., ::-1]
    #------------------------------------
    HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)


    l_range = np.array([65, 96, 103])
    u_range = np.array([105, 157, 191])
    #yellow mask
    mask_yellow = cv2.inRange(HSV, l_range, u_range)
    mask_yellow = cv2.medianBlur(mask_yellow, 3)
    mask_int_yellow = mask_yellow.astype(np.uint8)
    kernel = np.ones((k,k))
    mask_int_yellow = cv2.morphologyEx(mask_int_yellow, cv2.MORPH_CLOSE, kernel)
    mask_int_yellow = cv2.medianBlur(mask_int_yellow,m_blur)

    return mask_int_yellow



#filter yellow color

def filter_black(img):
    k = 19

    #------------------------------------
    HLS = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    HUE = HLS[:, :, 0]
    LIGHT = HLS[:, :, 1]
    SAT = HLS[:, :, 2]

    mask_black = (LIGHT < 25)
    mask_int_black = mask_black.astype(np.uint8)
    kernel = np.ones((15,15))
    mask_int_black = cv2.morphologyEx(mask_int_black, cv2.MORPH_CLOSE, kernel)
    kernel = np.ones((19,19))
    mask_int_black = cv2.morphologyEx(mask_int_black, cv2.MORPH_OPEN, kernel)

    return mask_int_black



# Calculate the contours areas

def cnts_area(cnts):
    area = []
    for i in range(len(cnts)):
        area.append(cv2.contourArea(cnts[i]))
    return area




# Get the size of one track
def size_of_track(img, filter_func, minArea):
    cnts = contours(filter_func, minArea)
    size = cnts_area(cnts[0])
    return size




def count_trains(area, one_size):
   n_train = 0
   score = 0
   #max_area = np.max(area)
   min_area = np.min(area)

   for i in range(len(area)):

      size = area[i]/one_size
      #print(size)
      if size < 1:
         n_train += 1
         score += 1
      elif size >=1 and size <2:
         n_train += 2
         score += 2
      elif size >= 2 and size < 3:
         n_train += 3
         score += 4
      elif size >= 3 and size <4:
         n_train += 4
         score += 7
      elif size >= 4 and size < 5:
         n_train += 5
         score += 15
      elif size >=5 and size < 6:
         n_train += 6
         score += 15
      else:
         score += 21


   return n_train, score



# The number of green trains

def count_green(img, img_green_tmp):
    n_green = 0
    green_cnts = contours(filter_green(img.copy(), k=23, g_blur=1, m_blur=27), 1000)
    green__size = size_of_track(img_green_tmp, filter_green(img_green_tmp, k=23, g_blur=1, m_blur=27), 500)
    green_area = cnts_area(green_cnts[0])
    # print(green__size)
    if not green_area:
        return 0, 0
    else:
        return count_trains(green_area, green__size[0])



def remove_tracks(img, template, cnts):

    match = TempMatch(template)
    result_match = match.match(img)
    img = result_match[2]
    rect = result_match[3]
    coord = result_match[0]
    center_coord = []

    for i in range (len(coord[0])):
        center_x = coord[1][i] + 0.5*match.w
        center_y = coord[0][i] + 0.5*match.h
        center_coord.append((center_x, center_y))

    hull_list = []

    for i in range(len(cnts)):
        hull = cv2.convexHull(cnts[i])
        hull_list.append(hull)

    id  = []

    for i in range(len(center_coord)):
        for j in range(len(hull_list)):
            result = cv2.pointPolygonTest(hull_list[j], center_coord[i], False)
            if result == 1.0 or result==0.0:
                id.append(j)

    id_sort = sorted(id, reverse=True)
    # print(id_sort)
    id_sort = dict.fromkeys(id_sort)
    for i in id_sort:
        hull_list.pop(i)

    return hull_list


def count_blue(img, img_blue_tmp):
    n_blue = 0
    template = cv2.imread(f'{PATH}blue_no_train.jpg', 0)
    blue_cnts = contours(filter_blue(img), 1000)
    hull_list = remove_tracks(img.copy(), template, blue_cnts[0])
    #blue__size = size_of_track(img_blue_tmp, filter_blue(img_blue_tmp, k=23, g_blur=1, m_blur=27), 1000)
    blue_size = 4200
    blue_area = cnts_area(hull_list)
    if not blue_area:
        return 0, 0
    else:
        return count_trains(blue_area, blue_size)




# The number of blue trains

def count_yellow(img, img_yellow_tmp):
    n_yellow = 0
    template = cv2.imread(f'{PATH}yellow_no_train1.jpg', 0)
    yellow_cnts = contours(filter_yellow(img, k=13, g_blur=1, m_blur=17), 1000)
    hull_list = remove_tracks(img.copy(), template, yellow_cnts[0])
    #yellow__size = size_of_track(img_yellow_tmp, filter_yellow(img_yellow_tmp,k=3, g_blur=1, m_blur=7), 1000)
    yellow_area = cnts_area(hull_list)
    yellow_size = 6700 - 1200
    if not yellow_area:
        return 0, 0
    else:
        return count_trains(yellow_area, yellow_size)





def count_red(img, img_red_tmp):
    n_red = 0
    # COntours for red trains

    red_cnts = contours(filter_red(img, k=23, g_blur=1, m_blur=27), 1000)
    #red__size = size_of_track(img_red_tmp, filter_red(img_red_tmp, k=23, g_blur=1, m_blur=27), 1000)
    red_size = 5000
    # The number of red trains
    red_area = cnts_area(red_cnts[0])
    # print(red_area)
    if not red_area:
        return 0, 0
    else:
        return count_trains(red_area, red_size)



def count_black(img, img_black_tmp):
    n_black = 0
    # COntours for red trains
    black_cnts = contours(filter_black(img), 6000)
    # The number of red trains
    black_area = cnts_area(black_cnts[0])
    # print(black_area)
    #black__size = size_of_track(img_black_tmp, filter_black(img_black_tmp), 500)
    black_size = 6500

    if not black_area:
        return 0, 0
    else:
        return count_trains(black_area, black_size)





def predict_image(img):
    city_cntrs = city_coord(img)
    city_center = filter_cntrs(city_cntrs)

    img_green_tmp = cv2.imread(f'{PATH}template_green.jpg')
    img_blue_tmp = cv2.imread(f'{PATH}template_blue.jpg')
    img_yellow_tmp = cv2.imread(f'{PATH}template_yellow.jpg')
    img_red_tmp = cv2.imread(f'{PATH}template_red.jpg')
    img_black_tmp = cv2.imread(f'{PATH}template_black.jpg')

    n_green, score_green = count_green(img, img_green_tmp)
    n_blue, score_blue = count_blue(img, img_blue_tmp)
    n_yellow, score_yellow = count_yellow(img, img_yellow_tmp)
    n_red, score_red = count_red(img, img_red_tmp)
    n_black, score_black = count_black(img, img_black_tmp)

    n_trains = {'blue': n_blue, 'green': n_green, 'black': n_black, 'yellow': n_yellow, 'red': n_red}
    scores = {'blue': score_blue, 'green': score_green, 'black': score_black, 'yellow': score_yellow, 'red':score_red}
    city_center = np.int64(city_center)

    return city_center, n_trains, scores




