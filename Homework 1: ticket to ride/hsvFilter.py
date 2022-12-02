import cv2 as cv
from matplotlib import  pyplot as plt
import numpy as np
from numpy.core.fromnumeric import resize
from scipy.signal import convolve2d
img1 = cv.imread("train/all.jpg") 
# img2 = cv.imread("train/black_blue_green.jpg")
# img3 = cv.imread("train/black_red_yellow.jpg")
# img4 = cv.imread("train/red_green_blue_inaccurate.jpg")
# img5 = cv.imread("train/red_green_blue.jpg")

image = img1
gray= cv.cvtColor(image, cv.COLOR_BGR2GRAY)

def empty(a):
    pass

cv.namedWindow("Range HSV")
cv.resizeWindow("Range HSV", 800,500)
cv.createTrackbar("HUE Min", "Range HSV", 0, 180, empty)
cv.createTrackbar("HUE Max", "Range HSV", 180, 180, empty)
cv.createTrackbar("HUE Min2", "Range HSV", 0, 180, empty)
cv.createTrackbar("HUE Max2", "Range HSV", 180, 180, empty)
cv.createTrackbar("SAT Min", "Range HSV", 0, 255, empty)
cv.createTrackbar("SAT Max", "Range HSV", 255, 255, empty)
cv.createTrackbar("VAL Min", "Range HSV", 0, 255, empty)
cv.createTrackbar("VAL Max", "Range HSV",255, 255, empty)
cv.createTrackbar("blur", "Range HSV",1, 50, empty)
cv.createTrackbar("k size", "Range HSV",1, 21, empty)
cv.createTrackbar("Canny High", "Range HSV",101, 255, empty)
cv.createTrackbar("dilate", "Range HSV",1, 255, empty)
cv.createTrackbar("dilate_iter", "Range HSV",1, 20, empty)
cv.createTrackbar("erode", "Range HSV",1, 255, empty)
cv.createTrackbar("erode_iter", "Range HSV",1, 20, empty)


image = cv.resize(image, (800, 600)) 

def filter(img, gain, case = 1):
    if gain%2 != 1 or gain == 0 :
        gain = gain + 1
    if case == 1: # filter2D
        kernel = np.ones((gain,gain),np.float32)/(gain**2)
        img =  cv.filter2D(img,-1,kernel)
    elif case ==2: # medianBlur
        img = cv.medianBlur(img, gain)
    elif case == 3: # gaussian blur
        img = cv.GaussianBlur(img,(gain,gain),cv.BORDER_DEFAULT)
    elif case ==4:
        kernel = np.array([[0,-1,0], 
                        [-1, 5, -1],
                        [0,-1,0]])
        img = cv.filter2D(image, -1, kernel) 

    # img = convolve2d(img, np.ones((gain, gain))/(gain**2), mode= "same")
    return img




while True:
    h_min = cv.getTrackbarPos("HUE Min", "Range HSV")
    h_max = cv.getTrackbarPos("HUE Max", "Range HSV")
    h_min2 = cv.getTrackbarPos("HUE Min2", "Range HSV")
    h_max2 = cv.getTrackbarPos("HUE Max2", "Range HSV")
    s_min = cv.getTrackbarPos("SAT Min", "Range HSV")
    s_max = cv.getTrackbarPos("SAT Max", "Range HSV")
    v_min = cv.getTrackbarPos("VAL Min", "Range HSV")
    v_max = cv.getTrackbarPos("VAL Max", "Range HSV")
    blur_ = cv.getTrackbarPos("blur", "Range HSV")
    k = cv.getTrackbarPos("k size", "Range HSV")
    canny_high = cv.getTrackbarPos("Canny High", "Range HSV")
    dilate = cv.getTrackbarPos("dilate", "Range HSV")
    dilate_iteration = cv.getTrackbarPos("dilate_iter", "Range HSV")
    erode = cv.getTrackbarPos("erode", "Range HSV")
    erode_iteration = cv.getTrackbarPos("erode_iter", "Range HSV")
    if dilate == 0:
        dilate = 1
    if erode == 0:
        erode = 1

    if k%2 != 1 or k == 0 :
           k = k + 1
   
    rgb = cv.cvtColor(image, cv.COLOR_BGR2HSV )


    
    lower_range = np.array([h_min, s_min, v_min])
    upper_range = np.array([h_max, s_max, v_max])
    lower_range2 = np.array([h_min2, s_min, v_min])
    upper_range2 = np.array([h_max2, s_max, v_max])

    gray_low = np.array([h_min])
    gray_max = np.array([h_max])

    # hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HLS)
    # img = gray.copy()
    # img = img.astype(np.uint8)
    # img = cv.resize(img, (800, 600))
    # mask = cv.inRange(filter(img.copy(),blur_, 2), gray_low, gray_max)
    mask = cv.inRange(filter(hsv.copy(),blur_, 2), lower_range, upper_range)
    mask2 = cv.inRange(filter(hsv.copy(),blur_, 2), lower_range2, upper_range2)
    mask = cv.bitwise_or(mask, mask2)
    # mask = cv.inRange(hsv.copy(), lower_range, upper_range)
   
   
    mask_int = mask.astype(np.uint8)     
    kernel = np.ones((k,k))
    erode_kernel = np.ones((erode,erode))
    dilate_kernel = np.ones((dilate,dilate))
    close_kernel = np.ones((3,3))
    # mask_int = cv.morphologyEx(mask_int, cv.MORPH_CLOSE, close_kernel, iterations=3)
    # mask_int = cv.morphologyEx(mask_int, cv.MORPH_OPEN, close_kernel, iterations=1)
    mask_int = cv.morphologyEx(mask_int, cv.MORPH_ERODE, erode_kernel, iterations=erode_iteration)
    mask_int = cv.morphologyEx(mask_int, cv.MORPH_DILATE, dilate_kernel, iterations=dilate_iteration)
    # mask_int = cv.medianBlur(mask_int, 1)
    
    cv.imshow("original Image", image)
    cv.imshow("mask ", mask)
    cv.imshow("mask int/ mask", mask_int)
    # cv.imshow("bitwise ", bitwise)
    # cv.imshow("bitwise ", bitwise)
    # # cv.imshow("canny ", canny)
    # cv.imshow("contours ", rgbImg)
    

    key = cv.waitKey(1) & 0xFF
    if key == ord('q'):
        mode = not mode
    elif key == 27:
        break

cv.destroyAllWindows()
