import cv2
from matplotlib import  pyplot as plt
import numpy as np
from numpy.core.fromnumeric import resize
from scipy.signal import convolve2d
img1 = cv2.imread("train/all.jpg") 
# img2 = cv2.imread("train/black_blue_green.jpg")
# img3 = cv2.imread("train/black_red_yellow.jpg")
# img4 = cv2.imread("train/red_green_blue_inaccurate.jpg")
# img5 = cv2.imread("train/red_green_blue.jpg")

image = img1
gray= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def empty(a):
    pass

cv2.namedWindow("Range HSV")
cv2.resizeWindow("Range HSV", 800,500)
cv2.createTrackbar("HUE Min", "Range HSV", 0, 180, empty)
cv2.createTrackbar("HUE Max", "Range HSV", 180, 180, empty)
cv2.createTrackbar("HUE Min2", "Range HSV", 0, 180, empty)
cv2.createTrackbar("HUE Max2", "Range HSV", 180, 180, empty)
cv2.createTrackbar("SAT Min", "Range HSV", 0, 255, empty)
cv2.createTrackbar("SAT Max", "Range HSV", 255, 255, empty)
cv2.createTrackbar("VAL Min", "Range HSV", 0, 255, empty)
cv2.createTrackbar("VAL Max", "Range HSV",255, 255, empty)
cv2.createTrackbar("blur", "Range HSV",1, 50, empty)
cv2.createTrackbar("k size", "Range HSV",1, 21, empty)
cv2.createTrackbar("Canny High", "Range HSV",101, 255, empty)
cv2.createTrackbar("dilate", "Range HSV",1, 255, empty)
cv2.createTrackbar("dilate iter", "Range HSV",1, 20, empty)



image = cv2.resize(image, (800, 600)) 

def filter(img, gain, case = 1):
    if gain%2 != 1 or gain == 0 :
        gain = gain + 1
    if case == 1: # filter2D
        kernel = np.ones((gain,gain),np.float32)/(gain**2)
        img =  cv2.filter2D(img,-1,kernel)
    elif case ==2: # medianBlur
        img = cv2.medianBlur(img, gain)
    elif case == 3: # gaussian blur
        img = cv2.GaussianBlur(img,(gain,gain),cv2.BORDER_DEFAULT)
    elif case ==4:
        kernel = np.array([[0,-1,0], 
                        [-1, 5, -1],
                        [0,-1,0]])
        img = cv2.filter2D(image, -1, kernel) 

    # img = convolve2d(img, np.ones((gain, gain))/(gain**2), mode= "same")
    return img




while True:
    h_min = cv2.getTrackbarPos("HUE Min", "Range HSV")
    h_max = cv2.getTrackbarPos("HUE Max", "Range HSV")
    h_min2 = cv2.getTrackbarPos("HUE Min2", "Range HSV")
    h_max2 = cv2.getTrackbarPos("HUE Max2", "Range HSV")
    s_min = cv2.getTrackbarPos("SAT Min", "Range HSV")
    s_max = cv2.getTrackbarPos("SAT Max", "Range HSV")
    v_min = cv2.getTrackbarPos("VAL Min", "Range HSV")
    v_max = cv2.getTrackbarPos("VAL Max", "Range HSV")
    blur_ = cv2.getTrackbarPos("blur", "Range HSV")
    k = cv2.getTrackbarPos("k size", "Range HSV")
    canny_high = cv2.getTrackbarPos("Canny High", "Range HSV")
    dilate = cv2.getTrackbarPos("dilate", "Range HSV")
    dilate_iteration = cv2.getTrackbarPos("dilate_iter", "Range HSV")
    if dilate == 0:
        dilate = 1

    if k%2 != 1 or k == 0 :
           k = k + 1
   
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2HSV )


    
    lower_range = np.array([h_min, s_min, v_min])
    upper_range = np.array([h_max, s_max, v_max])
    lower_range2 = np.array([h_min2, s_min, v_min])
    upper_range2 = np.array([h_max2, s_max, v_max])

    gray_low = np.array([h_min])
    gray_max = np.array([h_max])

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # img = gray.copy()
    # img = img.astype(np.uint8)
    # img = cv2.resize(img, (800, 600))
    # mask = cv2.inRange(filter(img.copy(),blur_, 2), gray_low, gray_max)
    mask = cv2.inRange(filter(hsv.copy(),blur_, 2), lower_range, upper_range)
    mask2 = cv2.inRange(filter(hsv.copy(),blur_, 2), lower_range2, upper_range2)
    mask = cv2.bitwise_or(mask, mask2)
    # mask = cv2.inRange(hsv.copy(), lower_range, upper_range)
   
   
    mask_int = mask.astype(np.uint8)     
    kernel = np.ones((k,k))
    erode_kernel = np.ones((3,3))
    close_kernel = np.ones((3,3))
    # mask_int = cv2.morphologyEx(mask_int, cv2.MORPH_CLOSE, close_kernel, iterations=3)
    # mask_int = cv2.morphologyEx(mask_int, cv2.MORPH_OPEN, close_kernel, iterations=1)
    # mask_int = cv2.morphologyEx(mask_int, cv2.MORPH_ERODE, kernel)
    mask_int = cv2.morphologyEx(mask_int, cv2.MORPH_DILATE, kernel)
    # mask_int = cv2.medianBlur(mask_int, 1)
    
    cv2.imshow("original Image", image)
    cv2.imshow("mask ", mask)
    cv2.imshow("mask int/ mask", mask_int)
    # cv2.imshow("bitwise ", bitwise)
    # cv2.imshow("bitwise ", bitwise)
    # # cv2.imshow("canny ", canny)
    # cv2.imshow("contours ", rgbImg)
    

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        mode = not mode
    elif key == 27:
        break

cv2.destroyAllWindows()
