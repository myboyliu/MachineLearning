import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import tensorflow

import sys
sys.path.append("..")
from common.DataLoad import DataLoad

if __name__ == '__main__':
    img = cv.imread('Lena.png', 1)
    img = cv.rectangle(img, (100,100), (190,200),(0,255,0),2)
    image = img[100:190, 100:200]
    cv.imshow("canny", img)
    cv.imshow("ss", image)
    cv.waitKey(0)
    # plt.imshow(img, 'gray')
    # plt.show()

