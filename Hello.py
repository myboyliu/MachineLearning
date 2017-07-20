import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters,feature
import matplotlib as mpl

# 打开图像并转化成灰度图像
mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题

plt.figure(figsize=(17,13), facecolor='w')

image = cv2.imread('images/12/Lena.png', cv2.IMREAD_GRAYSCALE)
image = cv2.GaussianBlur(image
                         ,(3,3),0)
row = 3
col = 4
plt.subplot(row,col,1)
plt.imshow(image,cmap=plt.cm.gray)
plt.title(u'原图')

edges = filters.prewitt(image)
plt.subplot(row,col,2)
plt.imshow(edges,cmap=plt.cm.gray)
plt.title(u'Prewitt算子')

edges = filters.roberts(image)
plt.subplot(row,col,3)
plt.imshow(edges,cmap=plt.cm.gray)
plt.title(u'Roberts算子')

edges = filters.sobel(image)
plt.subplot(row,col,4)
plt.imshow(edges,cmap=plt.cm.gray)
plt.title(u'Sobel算子')

edges = cv2.Sobel(image,cv2.CV_64F,1,0,ksize=5)
plt.subplot(row,col,5)
plt.imshow(edges,cmap=plt.cm.gray)
plt.title(u'Sobel-X算子')

edges = cv2.Sobel(image,cv2.CV_64F,0,1,ksize=5)
plt.subplot(row,col,6)
plt.imshow(edges,cmap=plt.cm.gray)
plt.title(u'Sobel-Y算子')

edges = cv2.Laplacian(image,cv2.CV_64F)
plt.subplot(row,col,7)
plt.imshow(edges,cmap=plt.cm.gray)
plt.title(u'Laplacian算子')

edges = feature.canny(image)
plt.subplot(row,col,8)
plt.imshow(edges,cmap=plt.cm.gray)
plt.title(u'canny算子 $\sigma$=1')

edges = feature.canny(image, sigma=3)
plt.subplot(row,col,9)
plt.imshow(edges,cmap=plt.cm.gray)
plt.title(u'canny算子 $\sigma$=3')

i1 = cv2.GaussianBlur(image, (15,15), 5)
i2 = cv2.GaussianBlur(image, (21,21), 5)
GaussSub = cv2.subtract(i1, i2)
plt.subplot(row,col,10)
plt.imshow(GaussSub,cmap=plt.cm.gray)
plt.title(u'高斯差分')

plt.suptitle(u'边缘算子比较', fontsize=18)
plt.show()