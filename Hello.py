import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cv2
from skimage import filters
import scipy.signal as ss

mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题

image = cv2.imread('images/12/USA.png')
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

plt.figure(figsize=(25,5), facecolor='w')
plt.subplot(241)
plt.title('原图')
plt.imshow(image_gray, cmap='gray')

plt.subplot(242)
plt.title('Sobel-X')
SobelX = np.array([[-1, 0, 1],
                   [ -2, 0, 2],
                   [ -1, 0, 1]])
image_x = ss.convolve2d(image_gray, SobelX, mode="same")
plt.imshow(image_x, cmap='gray')

plt.subplot(243)
plt.title('Sobel-Y')
SobelY = np.array([[-1,-2,-1],
                   [ 0, 0, 0],
                   [ 1, 2, 1]])
image_y = ss.convolve2d(image_gray, SobelY, mode="same")
plt.imshow(image_y, cmap='gray')

grident = image_x* image_x+ image_y * image_y
gridentSqrt = np.sqrt(grident)
b = (255/gridentSqrt.max()) * gridentSqrt

plt.subplot(244)
plt.title('Sobel')
image_Sobel = b.copy()
plt.imshow(image_Sobel, cmap='gray')

scale = 4;
cuteoff = scale*np.mean(b)

plt.subplot(245)
plt.title('Sobel二值化')

b[b>=cuteoff] = 255
b[b<cuteoff] = 0
plt.imshow(b, cmap='gray')

rThresh = scale * np.mean(grident)
image_NMS = b.copy()
for i in range(1, image_gray.shape[0]-1):
    sbx = image_x[i]
    sby = image_y[i]
    pre = b[i-1]
    cur = b[i]
    lst = b[i+1]
    for j in range(1, image_gray.shape[1]-1):
        if(cur[j] > cuteoff and (
                (sbx[j] > sby[j] and cur[j] > cur[j-1] and cur[j] > cur[j+1]) or
                (sby[j] > sbx[j] and cur[j] > pre[j] and cur[j] > lst[j])
        )):
            image_NMS[i,j]=0

plt.subplot(246)
plt.title('Sobel非极大值抑制1')
plt.imshow(image_NMS, cmap='gray')

image_NMS = b.copy()
for i in range(1, image_gray.shape[0]-1):
    for j in range(1, image_gray.shape[1]-1):
        if (grident[i,j] > rThresh and
                ((grident[i,j] > grident[i-1,j] and grident[i,j] > grident[i+1, j])
                 or
                     (grident[i,j] > grident[i, j-1] and grident[i,j] > grident[i, j+1]))):
            image_NMS[i,j]=1

plt.subplot(247)
plt.title('Sobel非极大值抑制2')
plt.imshow(image_NMS, cmap='gray')
plt.show()