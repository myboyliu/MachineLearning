import cv2
import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
image_Ori = cv2.imread('images/12/Lena.png')
image_Dirty = cv2.imread('images/12/Dirty.png')
mask = cv2.imread('images/12/Lena_Dirty.png',0)

mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题

def transfer(img):
    height = img.shape[0]
    width = img.shape[1]
    rects = ((width - 169, height - 22, width, height),(1, 1, 164, 21)) #水印区域

    mask = np.zeros(img.shape[:2], np.uint8)
    for rect in rects:
        x1, y1, x2, y2 = rect
        cv2.rectangle(mask, (x1, y1), (x2, y2), (255, 255, 255), -1)
        img = cv2.inpaint(img, mask, 10.0, cv2.INPAINT_TELEA) #蒙版
    return img

# plt.figure(figsize=(10,5), facecolor='w')
# plt.subplot(141)
# plt.imshow(cv2.cvtColor(image_Ori, cv2.COLOR_BGR2RGB))
# plt.title('原始图片')
# plt.xticks([]), plt.yticks([])
#
# plt.subplot(142)
# plt.imshow(cv2.cvtColor(image_Dirty, cv2.COLOR_BGR2RGB))
# plt.title('污点图片')
# plt.xticks([]), plt.yticks([])
#
# plt.subplot(143)
# img = transfer(img)
# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# plt.title('修复图片')
# plt.xticks([]), plt.yticks([])
image_Dirty = transfer(image_Dirty)
plt.imshow(image_Dirty)
plt.suptitle('低通滤波器比较')
plt.show()