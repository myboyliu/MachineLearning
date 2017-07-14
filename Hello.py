import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import data,filters

def imconv(image_array,suanzi):
    image = image_array.copy()     # 原图像矩阵的深拷贝
    dim1,dim2 = image.shape
    # 对每个元素与算子进行乘积再求和(忽略最外圈边框像素)
    for i in range(1,dim1-1):
        for j in range(1,dim2-1):
            image[i,j] = (image_array[(i-1):(i+2),(j-1):(j+2)]*suanzi).sum()

    # 由于卷积后灰度值不一定在0-255之间，统一化成0-255
    image = image*(255.0/image.max())

    # 返回结果矩阵
    return image

suanzi_x = np.array([[-1, 0, 1],
                     [ -1, 0, 1],
                     [ -1, 0, 1]])

# y方向的Prewitt算子
suanzi_y = np.array([[-1,-1,-1],
                     [ 0, 0, 0],
                     [ 1, 1, 1]])


# 打开图像并转化成灰度图像
image = cv2.imread('images/12/Lena.png', cv2.IMREAD_GRAYSCALE)

# 转化成图像矩阵
image_array = np.array(image)

# 得到x方向矩阵
image_x = imconv(image_array,suanzi_x)

# 得到y方向矩阵
image_y = imconv(image_array,suanzi_y)

# 得到梯度矩阵
image_xy = np.sqrt(image_x**2+image_y**2)
# 梯度矩阵统一到0-255
image_xy = (255.0/image_xy.max())*image_xy
edges = filters.prewitt(image)
# 绘出图像
plt.figure(figsize=(10,8), facecolor='w')
plt.subplot(2,2,1)
plt.imshow(image_array,cmap=plt.cm.gray)
plt.axis("off")
plt.subplot(2,2,2)
plt.imshow(edges,cmap=plt.cm.gray)
plt.axis("off")
plt.subplot(2,2,3)
plt.imshow(image_y,cmap=plt.cm.gray)
plt.axis("off")
plt.subplot(2,2,4)
plt.imshow(image_xy,cmap=plt.cm.gray)
plt.axis("off")
plt.show()