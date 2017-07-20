import cv2
import numpy as np
import matplotlib.pyplot as plt

def imConv(image_array,suanzi, kernelDim):
    image = image_array.copy()

    dim1,dim2 = image.shape
    newImageDelta = int((kernelDim - 1)/2)
    for i in range(newImageDelta,dim1-newImageDelta):
        for j in range(newImageDelta,dim2-newImageDelta):
            image[i,j] = (image_array[(i-newImageDelta):(i+newImageDelta+1),(j-newImageDelta):(j+newImageDelta+1)]*suanzi).sum()

    image = image*(255.0/image.max())

    return image
image_BGR = cv2.imread('images/12/Lena.png')
image_RGB = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2RGB)
image_Gray = cv2.cvtColor(image_RGB, cv2.COLOR_RGB2GRAY)

kernelDim = 3
kernel = np.ones((kernelDim,kernelDim), np.float32) / (kernelDim * kernelDim)
suanzi_x = np.array([[-1, 0, 1],
                     [ -1, 0, 1],
                     [ -1, 0, 1]])
#
# y方向的Prewitt算子
suanzi_y = np.array([[-1,-1,-1],
                     [ 0, 0, 0],
                     [ 1, 1, 1]])

RuiHua = np.array([[0,-1,0],
                   [ -1, 3, -1],
                   [0, -1, 0]])
plt.figure(figsize=(15,8))
plt.subplot(231)
plt.imshow(image_Gray, cmap='gray')

plt.subplot(232)
image_x = imConv(image_Gray, suanzi_x,kernelDim)
plt.imshow(image_x, cmap='gray')

plt.subplot(233)
image_y = imConv(image_Gray,suanzi_y, kernelDim)
plt.imshow(image_y, cmap='gray')

plt.subplot(234)
# 得到梯度矩阵
image_xy = np.sqrt(image_x**2+image_y**2)
# 梯度矩阵统一到0-255
image_xy = (255.0/image_xy.max())*image_xy
plt.imshow(image_xy, cmap='gray')

plt.subplot(235)
plt.imshow(imConv(image_Gray, kernel, kernelDim), cmap='gray')

plt.subplot(236)
plt.imshow(imConv(image_Gray, RuiHua, kernelDim), cmap='gray')
plt.show()


