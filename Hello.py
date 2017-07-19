import cv2
import numpy as np
import matplotlib.pyplot as plt

def getMatrix(matrix, x, y, length):
    matrix1 = matrix[:(x+length+1), :(y+length+1)]
    return matrix1[(x-length):, (y-length):]

image_BGR = cv2.imread('images/12/Cat.jpg')
image_RGB = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2RGB)
image_Gray = cv2.cvtColor(image_RGB, cv2.COLOR_RGB2GRAY)

kernelDim = 3

newImageDelta = int((kernelDim - 1)/2)
image = np.ones((image_Gray.shape[0] + 2 * newImageDelta) * (image_Gray.shape[1]+ 2 * newImageDelta),np.uint8) * 255
image = image.reshape(image_Gray.shape[0]+ 2 * newImageDelta, image_Gray.shape[1]+ 2 * newImageDelta)
delta = newImageDelta
for x in range(image_Gray.shape[0]):
    for y in range(image_Gray.shape[1]):
        image[x+delta,y+delta] = image_Gray[x,y]

kernel = np.ones((kernelDim,kernelDim), np.float32) / (kernelDim * kernelDim)
GaoSi = np.array(([1,2,1],[2,4,2],[1,2,1]), dtype=np.int)
Ruihua = np.array(([0,-1,0],[-1,3,-1],[0,-1,0]), dtype=np.int)
for x in range(image_Gray.shape[0]):
    for y in range(image_Gray.shape[1]):
        xNew = x + delta
        yNew = y + delta
        matrixNew = getMatrix(image, xNew, yNew, delta)
        value = np.sum(matrixNew * Ruihua)
        image[xNew, yNew] = value
plt.figure(figsize=(15,8))
plt.subplot(121)
plt.imshow(image_Gray, cmap='gray')

plt.subplot(122)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB), cmap='gray')
plt.show()


