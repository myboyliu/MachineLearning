import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread('images/12/opencv-logo.png')

plt.figure(figsize=(13,7))
plt.subplot(231)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original')
plt.xticks([]), plt.yticks([])

kernel = np.ones((3,3),np.float32)/9
dst = cv2.filter2D(img,-1,kernel)
plt.subplot(232),
plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
plt.title('3*3')
plt.xticks([]), plt.yticks([])

kernel = np.ones((5,5),np.float32)/25
dst = cv2.filter2D(img,-1,kernel)
plt.subplot(233)
plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
plt.title('5*5')
plt.xticks([]), plt.yticks([])

kernel = np.ones((10,10),np.float32)/100
dst = cv2.filter2D(img,-1,kernel)
plt.subplot(234)
plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
plt.title('9*9')
plt.xticks([]), plt.yticks([])

kernel = np.ones((20,20),np.float32)/300
dst = cv2.filter2D(img,-1,kernel)
plt.subplot(235)
plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
plt.title('19*19')
plt.xticks([]), plt.yticks([])

blur = cv2.blur(img,(3,3)) # blur就是做的平滑均值滤波
plt.subplot(236)
plt.imshow(cv2.cvtColor(blur, cv2.COLOR_BGR2RGB))
plt.title('3*3')
plt.xticks([]), plt.yticks([])
plt.show()