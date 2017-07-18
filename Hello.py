import cv2
import numpy as np

image_BGR = cv2.imread('images/12/Cat.jpg')
image_RGB = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2RGB)
r,g,b = cv2.split(image_RGB)
image_Ex = cv2.merge([r,g,b])
t = np.zeros((image_BGR.shape[0],image_BGR.shape[1]), dtype=image_BGR.dtype)
image_Ex = cv2.merge([r,g, t])
# image_Ex = cv2.merge([r,t, b])
# image_Ex = cv2.merge([t,g, b])

cv2.imshow('r', image_Ex)
cv2.waitKey(0)
cv2.destroyAllWindows()
