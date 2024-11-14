import cv2
import numpy as np

img = cv2.imread('C:/Users/harsh/OneDrive/Desktop/manyaaaa/k3.jpg', 0)
kernel = np.ones((5,5), np.uint8)

dilation = cv2.dilate(img, kernel, iterations=1)
erosion = cv2.erode(img, kernel, iterations=1)

resized_dilation = cv2.resize(dilation, (500, 500), interpolation=cv2.INTER_AREA)
resized_erosion = cv2.resize(erosion, (500, 500), interpolation=cv2.INTER_AREA)

cv2.imshow('Dilated Image', resized_dilation)
cv2.imshow('Eroded Image', resized_erosion)
cv2.waitKey(0)
cv2.destroyAllWindows()
