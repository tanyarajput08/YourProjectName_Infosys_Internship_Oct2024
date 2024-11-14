import cv2
import numpy as np

img1 = cv2.imread("C:/Users/harsh/OneDrive/Desktop/manyaaaa/k2.jpg")
img2 = cv2.imread("C:/Users/harsh/OneDrive/Desktop/manyaaaa/k3.jpg")

img1 = cv2.resize(img1, (500, 500))
img2 = cv2.resize(img2, (500, 500))

h_concat = np.hstack((img1, img2))
v_concat = np.vstack((img1, img2))

# Resize the vertically concatenated image
v_concat_resized = cv2.resize(v_concat, (500, 500), interpolation=cv2.INTER_AREA)

cv2.imshow('Horizontal Concatenation', h_concat)
cv2.imshow('Vertical Concatenation', v_concat_resized)

cv2.waitKey(0)
cv2.destroyAllWindows()
