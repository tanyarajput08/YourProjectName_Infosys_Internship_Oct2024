import cv2

img = cv2.imread('C:/Users/harsh/OneDrive/Desktop/manyaaaa/k3.jpg')
(h, w) = img.shape[:2]
center = (w // 2, h // 2)

matrix = cv2.getRotationMatrix2D(center, 45, 1.0)
rotated = cv2.warpAffine(img, matrix, (w, h))

# Resize the rotated image to a smaller size
resized_rotated = cv2.resize(rotated, (500, 500), interpolation=cv2.INTER_AREA)

cv2.imshow('Resized Rotated Image', resized_rotated)
cv2.waitKey(0)
cv2.destroyAllWindows()
