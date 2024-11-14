import cv2

img = cv2.imread('C:/Users/harsh/OneDrive/Desktop/manyaaaa/k3.jpg', 0)
equalized = cv2.equalizeHist(img)

resized_equalized = cv2.resize(equalized, (500, 500), interpolation=cv2.INTER_AREA)

cv2.imshow('Equalized Image', resized_equalized)
cv2.waitKey(0)
cv2.destroyAllWindows()


