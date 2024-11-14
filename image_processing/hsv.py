import cv2

img = cv2.imread('C:/Users/harsh/OneDrive/Desktop/manyaaaa/k3.jpg')
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

resized_hsv_img = cv2.resize(hsv_img, (500, 500), interpolation=cv2.INTER_AREA)

cv2.imshow('HSV Image', resized_hsv_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
