import cv2

image = cv2.imread("C:/Users/harsh/OneDrive/Desktop/manyaaaa/k3.jpg")
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Resize the grayscale image to a smaller size
resized_gray_image = cv2.resize(gray_image, (500, 500), interpolation=cv2.INTER_AREA)

cv2.imwrite("C:/Users/harsh/OneDrive/Desktop/manyaaaa/k6.jpg", resized_gray_image)

cv2.imshow('Resized Grayscale Image', resized_gray_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

