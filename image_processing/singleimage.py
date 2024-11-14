import cv2

image = cv2.imread("C:/Users/harsh/OneDrive/Desktop/manyaaaa/k3.jpg")

resized_image = cv2.resize(image, (500, 500))


cv2.imshow('Woman', resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(resized_image.shape)
