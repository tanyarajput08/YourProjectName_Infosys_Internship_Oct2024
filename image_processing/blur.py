import cv2

img = cv2.imread("C:/Users/harsh/OneDrive/Desktop/manyaaaa/k3.jpg")
blur = cv2.GaussianBlur(img, (11, 11), 0)
resized_blur = cv2.resize(blur, (int(blur.shape[1] * 0.5), int(blur.shape[0] * 0.5)), interpolation=cv2.INTER_AREA)
cv2.imshow('Blurred Image', resized_blur)
cv2.waitKey(0)
cv2.destroyAllWindows()
