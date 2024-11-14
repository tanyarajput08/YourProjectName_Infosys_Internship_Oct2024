import cv2
import numpy as np

# Read the image
img = cv2.imread('C:/Users/harsh/OneDrive/Desktop/manyaaaa/k3.jpg', 0)

# Create a kernel
kernel = np.ones((5,5), np.uint8)

# Dilation: Expands bright areas (objects)
dilated = cv2.dilate(img, kernel, iterations=1)

# Erosion: Shrinks bright areas (objects)
eroded = cv2.erode(img, kernel, iterations=1)

# Morphing through gradual dilation and erosion to see changes
morphing1 = cv2.dilate(img, kernel, iterations=2)  # More dilation (expansion)
morphing2 = cv2.erode(dilated, kernel, iterations=2)  # Erosion after dilation (shrink back)

# Resize the images to a smaller size
resized_dilated = cv2.resize(dilated, (500, 500), interpolation=cv2.INTER_AREA)
resized_eroded = cv2.resize(eroded, (500, 500), interpolation=cv2.INTER_AREA)
resized_morphing1 = cv2.resize(morphing1, (500, 500), interpolation=cv2.INTER_AREA)
resized_morphing2 = cv2.resize(morphing2, (500, 500), interpolation=cv2.INTER_AREA)

# Display images
cv2.imshow('Original Image', img)
cv2.imshow('Resized Dilated Image (Morphing 1)', resized_morphing1)
cv2.imshow('Resized Eroded Image (Morphing 2)', resized_morphing2)

cv2.waitKey(0)
cv2.destroyAllWindows()




