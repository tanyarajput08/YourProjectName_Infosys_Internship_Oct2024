import cv2

# Load the image and template as grayscale
img = cv2.imread('C:/Users/harsh/OneDrive/Desktop/manyaaaa/k3.jpg', 0)  # Grayscale image
template = cv2.imread('C:/Users/harsh/OneDrive/Desktop/manyaaaa/k3.jpg', 0)  # Grayscale template

# Perform template matching
result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)

# Find the location of the match
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
top_left = max_loc
h, w = template.shape[:2]
bottom_right = (top_left[0] + w, top_left[1] + h)

# Draw a rectangle around the matched region
cv2.rectangle(img, top_left, bottom_right, (255, 0, 0), 2)  # Red color for rectangle

# Resize the image to a smaller size for display
resized_img = cv2.resize(img, (600, 600))  # Resize to 600x600, adjust as needed

# Show the result
cv2.imshow('Detected Template', resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()



