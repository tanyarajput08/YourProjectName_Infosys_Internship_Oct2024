import cv2

img = cv2.imread('C:/Users/harsh/OneDrive/Desktop/manyaaaa/k3.jpg', 0)
edges = cv2.Canny(img, 100, 200)

resized_edges = cv2.resize(edges, (500, 500), interpolation=cv2.INTER_AREA)

cv2.imshow('Edges', resized_edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
