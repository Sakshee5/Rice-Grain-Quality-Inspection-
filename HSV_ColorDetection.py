import cv2
import numpy as np

Winname = "Frame:"

def nothing(x):
    pass

cv2.namedWindow(Winname)
cv2.createTrackbar('H', Winname, 0, 179, nothing)
cv2.createTrackbar('S', Winname, 0, 255, nothing)
cv2.createTrackbar('V', Winname, 0, 255, nothing)
cv2.createTrackbar('H2', Winname, 179, 179, nothing)
cv2.createTrackbar('S2', Winname, 255, 255, nothing)
cv2.createTrackbar('V2', Winname, 255, 255, nothing)

img = cv2.imread(r"Images/DSC01902.JPG")

if img is None:
    print("Error: Image not found or cannot be read.")
    exit()

# Resize the image to fit the window
def resize_image(image, scale_percent):
    width = int(image.shape[1] * scale_percent / 400)
    height = int(image.shape[0] * scale_percent / 400)
    dim = (width, height)
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized

scale_percent = 30  # Percentage to resize (e.g., 30% of the original size)
img_resized = resize_image(img, scale_percent)

while True:
    H = cv2.getTrackbarPos('H', Winname)
    S = cv2.getTrackbarPos('S', Winname)
    V = cv2.getTrackbarPos('V', Winname)
    H2 = cv2.getTrackbarPos('H2', Winname)
    S2 = cv2.getTrackbarPos('S2', Winname)
    V2 = cv2.getTrackbarPos('V2', Winname)

    hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    lower_boundary = np.array([H, S, V])
    upper_boundary = np.array([H2, S2, V2])
    mask = cv2.inRange(hsv, lower_boundary, upper_boundary)

    final = cv2.bitwise_and(img_resized, img_resized, mask=mask)
    cv2.imshow(Winname, final)

    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
