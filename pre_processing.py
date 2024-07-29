import cv2
import numpy as np
import time

def mask_and_crop(img_path):
    # Read and resize the image
    img = cv2.imread(img_path)
    img = cv2.resize(img, (800, 1200))
    
    # Convert the image to HSV color space
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define the color range for masking
    lower = np.array([70, 110, 0])
    upper = np.array([109, 255, 255])

    # Create the mask and the result image
    mask = cv2.inRange(imgHSV, lower, upper)
    imgResult = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(mask))

    # Use morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    imgResult = cv2.morphologyEx(imgResult, cv2.MORPH_CLOSE, kernel)
    imgResult = cv2.morphologyEx(imgResult, cv2.MORPH_OPEN, kernel)

    # Trim the black borders
    def crop(frame):
        coords = np.argwhere(frame[:, :, 0] != 0)  # Find coordinates where the image is not black
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1  # slices are exclusive at the top
        trimmed = frame[y0:y1, x0:x1]
        return trimmed

    final = crop(imgResult)
    
    # Add padding
    final = cv2.copyMakeBorder(final, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=0)

    return final

# Example usage
# img_path = r"Images/DSC01912.JPG"
# start = time.time()
# img = mask_and_crop(img_path)
# end = time.time()
# print(f"Processing time for {end - start:.4f} seconds")
# cv2.imshow('Processed Image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


def mask_and_crop_high_complexity(img_path):
    """
    Img ---> converted to HSV color space to detect background ---> create a mask of grain ---> make sure all background
    pixels are set to black colour ---> crop grain ---> add 5 pixel black background padding on each side
    """
    img = cv2.resize(cv2.imread(img_path), (800, 1200))
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower = np.array([70, 110, 0])
    upper = np.array([109, 255, 255])

    mask = cv2.inRange(imgHSV, lower, upper)
    imgResult = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(mask))

    def neighbours(im):
        for i in range(im.shape[1]):  # set top and bottom row of pixels to 0 incase they are not masked right
            im[0, i, :] = 0
            im[im.shape[0] - 1, i, :] = 0

        for i in range(im.shape[0]):  # set right and left column of pixels to 0
            im[i, 0, :] = 0
            im[i, im.shape[1] - 1, :] = 0

        neighbours = [1, 2, 3, 4, 5]
        # check top bottom right and left pixels of every pixel till it's fifth neighbour and if they are all zero then
        # set pixel in question to zero incase they were not masked right

        for val in neighbours:
            for i in range(val, im.shape[0] - val):
                for j in range(1, im.shape[1] - val):
                    if np.any(im[i, j, :]) != 0:
                        if np.all(im[i + val, j, :] == 0) and np.all(im[i, j + val, :] == 0) and np.all(
                                im[i - val, j, :] == 0) and np.all(im[i, j - val, :] == 0):
                            im[i, j, :] = 0

        return im

    def trim(frame):
        # crop top
        if not np.sum(frame[0]):
            return trim(frame[1:])
        # crop bottom
        elif not np.sum(frame[-1]):
            return trim(frame[:-2])
        # crop left
        elif not np.sum(frame[:, 0]):
            return trim(frame[:, 1:])
        # crop right
        elif not np.sum(frame[:, -1]):
            return trim(frame[:, :-2])
        return frame

    final = cv2.copyMakeBorder(trim(neighbours(imgResult)), 10, 10, 10, 10, cv2.BORDER_CONSTANT, None, value=0)

    return final
