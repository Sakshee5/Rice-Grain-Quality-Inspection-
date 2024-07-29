import cv2
from PIL import Image
from collections import defaultdict
import numpy as np
from sklearn.cluster import KMeans

def quantize_image(image, n_colors):
    """Quantize an image to a specified number of colors using K-means clustering.
    
    Parameters:
    image (numpy.ndarray): The input image to be quantized, expected to be a 3D numpy array (height, width, 3).
    n_colors (int): The number of colors to reduce the image to.
    
    Returns:
    tuple: A tuple containing:
        - new_image (numpy.ndarray): The quantized image with reduced colors, same shape as the input image.
        - new_colors (numpy.ndarray): The array of RGB color values representing the new color palette, shape (n_colors, 3).
    """
    # Reshape the image to be a list of pixels
    pixels = image.reshape(-1, 3)
    
    # Apply KMeans to find the top n_colors in the image
    kmeans = KMeans(n_clusters=n_colors, random_state=42)
    kmeans.fit(pixels)
    
    # Replace each pixel with its closest cluster center
    new_colors = kmeans.cluster_centers_.astype(int)
    labels = kmeans.labels_
    new_image = new_colors[labels].reshape(image.shape)
    
    return new_image, new_colors

def find_unique_colors(img):
    # Load the image
    image = cv2.imread(img)

    # Reshape the image to be a list of pixels
    pixels = image.reshape(-1, 3)

    # Find all unique colors
    unique_colors = np.unique(pixels, axis=0)

    # Convert unique colors to a list of tuples for easier handling
    unique_colors_list = [tuple(color) for color in unique_colors]

    # Print the unique colors
    print(f"Unique colors: {unique_colors_list}")

    return unique_colors_list

def replace_color_exact(image, target_color, replacement_color):
    # Ensure the target and replacement colors are in BGR format
    target_color = np.array(target_color, dtype=np.uint8)
    replacement_color = np.array(replacement_color, dtype=np.uint8)
    
    # Create a mask where the target color matches exactly
    mask = np.all(image == target_color, axis=-1)
    
    # Replace the target color with the replacement color
    image[mask] = replacement_color

    return image

def get_defect_percent(image_path, good_grain):
    """
    :param image_path: path to concerned image file
    :param good_grain: RGB value of color segmented as good grain (needs to be manually extracted from by_color dict)
    :return: percent of defect (chalkiness + damage)
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    by_color = defaultdict(int)
    img = Image.open(image_path)  # RGB FORMAT
    for pixel in img.getdata():
        by_color[pixel] += 1

    print(by_color)

    total_grain = image.shape[0] * image.shape[1] - by_color[(0, 0, 0)]

    del by_color[good_grain]
    del by_color[(0, 0, 0)]

    defect_grain_percent = ((sum(list(by_color.values())) - 4000) / total_grain)

    return defect_grain_percent