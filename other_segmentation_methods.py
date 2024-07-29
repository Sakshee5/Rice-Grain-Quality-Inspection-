import cv2
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, MeanShift, estimate_bandwidth
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster
import matplotlib.pyplot as plt
from pre_processing import mask_and_crop

img_path = r"Images/DSC01912.JPG"
img = mask_and_crop(img_path)
image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
image_reshaped = image_rgb.reshape((-1, 3))

# Standardize features
scaler = StandardScaler()
image_standardized = scaler.fit_transform(image_reshaped)

# Helper function to display images
def display_image(title, image_data):
    plt.figure(figsize=(10, 10))
    plt.title(title)
    plt.imshow(image_data)
    plt.axis('off')
    plt.show()

# K-means clustering
print('Executing K means clustering')
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans_labels = kmeans.fit_predict(image_standardized)
kmeans_segmented = kmeans_labels.reshape(image_rgb.shape[:2])
display_image('K-means Segmentation', kmeans_segmented)

# DBSCAN clustering
print('Executing DBSCAN clustering')
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(image_standardized)
dbscan_segmented = dbscan_labels.reshape(image_rgb.shape[:2])
display_image('DBSCAN Segmentation', dbscan_segmented)

# Hierarchical clustering
print('Executing Hierarchical clustering')
hierarchical = linkage(image_standardized, method='ward')
hierarchical_labels = fcluster(hierarchical, t=5, criterion='maxclust')
hierarchical_segmented = hierarchical_labels.reshape(image_rgb.shape[:2])
display_image('Hierarchical Clustering Segmentation', hierarchical_segmented)

# Gaussian Mixture Model clustering
print('Executing GMM clustering')
gmm = GaussianMixture(n_components=5, random_state=0)
gmm_labels = gmm.fit_predict(image_standardized)
gmm_segmented = gmm_labels.reshape(image_rgb.shape[:2])
display_image('GMM Segmentation', gmm_segmented)

# Mean Shift clustering
print('Executing Mean Shift clustering')
bandwidth = estimate_bandwidth(image_standardized, quantile=0.2, n_samples=500)
mean_shift = MeanShift(bandwidth=bandwidth, bin_seeding=True)
mean_shift_labels = mean_shift.fit_predict(image_standardized)
mean_shift_segmented = mean_shift_labels.reshape(image_rgb.shape[:2])
display_image('Mean Shift Segmentation', mean_shift_segmented)
