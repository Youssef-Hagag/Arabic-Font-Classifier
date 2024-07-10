import cv2
import numpy as np

def extract_sift_features(image):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return descriptors

def build_histograms(images, kmeans):
    histograms = []
    for img in images:
        sift_features = extract_sift_features(img)
        histogram = np.zeros(len(kmeans))
        if sift_features is not None:
            for feature in sift_features:
                idx = np.argmin(np.linalg.norm(kmeans - feature, axis=1))
                histogram[idx] += 1
        histograms.append(histogram)
    return histograms

def build_bovw(images, vocabulary_size):
    descriptors_list = []
    for image in images:
        descriptors = extract_sift_features(image)
        if descriptors is not None:
            descriptors_list.append(descriptors)
    descriptors_stack = np.vstack(descriptors_list)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, kmeans_centers = cv2.kmeans(descriptors_stack, vocabulary_size, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    return kmeans_centers
