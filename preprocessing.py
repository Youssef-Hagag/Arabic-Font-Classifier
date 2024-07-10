import cv2
import numpy as np
import utils


# Example usage
input_folder = "./dataset/"
output_folder = "./preprocessed/"
target_size = (400, 400)
offset = 5

def preprocess_image(image, resize_width=100, resize_height=100):
    #========> Noise Filtering <============
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Enhance the median filter
    median_filtered = cv2.medianBlur(gray, 3)

    #========> white background, black text  <============
    dist = cv2.threshold(median_filtered, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    hist = cv2.calcHist([dist], [0], None, [256], [0, 256])
    inverted_image = dist.copy()
    if hist[0] > hist[-1]:
        temp = inverted_image.copy()
        inverted_image[temp <= 64] = 255
        inverted_image[temp >= 128] = 0
    else:
        inverted_image = dist
    
    
    #========> Rotating the image <=========
    
    edges = cv2.Canny(inverted_image, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

    rotated_image = inverted_image.copy()
    if lines is not None:
        angles = []
        for line in lines:
            rho, theta = line[0]
            angle = np.degrees(theta) - 90
            angles.append(angle)

        median_angle = np.median(angles)
        M = cv2.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), median_angle, 1.0)
        rotated_image = cv2.warpAffine(inverted_image, M, (image.shape[1], image.shape[0]),  borderValue=(255,255,255))

    #========> Cropping the image to only the text area <=========
    pre_cropped = rotated_image[:-20, :-20]  # Perform pre-cropping

    inverted_image = 255*(pre_cropped < 50).astype(np.uint8)  # To invert the text to white
    
    inverted_image = cv2.morphologyEx(inverted_image, cv2.MORPH_OPEN, np.ones(
        (2, 2), dtype=np.uint8))  # Perform noise filtering
    
    coords = cv2.findNonZero(inverted_image)  # Find all non-zero points (text)
    x, y, w, h = cv2.boundingRect(coords)  # Find minimum spanning bounding box
    
    x_start = x - offset if x - offset >= 0 else x
    y_start = y - offset if y - offset >= 0 else y
    
    # Crop the image - note we do this on the original image
    rect = pre_cropped[y_start:y+h+offset, x_start:x+w+offset]

    # Perform Opening to fill gaps
    filtered_image = cv2.morphologyEx(rect, cv2.MORPH_HITMISS, np.ones((2, 2), dtype=np.uint8))
        
    return filtered_image

def preprocess_images(images):
    preprocessed_images = []
    for image in images:
        preprocessed_image = preprocess_image(image)
        preprocessed_images.append(preprocessed_image)
    return preprocessed_images


if(__name__ == '__main__'):
    images, labels = utils.read_images_with_folder_labels(input_folder)
    preprocessed_images = preprocess_images(images)
    utils.write_images_with_labels(preprocessed_images, labels, output_folder)
