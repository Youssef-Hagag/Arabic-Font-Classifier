import csv
import os
import cv2
import matplotlib.pyplot as plt


def read_csv_file(file_path):
  data = []
  with open(file_path, 'r') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
      data.append(row)
  return data

def write_csv(array1, array2, column_names, file_path):
  data = list(zip(array1, array2))
  with open(file_path, 'w', newline='') as file:
    csv_writer = csv.writer(file)
    csv_writer.writerow(column_names)
    csv_writer.writerows(data)

def write_images_with_labels(images, labels, output_folder):
  idx_list = [0,0,0,0]
  for image, label in zip(images, labels):
    label_folder = os.path.join(output_folder, str(label))
    os.makedirs(label_folder, exist_ok=True)
    image_path = os.path.join(label_folder, f"{label}_{idx_list[label]}.png")
    idx_list[label] += 1
    cv2.imwrite(image_path, image)

def read_images(file_paths):
  images = []
  for file_path in file_paths:
    image = cv2.imread(file_path)
    images.append(image)
  return images

def read_images_from_folder(folder_path):
  images = []
  image_list = os.listdir(folder_path)
  sorted_image_list = sorted(image_list, key=natural_sort_key)

  for file_name in sorted_image_list:
    file_path = os.path.join(folder_path, file_name)
    image = cv2.imread(file_path)
    images.append(image)
  return images

def natural_sort_key(s):
    """Key function for natural sorting."""
    import re
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def write_images(images, folder_path, namePrefix):
  for i, image in enumerate(images):
    image_path = os.path.join(folder_path, f"{namePrefix}_{i}.png")
    cv2.imwrite(image_path, image)


def read_images_with_folder_labels(root_path):
  images = []
  labels = []
  label_counter = 0

  for folder_name in os.listdir(root_path):
    folder_path = os.path.join(root_path, folder_name)
    if os.path.isdir(folder_path):
      for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        image = cv2.imread(file_path)
        images.append(image)
        labels.append(label_counter)
      label_counter += 1

  return images, labels

def write_txt(filename, list):
  with open(filename, 'w') as file:
    for item in list:
      file.write("%s\n" % item)

def read_scores_from_file(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()
        scores = [float(line.strip()) for line in lines if line.strip()]
    return scores

def create_vocab_size_accuracy_graph(average_scores):
  vocabulary_sizes = list(range(50, 601, 50))

  # Plot line graph
  plt.figure(figsize=(10, 6))
  plt.plot(vocabulary_sizes, average_scores, marker='o', color='green', linestyle='-')
  plt.title('Line Graph of Average Scores')
  plt.xlabel('Vocabulary Size')
  plt.ylabel('Average Score (3 Folds)')
  plt.grid(True)
  plt.show()