import cv2
from sklearn.metrics import accuracy_score
import joblib
import feature_extraction
import preprocessing
import utils
import time
import argparse

parser = argparse.ArgumentParser(description='NN Classification module')

parser.add_argument('--single', type=str, default="",
                    help='path to a single message to classify it')

parser.add_argument('--evaluate', type=str, default="",
                    help='path to a single message to calculate accuray of classification')

def classify(clf, kmeans, X_test):
    # Build histograms for test images
    X_test_hist = feature_extraction.build_histograms(X_test, kmeans)
   
    # Make predictions
    y_pred = clf.predict(X_test_hist)
    
    return y_pred
  
def classify_image(image):
  # Load the trained kmeans model
    kmeans = joblib.load('./output/kmeans.pkl')
    
    # Load the trained classifier
    clf = joblib.load('./output/classifier.pkl')
    
    preprocessed =image
    
    try:
      num_channels = image.shape[-1] if len(image.shape) == 3 else 1
      if num_channels != 1:
        preprocessed = preprocessing.preprocess_image(image)
    except Exception as e:
      print("Error in preprocessing image: ", e)
      return "Error in preprocessing image"

    y_pred = classify(clf, kmeans, [preprocessed])
    
    return y_pred[0]

if __name__ == "__main__":
    args = parser.parse_args()

    test_folder_path = './test/'
    
    # Load the trained kmeans model
    kmeans = joblib.load('./output/kmeans.pkl')
    
    # Load the trained classifier
    clf = joblib.load('./output/classifier.pkl')
    
    if(args.single):
        # Load the test image
        image = cv2.imread(args.single)
        X_test = []
        start_time = time.time()
        preprocessed = preprocessing.preprocess_image(image)
        X_test.append(preprocessed)
        end_time = time.time()
        print("Preprocessing Finished in", end_time - start_time, "seconds!")
        
        # Classify the test image
        y_pred = classify(clf, kmeans, X_test)
        
        print("Predicted label: ", y_pred[0])
        exit(0)

    elif(args.evaluate):
      # Load the test images and their labels
      images = utils.read_images_from_folder(args.evaluate)
      print("Number of test images: ", len(images))
      
      y_pred = []
      timing = []
      
      for i in range(len(images)):
        start_time = time.time()
        preprocessed = preprocessing.preprocess_image(images[i])
        if i == 15:
           cv2.imshow("title" ,preprocessed)
           cv2.waitKey(0)
        res = classify(clf, kmeans, [preprocessed])
        end_time = time.time()
        timing.append(round(end_time - start_time,3))
        y_pred.append(res[0])      
      
      # Write predicted and actual labels to a CSV file
      utils.write_txt('./output/results.txt', y_pred)
      utils.write_txt('./output/time.txt', timing)

      print("Predictions are written to results.txt")
      print("Times are written to results.txt")
      exit(0)

    else:
      # Load the test images and their labels
      images, Y_test = utils.read_images_with_folder_labels(test_folder_path)
      print("Number of test images: ", len(images), "Number of Y_test images: ", len(Y_test))
      
      X_test = []
      
      start_time = time.time()
      for i in range(len(images)):
        X_test.append(preprocessing.preprocess_image(images[i]))
      end_time = time.time()
      print("Preprocessing Finished in", end_time - start_time, "seconds!")
      

      # Classify the test images
      y_pred = classify(clf, kmeans, X_test)
      
      # Write predicted and actual labels to a CSV file
      utils.write_csv(Y_test, y_pred, ["Actual", "Predicted"], "./output/predictions.csv")

      print("Predictions written to predictions.csv")

      # Calculate accuracy
      accuracy = accuracy_score(Y_test, y_pred)
      print("Accuracy: {:.2f}%".format(accuracy * 100))
