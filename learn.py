from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
import joblib
import feature_extraction
import utils
import time

folder_path = './preprocessed/'

def train_classifier(X_train, y_train, vocabulary_size = 500):
    start_time = time.time()
    kmeans = feature_extraction.build_bovw(X_train, vocabulary_size)
    end_time = time.time()
    print("BOVW Built in", end_time - start_time, "seconds!")
    joblib.dump(kmeans, './output/kmeans.pkl')

    start_time = time.time()
    X_train_hist = feature_extraction.build_histograms(X_train, kmeans)
    end_time = time.time()
    print("Training Histogram Built in", end_time - start_time, "seconds!")
    joblib.dump(X_train_hist, './output/x_train_hist.pkl')
    
    start_time = time.time()
    clf = make_pipeline(StandardScaler(), SVC(C=10, gamma=0.001, kernel='rbf'))
    clf.fit(X_train_hist, y_train)
    end_time = time.time()
    print("Model Trained in", end_time - start_time, "seconds!")
    return clf, kmeans

if __name__ == "__main__":
    images, labels = utils.read_images_with_folder_labels(folder_path)
    print("Number of images:", len(images))
    print("Number of labels:", len(labels))
    clf, kmeans = train_classifier(images, labels)
    # Save the classifier using joblib or pickle
    joblib.dump(clf, './output/classifier.pkl')

