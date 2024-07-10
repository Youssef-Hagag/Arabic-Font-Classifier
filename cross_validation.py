from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import learn
import classify
import utils
import time

# Perform cross-validation
if __name__ == '__main__':
    print("Reading images...")
    start_time = time.time()
    images, labels = utils.read_images_with_folder_labels('./preprocessed/')
    end_time= time.time()
    print("Images read in", end_time - start_time, "seconds!")


def cross_validate(vocabulary_size = 500, n_folds = 5):
    fold_scores = []
    output_file = f"./output/analysis/fold_scores_{vocabulary_size}.txt"  # Name of the output file

    for _ in range(n_folds):
        print(f"==============> Fold #{_+1} <=================")
        # Split dataset into training and testing sets for this fold
        X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2)

        # Train the model u sing learn.py
        start_time = time.time()
        clf, kmeans = learn.train_classifier(X_train, y_train, vocabulary_size)
        end_time = time.time()
        # Classify samples using classify.py
        predictions = classify.classify(clf, kmeans, X_test)
        
        # Evaluate the model
        score = accuracy_score(y_test, predictions)
        print(f"Fold #{_+1} Score:", score)
        fold_scores.append(score)

    # Calculate the average score
    average_score = sum(fold_scores) / len(fold_scores)
    print("Average accuracy:", average_score)
    utils.write_txt(output_file, fold_scores)

