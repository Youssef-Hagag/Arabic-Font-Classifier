import cross_validation
import utils

def analyze_vocabulary_size():
    for vocab in range(50,601,50):
        cross_validation.cross_validate(vocab, n_folds=3)

def create_accuracy_list():
    average_scores = []  # Array to store the average scores

    for vocabulary_size in range(50,601,50):
        file_path = f"./output/analysis/fold_scores_{vocabulary_size}.txt"
        
        # Read the scores from the file
        scores = utils.read_scores_from_file(file_path)
        
        # Calculate the average score
        average_score = sum(scores) / len(scores)
        average_scores.append(average_score)
    
    print(average_scores)

    return average_scores


# analyze_vocabulary_size()
avg_accuracy_list = create_accuracy_list()
utils.create_vocab_size_accuracy_graph(avg_accuracy_list)