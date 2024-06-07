from project_io import *
from evaluation import *


if __name__ == '__main__':
    # Project constants
    all_classes = ['bird', 'dog', 'cat', 'deer', 'frog']
    batch_no = 1
    criteria = TestCriteria.ONLY_H

    # Load batch file and metadata, get all train and test pairs (50 train - 10 test, total of 60 image labels)
    meta_data, data_batch = get_raw_data(batch_no)
    all_data_pairs = get_train_and_test_images(meta_data, data_batch, all_classes)

    # Produce histograms for all data pairs in all classes
    all_hist_values = produce_histograms(all_data_pairs, data_batch)

    # Get most similar three records
    results = dict()
    for test_class in all_classes:  # for each data class...
        results[test_class] = get_most_similar_three(all_hist_values, all_classes, test_class)  # get most similar three

    # Print success rate according to results and criteria
    print_success_rate(results, criteria)

    # Save plots to disk
    save_results_to_file(results, criteria, data_batch)
