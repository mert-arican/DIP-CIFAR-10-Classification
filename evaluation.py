from convenience import *
from my_histogram import cos_similarity


def get_most_similar_three(all_hist_values, all_classes, test_class):
    test_histograms = all_hist_values[test_class].test_histograms  # get all test histograms
    results = dict()
    for test_histogram_pair in test_histograms:  # and for each test histogram found...
        result_record = TestResult(test_histogram_pair.label)
        for training_class in all_classes:  # for each data class
            training_images = all_hist_values[training_class].training_histograms  # get training images
            for train_hist_pair in training_images:  # and for each training image...
                y_similarity = cos_similarity(test_histogram_pair.y_value_hist, train_hist_pair.y_value_hist)
                h_similarity = cos_similarity(test_histogram_pair.h_value_hist, train_hist_pair.h_value_hist)
                y_res = SimilarityResult(training_class, train_hist_pair.label, y_similarity, 'Y value')
                h_res = SimilarityResult(training_class, train_hist_pair.label, h_similarity, 'H value')
                result_record.results[TestCriteria.ONLY_Y].append(y_res)
                result_record.results[TestCriteria.ONLY_H].append(h_res)
                result_record.results[TestCriteria.BOTH_H_Y].extend([h_res, y_res])
        for criteria in result_record.results.keys():
            result_record.results[criteria] = sorted(result_record.results[criteria], reverse=True)[:3]
        results[test_histogram_pair.label] = result_record
    return results


def print_success_rate(results, criteria):
    overall_success = dict()
    print(f'\n{TestCriteria.title[criteria]}')
    for test_class in results.keys():  # for every data class ...
        overall_success[test_class] = 0
        for label in results[test_class]:  # for every label in test images of that class ...
            # check whether best three contains any member has the same class with the test class
            success = len(list(filter(lambda x: x.data_class == test_class, results[test_class][label].results[criteria]))) != 0
            overall_success[test_class] += 1 if success else 0
    for test_class in overall_success.keys():
        print(f'\tSuccess rate for {test_class}: {overall_success[test_class]*10}%')
    print(f'\n\tOverall success rate: {sum(overall_success.values())*2}%')
