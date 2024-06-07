import numpy as np
from image_functions import *


def cos_similarity(vector1, vector2):
    return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))


class HistogramPair:
    def __init__(self, label, y_value_hist, h_value_hist):
        self.label = label
        self.y_value_hist = y_value_hist
        self.h_value_hist = h_value_hist

    def __str__(self):
        return f'label: {self.label}' + f' y_value: {self.y_value_hist}' + f' h_value: {self.h_value_hist}'

    def __repr__(self):
        return self.__str__()


class TrainAndTestHistograms:
    def __init__(self):
        self.training_histograms = []
        self.test_histograms = []

    def __str__(self):
        return f'training_histograms: {self.training_histograms}' + f' test_histograms: {self.test_histograms}'

    def __repr__(self):
        return self.__str__()


def create_hist(vector, bins, normalize=True):
    hist = []
    min_val = min(vector)
    step = (max(vector) - min_val) / bins
    step = step if step != 0 else 1
    for i in range(bins):
        hist.append(0)
    for i in vector:
        bin = min(bins-1, int((i - min_val) / step))
        hist[bin] += 1
    return np.array(hist) / (1024.0 if normalize else 1)


def produce_histograms(data_pairs, data_batch):
    _hist_values = dict()
    for data_pair in data_pairs:  # For each raw data pair (train-test pair)...
        hist_pair = TrainAndTestHistograms()

        for label in data_pair.training_data:  # create histograms for each training image ...
            y_value_hist = create_hist(get_y_value(get_image(data_batch, label)), 255)
            h_value_hist = create_hist(get_h_value(get_image(data_batch, label)), 360)
            training_hist = HistogramPair(label, y_value_hist, h_value_hist)
            hist_pair.training_histograms.append(training_hist)

        for label in data_pair.test_data:  # create histograms for each test image ...
            y_value_hist = create_hist(get_y_value(get_image(data_batch, label)), 255)
            h_value_hist = create_hist(get_h_value(get_image(data_batch, label)), 360)
            test_hist = HistogramPair(label, y_value_hist, h_value_hist)
            hist_pair.test_histograms.append(test_hist)

        _hist_values[data_pair.class_name] = hist_pair  # save histogram pairs in a dict for later use
    return _hist_values
