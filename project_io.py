from custom_plot import *
from convenience import *
from my_histogram import *


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict


def get_raw_data(batch_no):
    sample_data_batch = f'/Users/mertarican/PycharmProjects/HW2_19011622/venv/cifar-10-batches-py/data_batch_{batch_no}'
    meta_file_path = '/Users/mertarican/PycharmProjects/HW2_19011622/venv/cifar-10-batches-py/batches.meta'
    return unpickle(meta_file_path), unpickle(sample_data_batch)


class DataPair:
    def __init__(self, class_name, training_data, test_data):
        self.class_name = class_name
        self.training_data = training_data
        self.test_data = test_data


def get_train_and_test_images(meta_data, data_batch, class_names):
    data_pairs = []
    for name in class_names:
        i = meta_data['label_names'].index(name)
        class_name = meta_data['label_names'][i]
        class_data = get_first_n_occurrences(data_batch['labels'], i, 60)
        data_pairs.append(DataPair(class_name, class_data[:50], class_data[50:]))
    return data_pairs


def save_results_to_file(results, criteria, data_batch):
    cons = ['best', 'second', 'third']
    for test_class in results.keys():
        for index, label in enumerate(results[test_class]):
            all_desc = [f'original image: {test_class.upper()}', 'Y image', 'H image']
            original_image = get_image(data_batch, label)
            y_image = np.array(get_y_value(original_image))
            h_image = normalize_vector(np.array(get_h_value(original_image)), 255)
            all_images = [original_image, y_image, h_image]
            best_three = results[test_class][label].results[criteria]
            for i, best in enumerate(best_three):
                name = best.data_class
                all_images.append(get_image(data_batch, best.label))
                all_desc.append(f'{cons[i]} match: {name.capitalize()}\nsimilarity:{round(best.cosine_similarity, 2)} ({best.criteria})')
            success = len(list(filter(lambda x: x.data_class == test_class, results[test_class][label].results[criteria]))) != 0
            status = 'Successful' if success else 'Failed'
            write_plots_to_file(test_class, all_images, 32, 32, 2, 3, all_desc,
                                f'{test_class.capitalize()} Test {index+1}: {status}',
                                f'{test_class.capitalize()} Test {index+1}')