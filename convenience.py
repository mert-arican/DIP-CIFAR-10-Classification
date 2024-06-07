from enum import Enum


def get_first_n_occurrences(array, value, n, start=0):
    return [j + start for j, item in enumerate(array[start:]) if item == value][:n]


def init_dict_with_list(array, init_value):
    dic = dict()
    for element in array:
        dic[element] = init_value.copy()  # Wow! It has been a long time since the last side effect bug. (~ 1 hour lost)
    return dic


def capitalize(array):
    for i in range(len(array)):
        array[i] = array[i].capitalize()


def normalize_vector(array, upper_bound):
    max_val = max(array)
    array = (array / max_val) * upper_bound
    return array


def print_dict(dict):
    for key in dict:
        print(key, ': ', dict[key])


class SimilarityResult:
    def __init__(self, data_class, label, cosine_similarity, criteria):
        self.data_class = data_class
        self.label = label
        self.cosine_similarity = cosine_similarity
        self.criteria = criteria

    def __lt__(self, other):
        return self.cosine_similarity < other.cosine_similarity

    def __repr__(self):
        return f'class: {self.data_class}' + f' - similarity: {self.cosine_similarity}' + f' - hist: {self.criteria}'


class TestResult:
    def __init__(self, test_label):
        self.label = test_label
        self.results = init_dict_with_list(TestCriteria.allCases, list())


class TestCriteria(Enum):
    ONLY_Y = 1
    ONLY_H = 2
    BOTH_H_Y = 3


TestCriteria.allCases = [TestCriteria.ONLY_Y, TestCriteria.ONLY_H, TestCriteria.BOTH_H_Y]


TestCriteria.title = {
    TestCriteria.ONLY_Y: 'Considering only Y values:',
    TestCriteria.ONLY_H: 'Considering only H values:',
    TestCriteria.BOTH_H_Y: 'Considering both Y and H values:'
    }


TestCriteria.description = {
    TestCriteria.ONLY_Y: 'Y value',
    TestCriteria.ONLY_H: 'H value'
}
