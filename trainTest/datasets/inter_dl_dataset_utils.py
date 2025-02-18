from sklearn.model_selection import StratifiedShuffleSplit
from utils.common_params import *


def get_source_datasets(file_names, data_name: str, label_name: str):
    all_data_expanded, all_label = [], []
    for file_name in file_names:
        with open(file_name, 'rb') as f:
            data = np.load(f)[data_name]
            labels = np.load(f)[label_name]
        data_expanded = np.expand_dims(data, axis=1)
        all_data_expanded.extend(data_expanded)
        all_label.extend(labels)
    all_data_expanded, all_label = np.array(all_data_expanded), np.array(all_label)
    print('all_data_expanded.shape: ', all_data_expanded.shape, ', all_label.shape: ', all_label.shape)

    x_train, y_train, x_valid, y_valid, x_test, y_test = [], [], [], [], [], []
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=None)
    sssForValid = StratifiedShuffleSplit(n_splits=1, test_size=valid_to_train_ratio, random_state=None)

    for train_index, test_index in sss.split(all_data_expanded, all_label):
        x_trainAndValid, y_trainAndValid = all_data_expanded[train_index], all_label[train_index]
        x_test, y_test = all_data_expanded[test_index], all_label[test_index]
        for train_index_, valid_index_ in sssForValid.split(x_trainAndValid, y_trainAndValid):
            x_train, y_train = x_trainAndValid[train_index_], y_trainAndValid[train_index_]
            x_valid, y_valid = x_trainAndValid[valid_index_], y_trainAndValid[valid_index_]

    return x_train, y_train, x_valid, y_valid, x_test, y_test






