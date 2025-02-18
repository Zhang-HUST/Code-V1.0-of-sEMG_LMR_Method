import os
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from utils.common_params import *


def get_file_name(path, subject, subjects_list):
    if subject not in subjects_list:
        raise ValueError('subject not in subjects_list_global', subjects_list)

    file_name = os.path.join(path, ''.join(['Sub', subject, '_targetTrainData.npz']))

    return file_name


def get_save_path(base_path, model_name, subject):
    absolute_path = os.path.join(base_path, model_name, ''.join(['Sub', subject]))
    relative_path = os.path.relpath(absolute_path, base_path)

    return {'absolute_path': absolute_path, 'relative_path': relative_path}


def get_ml_datasets_of_repeated_experiments(data, label):
    all_x_train, all_y_train, all_x_test, all_y_test = [], [], [], []
    sss = StratifiedShuffleSplit(n_splits=K_of_repeated_experiments, test_size=test_ratio, random_state=None)

    for train_index, test_index in sss.split(data, label):
        X_train, Y_train = data[train_index], label[train_index]
        X_test, Y_test = data[test_index], label[test_index]
        all_x_train.append(X_train)
        all_y_train.append(Y_train)
        all_x_test.append(X_test)
        all_y_test.append(Y_test)

    return all_x_train, all_y_train, all_x_test, all_y_test


def get_ml_datasets_of_cross_validation(data, label):
    all_x_train, all_y_train, all_x_test, all_y_test = [], [], [], []
    skf = StratifiedKFold(n_splits=K_of_cross_validation, shuffle=True, random_state=None)

    for train_index, test_index in skf.split(data, label):
        X_train, Y_train = data[train_index], label[train_index]
        X_test, Y_test = data[test_index], label[test_index]
        all_x_train.append(X_train)
        all_y_train.append(Y_train)
        all_x_test.append(X_test)
        all_y_test.append(Y_test)

    return all_x_train, all_y_train, all_x_test, all_y_test


def get_dl_datasets_of_repeated_experiments(data, label):
    all_x_train, all_y_train, all_x_valid, all_y_valid, all_x_test, all_y_test = [], [], [], [], [], []
    sss = StratifiedShuffleSplit(n_splits=K_of_repeated_experiments, test_size=test_ratio, random_state=None)
    sssForValid = StratifiedShuffleSplit(n_splits=1, test_size=valid_to_train_ratio, random_state=None)

    for train_index, test_index in sss.split(data, label):
        X_trainAndValid, Y_trainAndValid = data[train_index], label[train_index]
        X_test, Y_test = data[test_index], label[test_index]
        all_x_test.append(X_test)
        all_y_test.append(Y_test)

        X_train, Y_train, X_valid, Y_valid = None, None, None, None
        for train_index_, valid_index_ in sssForValid.split(X_trainAndValid, Y_trainAndValid):
            X_train, Y_train = X_trainAndValid[train_index_], Y_trainAndValid[train_index_]
            X_valid, Y_valid = X_trainAndValid[valid_index_], Y_trainAndValid[valid_index_]
        all_x_train.append(X_train)
        all_y_train.append(Y_train)
        all_x_valid.append(X_valid)
        all_y_valid.append(Y_valid)

    return all_x_train, all_y_train, all_x_valid, all_y_valid, all_x_test, all_y_test


def get_dl_datasets_of_cross_validation(data, label):
    all_x_train, all_y_train, all_x_valid, all_y_valid, all_x_test, all_y_test = [], [], [], [], [], []
    skf = StratifiedKFold(n_splits=K_of_cross_validation, shuffle=True, random_state=None)
    sssForValid = StratifiedShuffleSplit(n_splits=1, test_size=valid_to_train_ratio, random_state=None)

    for train_index, test_index in skf.split(data, label):
        X_trainAndValid, Y_trainAndValid = data[train_index], label[train_index]
        X_test, Y_test = data[test_index], label[test_index]
        all_x_test.append(X_test)
        all_y_test.append(Y_test)

        X_train, Y_train, X_valid, Y_valid = None, None, None, None
        for train_index_, valid_index_ in sssForValid.split(X_trainAndValid, Y_trainAndValid):
            X_train, Y_train = X_trainAndValid[train_index_], Y_trainAndValid[train_index_]
            X_valid, Y_valid = X_trainAndValid[valid_index_], Y_trainAndValid[valid_index_]
        all_x_train.append(X_train)
        all_y_train.append(Y_train)
        all_x_valid.append(X_valid)
        all_y_valid.append(Y_valid)

    return all_x_train, all_y_train, all_x_valid, all_y_valid, all_x_test, all_y_test


class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.double)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
