from trainTest.datasets.common_dataset_utils import (get_ml_datasets_of_repeated_experiments,
                                                     get_ml_datasets_of_cross_validation)
from utils.common_params import *


def get_ml_datasets(file_name, feature_name: str, label_name: str):
    with open(file_name, 'rb') as f:
        features = np.load(f)[feature_name]
        labels = np.load(f)[label_name]
    features = features.reshape(features.shape[0], -1)
    print('features.shape: ', features.shape, ', labels.shape: ', labels.shape)

    if partitioning_method == 'repeated_experiments':
        all_x_train, all_y_train, all_x_test, all_y_test = get_ml_datasets_of_repeated_experiments(features, labels)
    elif partitioning_method == 'cross_validation':
        all_x_train, all_y_train, all_x_test, all_y_test = get_ml_datasets_of_cross_validation(features, labels)
    else:
        raise ValueError('Unsupported partitioning_method!')

    return all_x_train, all_y_train, all_x_test, all_y_test
