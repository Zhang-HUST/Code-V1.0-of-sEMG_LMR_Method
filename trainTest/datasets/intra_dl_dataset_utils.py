from torch.utils.data import DataLoader
from trainTest.datasets.common_dataset_utils import (get_dl_datasets_of_repeated_experiments,
                                                     get_dl_datasets_of_cross_validation, MyDataset)
from utils.common_params import *


def get_dl_datasets(file_name, data_name: str, label_name: str):
    with open(file_name, 'rb') as f:
        data = np.load(f)[data_name]
        labels = np.load(f)[label_name]
    data_expanded = np.expand_dims(data, axis=1)
    print('data.shape: ', data.shape, ', data_expanded.shape: ', data_expanded.shape, ', labels.shape: ', labels.shape)

    if partitioning_method == 'repeated_experiments':
        all_x_train, all_y_train, all_x_valid, all_y_valid, all_x_test, all_y_test\
            = get_dl_datasets_of_repeated_experiments(data_expanded, labels)
    elif partitioning_method == 'cross_validation':
        all_x_train, all_y_train, all_x_valid, all_y_valid, all_x_test, all_y_test \
            = get_dl_datasets_of_cross_validation(data_expanded, labels)
    else:
        raise ValueError('Unsupported partitioning_method!')

    return all_x_train, all_y_train, all_x_valid, all_y_valid, all_x_test, all_y_test


def get_dl_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test):
    train_dataset = MyDataset(X_train, y_train)
    val_dataset = MyDataset(X_val, y_val)
    test_dataset = MyDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=train_batch, shuffle=True, drop_last=True)
    valid_loader = DataLoader(val_dataset, batch_size=valid_batch, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=test_batch, shuffle=False, drop_last=False)

    return train_loader, valid_loader, test_loader



