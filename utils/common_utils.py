import os
import datetime
import numpy as np
import pandas as pd


def printlog(info, time=True, line_break=True):
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if time:
        print("\n" + "==========" * 8 + "%s" % nowtime)
    else:
        pass
    if line_break:
        print(info + '...\n')
    else:
        print(info)


def is_string_in_list(lst, target_string):
    return target_string in lst


def get_feature_list(channels, feature_type, concatenation=False):
    all_fea_names = []
    for i in range(len(channels)):
        channel = channels[i]
        if concatenation:
            all_fea_names.extend([''.join([channel, '_', feature_type[i]]) for i in range(len(feature_type))])
        else:
            all_fea_names.append([''.join([channel, '_', feature_type[i]]) for i in range(len(feature_type))])

    return np.array(all_fea_names)


def make_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    else:
        pass


def convert_to_2d(arr):
    arr = np.array(arr)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    else:
        pass

    return arr


def is_label_onehot(y_true, y_pre):
    y_true, y_pre = convert_to_2d(y_true), convert_to_2d(y_pre)
    if y_true.shape[1] == 1 or y_pre.shape[1] == 1:
        return False
    else:
        return True


def onehot2decimalism(arr, keep_dims=False):
    return np.argmax(arr, axis=1, keepdims=keep_dims)


def calculate_and_save_metrics(path, subjects):
    """
    The results were calculated for all subjects and saved as two CSV files:
        1. all_metrics_averaged_results.csv: Contains all test results for all subjects and computes the mean and standard deviation.
        2. alone_subject_averaged_results.csv: Mean and standard deviation of the test results for each subject.

    Parameters:
    - subjects: List of subjects
    -path: This holds the path
    """

    df1_metrics = []
    df2_metrics_mean = []
    df2_metrics_std = []
    columns = None

    for subject_order in subjects:
        subject = 'Sub' + subject_order
        metrics_file_name = os.path.join(path, subject, 'test_metrics.csv')
        if not os.path.exists(metrics_file_name):
            print(f"Subject: {subject}: {metrics_file_name}, file does not exist!")
        else:
            df = pd.read_csv(metrics_file_name, header=0, index_col=0)
            df1_metrics.extend(df.T.values[:-2, :])
            df2_metrics_mean.append(df.T.values[-2, :])
            df2_metrics_std.append(df.T.values[-1, :])
            columns = df.index

    print(f'Saving the average results of all tested metrics for all subjects...')
    df1 = pd.DataFrame(df1_metrics, index=range(1, len(df1_metrics) + 1), columns=columns)
    mean_row = df1.mean().to_frame().T
    mean_row.index = ['mean']
    df1 = pd.concat([df1, mean_row])
    std_row = df1[:-1].std().to_frame().T
    std_row.index = ['std']
    df1 = pd.concat([df1, std_row]).round(4)
    df1_save_name = os.path.join(path, 'all_metrics_averaged_results.csv')
    df1.to_csv(df1_save_name, index=True)

    print(f'Saving the averaged results of the individual subject test metrics averaged...')
    df2_metrics_mean, df2_metrics_std = np.round(np.array(df2_metrics_mean), 4), np.round(np.array(df2_metrics_std), 4)
    df2_metrics = np.array([f"{df2_metrics_mean[i, j]}+{df2_metrics_std[i, j]}"
                            for i in range(df2_metrics_mean.shape[0])
                            for j in range(df2_metrics_mean.shape[1])])
    df2_metrics = df2_metrics.reshape(df2_metrics_mean.shape)
    df2 = pd.DataFrame(df2_metrics, index=[f'Sub{i}' for i in subjects], columns=columns)
    mean_row = np.round(np.mean(df2_metrics_mean, axis=0), 4)
    std_row = np.round(np.std(df2_metrics_mean, axis=0), 4)
    df2.loc['mean'] = mean_row
    df2.loc['std'] = std_row
    df2_save_name = os.path.join(path, 'alone_subject_averaged_results.csv')
    df2.to_csv(df2_save_name, index=True)
    print("All results saved!")
