import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import multilabel_confusion_matrix
from utils.common_utils import is_label_onehot, onehot2decimalism

"""Evaluation metrics for classification tasks: 
accuracy, precision, recall, f1, specificity, npv, confusion_matrix and normalized confusion_matrix"""


def get_specificity_npv(y1, y2):
    MCM = multilabel_confusion_matrix(y1, y2)
    specificity = []
    npv = []
    for i in range(MCM.shape[0]):
        confusion = MCM[i]
        TP = confusion[1, 1]
        TN = confusion[0, 0]
        FP = confusion[0, 1]
        FN = confusion[1, 0]
        specificity.append(TN / float(TN + FP))  # Sensitivity
        npv.append(TN / float(FN + TN))  # Negative predictive value（NPV)
    test_specificity = np.average(specificity)
    test_npv = np.average(npv)

    return test_specificity, test_npv


def get_accuracy(y_true, y_pre, decimal=3):
    if is_label_onehot(y_true, y_pre):
        y_true, y_pre = onehot2decimalism(y_true), onehot2decimalism(y_pre)
    else:
        pass
    accuracy = accuracy_score(y_true, y_pre)

    return round(accuracy * 100.0, decimal)


def get_precision(y_true, y_pre, decimal=3, average_type='macro'):
    if is_label_onehot(y_true, y_pre):
        y_true, y_pre = onehot2decimalism(y_true), onehot2decimalism(y_pre)
    else:
        pass
    precision = precision_score(y_true, y_pre, average=average_type, zero_division=0)

    return round(precision * 100.0, decimal)


def get_recall(y_true, y_pre, decimal=3, average_type='macro'):
    if is_label_onehot(y_true, y_pre):
        y_true, y_pre = onehot2decimalism(y_true), onehot2decimalism(y_pre)
    else:
        pass
    recall = recall_score(y_true, y_pre, average=average_type)

    return round(recall * 100.0, decimal)


def get_f1(y_true, y_pre, decimal=3, average_type='macro'):
    if is_label_onehot(y_true, y_pre):
        y_true, y_pre = onehot2decimalism(y_true), onehot2decimalism(y_pre)
    else:
        pass
    f1 = f1_score(y_true, y_pre, average=average_type)
    return round(f1 * 100.0, decimal)


# specificity
def get_specificity(y_true, y_pre, decimal=3, average_type='macro'):
    if average_type == 'macro':
        if is_label_onehot(y_true, y_pre):
            y_true, y_pre = onehot2decimalism(y_true), onehot2decimalism(y_pre)
        else:
            pass
        specificity = get_specificity_npv(y_true, y_pre)[0]
    else:
        raise ValueError('Only macro average_type is supported')

    return round(specificity * 100.0, decimal)


def get_npv(y_true, y_pre, decimal=3, average_type='macro'):
    if average_type == 'macro':
        if is_label_onehot(y_true, y_pre):
            y_true, y_pre = onehot2decimalism(y_true), onehot2decimalism(y_pre)
        else:
            pass
        npv = get_specificity_npv(y_true, y_pre)[1]
    else:
        raise ValueError('Only macro average_type is supported')

    return round(npv * 100.0, decimal)
