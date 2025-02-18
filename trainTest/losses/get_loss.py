import torch.nn as nn


def get_classification_loss():
    criterion = nn.CrossEntropyLoss()

    return criterion
