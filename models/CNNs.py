import torch.nn as nn
from models.SCAttention import SCAttBlock
from models.model_utils import *
from utils.common_params import *


class CNNs1D(nn.Module):
    def __init__(self):
        super(CNNs1D, self).__init__()
        self.model_name = 'CNNs1D'

        # CNNs Part 1: CNN Block 1 ---- SCConv Attention ---- CNN Block 2 ---- CNN Block 3
        kernel_size_1, stride_1 = get_params_of_cnnBlock_1(window)
        self.cnn_part1_1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1, 5), stride=(1, 1),
                                                   padding='same'),
                                         nn.BatchNorm2d(16),
                                         nn.ReLU(),
                                         nn.MaxPool2d(kernel_size=kernel_size_1, stride=stride_1),
                                         nn.Dropout(p=0.2),
                                         )
        self.cnn_attention = SCAttBlock(op_channel=16, group_kernel_size=(1, 3))
        self.cnn_part1_2 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 3), stride=(1, 1),
                                                   padding='same'),
                                         nn.BatchNorm2d(32),
                                         nn.ReLU(),
                                         nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4)),
                                         nn.Dropout(p=0.2),
                                         nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 3), stride=(1, 1),
                                                   padding='same'),
                                         nn.BatchNorm2d(64),
                                         nn.ReLU(),
                                         nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 1)),
                                         nn.Dropout(p=0.2),
                                         )

        # CNNs Part 2: CNN Block 4 ---- CNN Block 5
        kernel_size_4, stride_4 = get_params_of_cnnBlock_4(C)
        kernel_size_5 = get_params_of_cnnBlock_5(C)
        self.cnn_part2 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 1), stride=(1, 1),
                                                 padding='same'),
                                       nn.BatchNorm2d(128),
                                       nn.ReLU(),
                                       nn.MaxPool2d(kernel_size=kernel_size_4, stride=stride_4),
                                       nn.Dropout(p=0.2),
                                       nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 1), stride=(1, 1),
                                                 padding='same'),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.AvgPool2d(kernel_size=kernel_size_5, stride=(1, 1)),
                                       nn.Dropout(p=0.2),
                                       )

        # Classifier: 256 ---- 32 ---- num_classes
        self.linear_part = nn.Sequential(nn.Linear(in_features=256, out_features=32),
                                         nn.BatchNorm1d(32),
                                         nn.ReLU(),
                                         nn.Dropout(p=0.2),
                                         nn.Linear(in_features=32, out_features=num_classes),
                                         )
        self.init_params()

    def forward(self, data):
        batch_size = data.shape[0]
        cnn_out1_1 = self.cnn_part1_1(data)
        cnn_att_out = self.cnn_attention(cnn_out1_1)
        cnn_out1_2 = self.cnn_part1_2(cnn_att_out)
        cnn_out_2 = self.cnn_part2(cnn_out1_2)
        cnn_out_2 = cnn_out_2.reshape(batch_size, -1)
        out = self.linear_part(cnn_out_2)

        return out

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def get_model_name(self):
        return self.model_name
