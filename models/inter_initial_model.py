import torch.nn as nn
from models.SCAttention import SCAttBlock
from models.model_utils import *
from utils.common_params import *


class CNNBiGRU(nn.Module):
    def __init__(self):
        super(CNNBiGRU, self).__init__()
        self.model_name = 'CNN-BiGRU'
        self.Extractor = Extractor()
        self.Classifier = Classifier()
        self.init_params()

    def forward(self, data):
        rnn_out = self.Extractor(data)
        out = self.Classifier(rnn_out)

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
            elif isinstance(m, nn.LSTM) or isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_normal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)

    def get_model_name(self):
        return self.model_name


class Extractor(nn.Module):
    def __init__(self):
        super(Extractor, self).__init__()
        # CNNs: CNN Block 1 ---- SCConv Attention ---- CNN Block 2
        kernel_size_1, stride_1 = get_params_of_cnnBlock_1(window)
        self.cnn_part_1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1, 5), stride=(1, 1),
                                                  padding='same'),
                                        nn.BatchNorm2d(16),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=kernel_size_1, stride=stride_1),
                                        nn.Dropout(p=0.2),
                                        )
        self.cnn_attention = SCAttBlock(op_channel=16, group_kernel_size=(1, 3))
        self.cnn_part_2 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 3), stride=(1, 1),
                                                  padding='same'),
                                        nn.BatchNorm2d(32),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4)),
                                        nn.Dropout(p=0.2),
                                        )
        self.rnn_part1 = nn.GRU(input_size=C * 32, hidden_size=64, num_layers=1, batch_first=True,
                                bidirectional=True)
        self.rnn_part2 = nn.GRU(input_size=128, hidden_size=128, num_layers=1, batch_first=True, bidirectional=True)

    def forward(self, data):
        cnn_out = self.cnn_part_1(data)
        cnn_att_out = self.cnn_attention(cnn_out)
        cnn_out = self.cnn_part_2(cnn_att_out)
        # Adjust the output of CNNs to the input shape of RNNs
        cnn_out = cnn_out.permute(0, 3, 2, 1).contiguous().view(cnn_out.size(0), cnn_out.size(-1), -1)
        rnn_out, _ = self.rnn_part1(cnn_out)
        rnn_out, _ = self.rnn_part2(rnn_out)
        rnn_out = rnn_out[:, -1, :]

        return rnn_out


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.linear_part = nn.Sequential(nn.Linear(in_features=256, out_features=32),
                                         nn.BatchNorm1d(32),
                                         nn.ReLU(),
                                         nn.Dropout(p=0.2),
                                         nn.Linear(in_features=32, out_features=num_classes),
                                         )

    def forward(self, data):
        out = self.linear_part(data)
        return out
