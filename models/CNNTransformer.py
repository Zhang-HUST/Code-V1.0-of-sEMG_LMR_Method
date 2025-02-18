import torch
import torch.nn as nn
from models.SCAttention import SCAttBlock
from models.model_utils import *
from utils.common_params import *


class CNNTransformer(nn.Module):
    def __init__(self):
        super(CNNTransformer, self).__init__()
        self.model_name = 'CNNTransformer'

        # CNNs Part 1: CNN Block 1 ---- SCConv Attention ---- CNN Block 2-5
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
                                         nn.MaxPool2d(kernel_size=(1, 8), stride=(1, 8)),
                                         nn.Dropout(p=0.2),
                                         )
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

        # Transformer
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=16 * C,  # input feature dim
            nhead=4,  # 4 self-attention layers in each multi-head self-attention layer in each encoder block
            dim_feedforward=256,  # 2 linear layers in each encoder block's feedforward network: dim 16 * C-->256--->16 * C
            dropout=0.2,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=4)

        # Classifier: linear_in_features ---- 32 ---- num_classes
        linear_in_features = 256 + 32
        self.linear_part = nn.Sequential(nn.Linear(in_features=linear_in_features, out_features=32),
                                         nn.BatchNorm1d(32),
                                         nn.ReLU(),
                                         nn.Dropout(p=0.2),
                                         nn.Linear(in_features=32, out_features=num_classes),
                                         )
        self.init_params()

    def forward(self, data):
        batch_size = data.shape[0]
        # CNNs
        cnn1_1_out = self.cnn_part1_1(data)
        cnn_att_out = self.cnn_attention(cnn1_1_out)
        cnn_out = self.cnn_part1_2(cnn_att_out)
        cnn_out = self.cnn_part2(cnn_out)
        branch1_out = cnn_out.reshape(batch_size, -1)
        # Transformer
        data_transformer = cnn_att_out.permute(0, 3, 2, 1).contiguous().view(cnn_att_out.size(0), cnn_att_out.size(-1),
                                                                             -1)
        transformer_output = self.transformer_encoder(data_transformer)
        branch2_out = torch.mean(transformer_output, dim=2)
        # Concat
        branchs_out = torch.cat((branch1_out, branch2_out), dim=1)
        out = self.linear_part(branchs_out)

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
