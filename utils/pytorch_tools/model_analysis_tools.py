import os
import torch
from torchviz import make_dot
from thop import profile
from torchsummary import summary as torchsummary
from torchinfo import summary as torchinfo
from utils.common_utils import printlog, make_dir
from fvcore.nn import parameter_count_table


"""
Model toolkit for single-branch input networks, including four functions:
1) Test model output dimension;
2) Computational model complexity: Time complexity: macs, madds; Space complexity: params size, memory required to store parameters;
3) Model visualization, printing the output size of each layer network;
4) Model visualization, save the model structure as.PDF or.jpg file, default.PDF type.
"""


class OneBranchModelTools:
    def __init__(self, target_model, input_size, batch_size, device):
        self.model = target_model
        self.input_size = input_size
        self.batch_size = batch_size
        self.device = device
        self.model = self.model.to(self.device)

    def summary_model(self, method='torchsummary'):
        if method == 'torchsummary':
            torchsummary(self.model, input_data=self.input_size, depth=6)
        elif method == 'torchinfo':
            input_size1 = (1, *self.input_size)
            torchinfo(self.model, input_size=input_size1, depth=6, verbose=1)
        elif method == 'fvcore':
            parameter_table = parameter_count_table(self.model, max_depth=4)
            print(parameter_table)
        else:
            raise ValueError('method must be summary or fvcore')

    def plot_model(self, model_name, save_format='pdf', show=True, verbose=False):
        printlog(info='Generate the model structure diagram by torchviz: ', time=True, line_break=True)
        input_data = torch.randn(self.batch_size, *self.input_size).to(self.device)
        dot = make_dot(self.model(input_data), params=dict(self.model.named_parameters()), show_attrs=verbose, show_saved=verbose)
        file_dir = os.path.join(os.getcwd(), 'modelVisualization')
        make_dir(file_dir)
        file_name = os.path.join(file_dir, model_name)
        dot.format = save_format
        if show:
            dot.view(filename=file_name, directory=file_dir, cleanup=True)
        else:
            dot.render(filename=file_name, directory=file_dir, cleanup=True)

    def calculate_complexity(self, verbose=True, print_log=True):
        printlog(info='Calculate model complexity: ', time=True, line_break=True)
        input_data = torch.randn(self.batch_size, *self.input_size).to(self.device)
        macs, params = profile(self.model, inputs=(input_data, ), verbose=verbose, report_missing=verbose)
        if print_log:
            print('macs:', macs, 'params:', params)
        return macs, params

    def test_output_shape(self):
        printlog(info='Test model output dimensions: ', time=True, line_break=True)
        input_data = torch.randn(self.batch_size, *self.input_size).to(self.device)
        output = self.model(input_data)
        print('output.shape', output.shape)
