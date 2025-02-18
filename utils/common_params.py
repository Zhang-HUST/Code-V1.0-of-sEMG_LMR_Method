import torch
import numpy as np

# 1. Data preprocessing and feature extraction
# 1.1. window and step: The window length and step length of overlapping windows (ms).
# 1.2. C: The number of channels for the input sEMG, C is 9 (SIAT-LLMD dataset) or 8 (HI-SP dataset).
# 1.3. raw_fs: The raw frequency (1920 Hz and 2000 Hz for SIAT-LLMD and HI-SP datasets, respectively).
# 1.4. tar_fs: The target frequency of the resampling method.
window = 128
step = int(0.75 * window)
C = 9
raw_fs = 1920
tar_fs = 1000

# 2. Dataset partitioning in the intra-subject scenario
# 2.1. partitioning_method: 'repeated_experiments' or 'cross_validation'.
# 2.2. K_of_repeated_experiments: When partitioning_method == 'repeated_experiments', we repeat the experiments K times.
# In each trial, a different random seed will be generated to ensure a different split.
# 2.3. test_ratio: The proportion of test set to total dataset (partitioning_method == 'repeated_experiments').
# 2.4. K_of_cross_validation:
# When partitioning_method == 'cross_validation', we use K-fold cross-validation to split the dataset, thus ensuring a complete separation of training and test data.
# In each trial, 1 fold is test set, (K−1) fold *0.9 is training set, and (K−1) fold*0.1 is validation set.
# 2.5. valid_to_train_ratio: The proportion of validation set to training set.
# (For both partitioning_method == 'repeated_experiments' and 'cross_validation').
partitioning_method = 'repeated_experiments'
K_of_repeated_experiments = 5
test_ratio = 0.2
K_of_cross_validation = 5
valid_to_train_ratio = 0.1

# 3. Model construction, training, and testing
# 3.1. num_classes: Number of classes, which is 5 (SIAT-LLMD dataset), and 7 or 3 (HI-SP dataset)
num_classes = 5

# 3.2. Number of iterations, initial learning rate, optimizer, learning rate decay, early stopping, and loss function
# 1)'epoch': int, maximum number of training rounds;
# 2)'initial_lr': Initial learning rate, 0.01 by default;
# 3)'initial_lr_low_tl': Effective only on the target domain training task of transfer learning;
# 4) 'initial_lr_high_tl': Effective only on the target domain training task of transfer learning;
# 5)'optimizer': optimizer, default 'Adam', one of ['Adam', 'RMSprop'];
# 6)'lr_scheduler': learning rate decay, scheduler_type: one of ['None', 'StepLR', 'MultiStepLR', 'ExponentialLR',
#                                                           AutoWarmupLR', 'GradualWarmupLR' ,'ReduceLROnPlateau'];
# 7)'early_stopping': early stopping.
max_epoch = 100
dl_callbacks = {'epoch': max_epoch,
                'initial_lr': 0.01,
                'initial_lr_low_tl': 0.01 * np.sqrt(0.1),
                'initial_lr_high_tl': 0.01,
                'optimizer': 'Adam',
                'lr_scheduler': {'scheduler_type': 'GradualWarmupLR',
                                 'params': {
                                     'StepLR': {'step_size': int(0.2 * max_epoch), 'gamma': np.sqrt(0.1)},
                                     'MultiStepLR': {
                                         'milestones': [int(0.2 * max_epoch), int(0.4 * max_epoch),
                                                        int(0.6 * max_epoch),
                                                        int(0.8 * max_epoch)],
                                         'gamma': np.sqrt(0.1)},
                                     'ExponentialLR': {'gamma': 0.9},
                                     'AutoWarmupLR': {'num_warm': 10},
                                     'GradualWarmupLR': {'multiplier': 1, 'total_epoch': 10},
                                     'ReduceLROnPlateau': {'mode': 'max', 'factor': np.sqrt(0.1),
                                                           'patience': 10, 'verbose': False,
                                                           'threshold': 0.00001, 'min_lr': 0.0001}}
                                 },
                'early_stopping': {'use_es': True, 'params': {'patience': 10, 'verbose': False, 'delta': 0.00001}}}

# 3.3. Drawing and saving settings during model training and testing
# 1) use_tqdm: Use progress bar to print output inside progress bar if True, use hiddenlayer to print output if False;
# 2) train_plot: Whether to use hiddenlayer Canvas to plot the training process;
# 3) print_interval: How many epochs to print an output or update tqdm once;
# 4) model_eval: Enable model.eval() before evaluating the model with validation and test sets;
# 5) test_metrics: list of metrics evaluated on the test set;
# 6) confusion_matrix: dist, refer to 'metrics/get_test_metrics function '.
dl_train_test_utils = {'use_tqdm': False,
                       'train_plot': True,
                       'print_interval': 2,
                       'model_eval': {'valid': True, 'test': True},
                       'test_metrics': ['accuracy', 'precision', 'recall', 'f1', 'specificity', 'npv'],
                       'confusion_matrix': {'get_cm': True,
                                            'params': {'show_type': 'all', 'plot': True, 'save_fig': True,
                                                       'save_results': True, 'cmap': 'YlGnBu'}},
                       }
ml_train_test_utils = {'save_model': True, 'parameter_optimization': True,
                       'test_metrics': ['accuracy', 'precision', 'recall', 'f1', 'specificity', 'npv'],
                       'confusion_matrix': {'get_cm': True,
                                            'params': {'show_type': 'all', 'plot': True, 'save_fig': True,
                                                       'save_results': True, 'cmap': 'YlGnBu'}},
                       }

# 3.4. Other settings
train_batch, valid_batch, test_batch = 32, 32, 32
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
