import pandas as pd
import hiddenlayer as hl
from tqdm import tqdm
from utils.common_utils import make_dir
from trainTest.metrics.metrics_utils import get_accuracy
from trainTest.metrics.get_test_metrics import GetTestResults, PlotConfusionMatrix
from trainTest.losses.get_loss import get_classification_loss
from trainTest.optimizers.get_optimizer import Optimizer
from trainTest.early_stopping.early_stopping import EarlyStopping
from trainTest.lr_schedulers.get_lr_scheduler import LrScheduler
from utils.common_params import *


def train_test_dl_models_source(settings, model, train_loader, valid_loader, test_loader):
    # 1. Model, test results save absolute path and filename
    utils, callbacks = dl_train_test_utils, dl_callbacks
    basic_path = settings['save_path']
    make_dir(basic_path)
    model_save_name = ''.join([basic_path, '/source_model.pt'])
    history_save_csv_name = ''.join([basic_path, '/history.csv'])
    early_stopping, scheduler, history_save_pic_name = None, None, None

    # 2. callbacks
    optimizer = Optimizer(model, optimizer_type=callbacks['optimizer'], lr=callbacks['initial_lr']).get_optimizer()
    if callbacks['lr_scheduler']['scheduler_type'] == 'None':
        print('Without using learning rate scheduler: ')
    else:
        print('Using learning rate scheduler: ', callbacks['lr_scheduler']['scheduler_type'])
        scheduler = LrScheduler(optimizer, callbacks['lr_scheduler']['scheduler_type'],
                                callbacks['lr_scheduler']['params'], callbacks['epoch']).get_scheduler()

    if callbacks['early_stopping']['use_es']:
        print('Using model early stopping: ')
        early_stopping = EarlyStopping(patience=callbacks['early_stopping']['params']['patience'],
                                       verbose=callbacks['early_stopping']['params']['verbose'],
                                       delta=callbacks['early_stopping']['params']['delta'],
                                       path=model_save_name)

    # 3. train_test_utils
    history = hl.History()
    progress_bar, canvas = None, None
    if utils['use_tqdm']:
        progress_bar = tqdm(total=callbacks['epoch'])
    if utils['train_plot']:
        history_save_pic_name = ''.join([basic_path, '/history.jpg'])
        canvas = hl.Canvas()

    for e in range(1, 1 + callbacks['epoch']):
        train_predict_labels, train_true_labels, train_loss = [], [], 0.0
        criterion = get_classification_loss()
        # 4. Training
        model.train()
        for data, target in train_loader:
            data = data.to(device=device)
            label = target
            target = target.to(device=device).view(-1)
            predict = model(data)
            loss = criterion(predict, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _, predict_index = torch.max(predict.detach().to(device='cpu'), dim=1)
            train_predict_labels.append(predict_index)
            train_true_labels.append(label)
            train_loss += loss.detach().to(device='cpu').item()
        train_predict_labels = torch.cat(train_predict_labels, dim=0).view(-1).numpy()
        train_true_labels = torch.cat(train_true_labels, dim=0).view(-1).numpy()
        acc_train_epoch = get_accuracy(train_true_labels, train_predict_labels, decimal=5)
        loss_train_epoch = np.round(train_loss / len(train_loader), 5)

        # 5. Valid
        if utils['model_eval']['valid']:
            model.eval()
        valid_predict_labels, valid_true_labels, valid_loss = [], [], 0.0
        with torch.no_grad():
            for data, target in valid_loader:
                data = data.to(device=device)
                label = target
                target = target.to(device=device).view(-1)
                predict = model(data)
                loss = criterion(predict, target)
                _, predict_index = torch.max(predict.detach().to(device='cpu'), dim=1)
                valid_predict_labels.append(predict_index)
                valid_true_labels.append(label)
                valid_loss += loss.detach().to(device='cpu').item()
        valid_predict_labels = torch.cat(valid_predict_labels, dim=0).view(-1).numpy()
        valid_true_labels = torch.cat(valid_true_labels, dim=0).view(-1).numpy()
        acc_valid_epoch = get_accuracy(valid_true_labels, valid_predict_labels, decimal=5)
        loss_valid_epoch = np.round(valid_loss / len(valid_loader), 5)
        # The learning rate is updated only after the validation is finished
        if callbacks['lr_scheduler']['scheduler_type'] in ['StepLR', 'MultiStepLR', 'AutoWarmupLR', 'GradualWarmupLR',
                                                           'ExponentialLR']:
            scheduler.step()
        elif callbacks['lr_scheduler']['scheduler_type'] == 'ReduceLROnPlateau':
            scheduler.step(acc_valid_epoch / 100)
        else:
            pass

        # 6. Drawing or printing
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        history.log(e, train_acc=acc_train_epoch, valid_acc=acc_valid_epoch, train_loss=loss_train_epoch,
                    valid_loss=loss_valid_epoch, lr=lr)
        if utils['train_plot']:
            with canvas:
                canvas.draw_plot([history["train_acc"], history["valid_acc"]],
                                 labels=["Train accuracy", "Valid accuracy"])
                canvas.draw_plot([history["train_loss"], history["valid_loss"]],
                                 labels=["Train loss", "Valid loss"])
                canvas.draw_plot([history["lr"]], labels=["lr"])

        if utils['use_tqdm']:
            progress_bar.update(1)
            progress_bar.set_description(
                f"train_acc: {acc_train_epoch:.3f}, valid_acc: {acc_valid_epoch:.3f}, train_loss: {loss_train_epoch:.3f}, valid_loss: {loss_valid_epoch:.3f}")
        else:
            if e == 1 or e % utils['print_interval'] == 0:
                history.progress()

        if callbacks['early_stopping']['use_es']:
            early_stopping(loss_valid_epoch, model)
            # early_stop is set to True when the early stop condition is reached
            if early_stopping.early_stop:
                print("Early stopping, save the model: ")
                break

    # 7. If early stopping is not used, the model is saved after the model training ends
    if not callbacks['early_stopping']['use_es']:
        print('Save the model after the last round of training: ')
        torch.save(model, model_save_name)

    # 8. Save the model training process
    print('Save the model training process: ')
    history_metrics = pd.DataFrame(history.history).T
    history_metrics.to_csv(history_save_csv_name, index=True)
    if utils['train_plot']:
        canvas.save(history_save_pic_name)

    # 9. Model test was performed only after the training
    if utils['use_tqdm']:
        progress_bar.close()
    model = torch.load(model_save_name)
    test_predict_labels, test_true_labels, test_loss = [], [], 0.0
    if utils['model_eval']['test']:
        model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device=device)
            label = target
            predict = model(data)
            _, predict_index = torch.max(predict.detach().to(device='cpu'), dim=1)
            test_predict_labels.append(predict_index)
            test_true_labels.append(label)
    test_predict_labels = torch.cat(test_predict_labels, dim=0).view(-1).numpy()
    test_true_labels = torch.cat(test_true_labels, dim=0).view(-1).numpy()

    # 10. test results
    test_results_utils = GetTestResults(utils['test_metrics'], test_true_labels, test_predict_labels, decimal=5)
    test_metrics = test_results_utils.calculate()
    test_metrics_dict = dict(zip(utils['test_metrics'], test_metrics))
    print('Test results: ')
    for key, value in test_metrics_dict.items():
        print(f"{key}:  {value}.")


    # 11. Calculate confusion matrix
    if utils['confusion_matrix']['get_cm']:
        print('Calculate confusion matrix: ')
        cm_save_jpg_name = ''.join([basic_path, '/confusion_matrix.jpg'])
        cm_save_csv_name = ''.join([basic_path, '/confusion_matrix.xlsx'])
        plot_confusion_matrix = PlotConfusionMatrix(test_true_labels, test_predict_labels,
                                                    label_type=None,
                                                    show_type=utils['confusion_matrix']['params']['show_type'],
                                                    plot=utils['confusion_matrix']['params']['plot'],
                                                    save_fig=utils['confusion_matrix']['params']['save_fig'],
                                                    save_results=utils['confusion_matrix']['params']['save_results'],
                                                    cmap=utils['confusion_matrix']['params']['cmap'], )
        plot_confusion_matrix.get_confusion_matrix(cm_save_jpg_name, cm_save_csv_name)
