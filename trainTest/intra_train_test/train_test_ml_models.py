from utils.common_utils import make_dir
from trainTest.metrics.get_test_metrics import GetTestResults, PlotConfusionMatrix
from utils.common_params import *


def train_test_ml_models(settings, model, x_train, y_train, x_test, y_test):
    # 1. Model, test results save absolute path and filename
    utils = ml_train_test_utils
    basic_path = settings['save_path']['absolute_path']
    make_dir(basic_path)
    model_save_name = ''.join([basic_path, '/model_', str(settings['current_exp_time']), '.pkl'])

    # 2. Obtaining the model
    print('model: %s ' % model.get_model_name())
    model.init()
    if utils['parameter_optimization']:
        # 3. Parameter optimization by GridSearchCV
        print('Parameter optimization by GridSearchCV: ')
        best_params = model.parameter_optimization(x_train, y_train)
        print('Optimized parameters: ', best_params)
        # 4. Set parameters
        print('Set the optimal parameters: ')
        model.set_params(best_params)
    else:
        print('Use the default parameters: ')

    # 5. Model training
    print('Model training: ')
    model.train(x_train, y_train)
    # 6. Model test
    print('Model test: ')
    pre_y_test = model.predict(x_test)
    # 7. Saving model
    if utils['save_model']:
        print('Saving model: ')
        model.save(model_save_name)

    # 8. Save test results
    test_results_utils = GetTestResults(utils['test_metrics'], y_test, pre_y_test, decimal=5)
    test_metrics = test_results_utils.calculate()
    test_metrics_save_name = ''.join([basic_path, '/test_metrics.csv'])
    pre_results_save_name = ''.join([basic_path, '/predicted_results_', str(settings['current_exp_time']), '.csv'])
    test_results_utils.save(settings, test_metrics, test_metrics_save_name, pre_results_save_name)
    test_metrics_dict = dict(zip(utils['test_metrics'], test_metrics))
    print('Test results: ')
    for key, value in test_metrics_dict.items():
        print(f"{key}:  {value}.")

    # 9. Calculate confusion matrix
    if utils['confusion_matrix']['get_cm']:
        print('Calculate confusion matrix: ')
        cm_save_jpg_name = ''.join([basic_path, '/confusion_matrix_', str(settings['current_exp_time']), '.jpg'])
        cm_save_csv_name = ''.join([basic_path, '/confusion_matrix_', str(settings['current_exp_time']), '.xlsx'])
        plot_confusion_matrix = PlotConfusionMatrix(y_test, pre_y_test,
                                                    label_type=None,
                                                    show_type=utils['confusion_matrix']['params']['show_type'],
                                                    plot=utils['confusion_matrix']['params']['plot'],
                                                    save_fig=utils['confusion_matrix']['params']['save_fig'],
                                                    save_results=utils['confusion_matrix']['params']['save_results'],
                                                    cmap=utils['confusion_matrix']['params']['cmap'], )
        plot_confusion_matrix.get_confusion_matrix(cm_save_jpg_name, cm_save_csv_name)
