import pandas as pd
import numpy as np
import time
import math
from matplotlib import pyplot as plt

import seaborn as sns
from sklearn.metrics import matthews_corrcoef, f1_score, accuracy_score, roc_auc_score, confusion_matrix
from sklearn.metrics import log_loss, make_scorer, roc_curve, auc

from sklearn.utils.multiclass import unique_labels

from sklearn.model_selection import learning_curve
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier


def save_metrics(train_x, train_y, valid_x, valid_y, clf, clf_names, dataset_name, show=True, save=True):
    """
    A function that tests the performance of a bunch of classifiers for a number of metrics.
    1) Matthews Correlation Coefficient
    2) Accuracy Score
    3) F1-Score (macro average)
    4) Area Under the Curve (macro average, ovr scenario)
    5) Time in seconds to fit each model to the training set.
    :param train_x: x_train from split
    :param train_y: y_train from split
    :param valid_x: x_validation from split
    :param valid_y: y_validation from split
    :param clf: List that contains classifiers (the classifiers should be hyperparameter tuned)
    :param clf_names: List with the names of the classifiers
    :param dataset_name: String, the name of the working dataset
    :param show: Bool, true for head plot (default : True)
    :param save: Bool, true for save dataframe (default : True)
    :return: None
    """
    mcc = []
    acc = []
    f1 = []
    auc_roc = []
    t = []

    for i in range(len(clf)):
        start = time.time()
        estimator = clf[i]
        estimator.fit(train_x, train_y)
        y_predict = estimator.predict(valid_x)
        y_proba = estimator.predict_proba(valid_x)
        end = time.time()

        m = matthews_corrcoef(y_true=valid_y, y_pred=y_predict)
        a = accuracy_score(y_true=valid_y, y_pred=y_predict)
        f = f1_score(y_true=valid_y, y_pred=y_predict, average='macro')
        au = roc_auc_score(y_true=valid_y, y_score=y_proba, average='macro', multi_class='ovr')
        times = end - start

        # Print metrics.
        # print(f'Classifier: {clf_names[i]}')
        # print(f'MCC: {m}.')
        # print(f'ACC: {a}.')
        # print(f'F1: {f}.')
        # print(f'AUC: {au}.')
        # print(f'Time: {times}.')
        # print('\n')

        mcc.append(m)
        acc.append(a)
        f1.append(f)
        auc_roc.append(au)
        t.append(times)

    mcc = np.array(mcc)
    acc = np.array(acc)
    f1 = np.array(f1)
    auc_roc = np.array(auc_roc)
    t = np.array(t)

    metrics = np.vstack((mcc, acc, f1, auc_roc, t))
    column = clf_names
    index = ['MCC', 'ACC', 'F1', 'AUC', 'Time']
    df_metrics = pd.DataFrame(data=metrics, index=index, columns=column)
    if save:
        df_metrics.to_excel(f'{dataset_name}_metrics.xlsx')
    if show:
        print(df_metrics.head())

    return None


# class_name = ['background', 'z-line', 'retroflex/ stomach', 'pylorus', 'cecum', 'retroflex/ rectum']
# class_name = ['esophagus', 'stomach', 'small_bowel', 'colon']
def confusion_matrix_heatmap(test_y, predicted_y, class_names, estimator_name, dataset_name, normalized=True):
    """
    A function that creates confusion matrix and plots it.
    :param test_y: pandas series, the y_test or valid.
    :param predicted_y: numpy array, the predicted y.
    :param class_names: List, with the labels names.
    :param estimator_name: String, name of the classifier.
    :param dataset_name: String, name of the dataset.
    :param normalized: Bool, if True it creates a normalized version. default=True
    :return: A figure of a heatmap plot of the confusion matrix.
    """

    labels = unique_labels(test_y)
    columns = [f'{class_names[i-1]}' for i in labels]  # Predicted
    index = [f'{class_names[i-1]}' for i in labels]  # Actual
    figure = plt.figure(figsize=(7, 7))
    if normalized:
        table = pd.DataFrame(confusion_matrix(test_y, predicted_y, normalize='true'), columns=columns, index=index)
        sns.heatmap(table, annot=True, fmt='0.2f', cmap='viridis', cbar=False)
        norm_name = 'Normalized'
    else:
        table = pd.DataFrame(confusion_matrix(test_y, predicted_y), columns=columns, index=index)
        sns.heatmap(table, annot=True, fmt='d', cmap='viridis')
        norm_name = ''

    plt.suptitle(f'Confusion Matrix {norm_name}', fontsize=20)
    plt.title(f'estimator: {estimator_name} dataset: {dataset_name}', style='italic')
    plt.xlabel('Predicted Labels', fontsize=10, fontweight='bold', labelpad=4)
    plt.ylabel('Ground Truth Labels', fontsize=10, fontweight='bold')
    #     plt.show()
    return figure


def plot_roc_curves(labels, train_x, test_x, train_y, test_y, class_name, estimator):
    """
    A function that creates and plots Receiver Operator Characteristic Curves and calculates Area Under the Curve in
    One vs Rest scenario.
    :param class_name: list with classes names like: ['background', 'z-line', 'retroflex/ stomach']
    :param labels: any y vector
    :param train_x: x_train
    :param test_x: y_train
    :param train_y: x_validation
    :param test_y: y_validation
    :param estimator: classifier should be NOT fitted!
    :return: figures
    """

    # Prepare labels for ROC curves for One-vs-Rest scenario.
    unique_classes = np.unique(labels)
    y_binarize = label_binarize(labels, classes=unique_classes)
    # Binarize train and test labels.
    train_y = label_binarize(train_y, classes=unique_classes)
    test_y = label_binarize(test_y, classes=unique_classes)
    n_classes = y_binarize.shape[1]

    # classifier fitting and predicting y test probabilities values.
    classifier = OneVsRestClassifier(estimator)
    classifier.fit(train_x, train_y)
    # y_predicted = classifier.predict(x_test)
    y_probabilities = classifier.predict_proba(test_x)

    # ROC curve and AUC calculation.
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(test_y[:, i].ravel(), y_probabilities[:, i].ravel())
        roc_auc[i] = auc(fpr[i], tpr[i])

    colors = ['tomato', 'mediumslateblue', 'yellowgreen', 'mediumaquamarine', 'darkorange', 'hotpink']
    # class_name = ['background', 'z-line', 'retroflex/ stomach', 'pylorus', 'cecum', 'retroflex/ rectum']
    plt.style.use('seaborn')
    fig, axes = plt.subplots(nrows=1, ncols=n_classes)
    fig.suptitle('Receiver Operating Characteristic Curves', y=0.8, fontsize=14, fontweight='bold')

    for i, ax in enumerate(axes.ravel()):
        # ax.plot(fpr[i], tpr[i], color=colors[i], label='Area Under the Curve %0.2f' % roc_auc[i])
        ax.plot(fpr[i], tpr[i], color=colors[i])
        ax.plot([], [], ' ', label=f'auc {roc_auc[i]:.4f}')  # Na to deis edo.
        ax.plot([0, 1], [0, 1], 'k--')
        ax.legend(loc='lower right')
        ax.set_title(f'{class_name[unique_classes[i]-1]} vs rest')
        ax.set_xlim([-0.05, 1.05])
        ax.set_ylim([-0.05, 1.05])
        ax.set_aspect('equal', adjustable='box')  # Gia na vgainoun ta plots se morfi tetragwnou.
        ax.set(xlabel='False Positive Rate', ylabel='True Positive Rate')
        ax.label_outer()  # Krivei ta y-labels sta endiamesa plots.

    # plt.show()
    return fig


def plot_learning_curves(features, labels, estimator, estimator_name, dataset_name,
                         train_split=0.8, n_training_samples=6):
    """
    plot_learning_curves: Plotting a learning curve based on a metric (currently Cross-Entropy Loss).
    ------------
    Parameters:
    dataframe: pandas dataframe
    estimator: sklearn estimator/ classifier
    train_split: int (0,1) default=0.8 (80% train and 20% validate)
    n_training_samples: int default=6

    Returns:
    A plot of the training and validation against samples.
    ------------
    Original File: og_code.py
    """
    # Data splitting in training and validation set
    n_samples = len(np.asarray(features))
    n_samples_train = math.floor(train_split * n_samples)
    # n_samples_validation = math.ceil((1.0 - train_split) * n_samples)

    if n_samples <= 3000:
        starting_point = 50
    elif 3000 < n_samples <= 5000:
        starting_point = 150
    else:
        starting_point = 300

    training_samples = np.linspace(starting_point, n_samples_train, n_training_samples, dtype=int)

    # Scorers.
    log_loss_scorer = make_scorer(log_loss, greater_is_better=False, needs_proba=True)
    # mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)

    # Calculating learning curves.
    train_sizes, train_scores, validation_scores = learning_curve(
        estimator=estimator, X=features, y=labels,
        train_sizes=training_samples, cv=5, scoring=log_loss_scorer)

    train_scores_mean = -train_scores.mean(axis=1)
    validation_scores_mean = -validation_scores.mean(axis=1)
    gap = validation_scores_mean[-1] - train_scores_mean[-1]
    figure = plt.figure()
    plt.style.use('seaborn')
    plt.plot(train_sizes, train_scores_mean, label='Training error')
    plt.plot(train_sizes, validation_scores_mean, label='Validation error')
    plt.plot([], [], ' ', label='Minimum difference %0.2f' % gap)
    plt.ylabel('Cross-Entropy Loss', fontsize=14)
    plt.xlabel('Training set size', fontsize=14)
    plt.suptitle('Learning Curves', fontsize=20)
    plt.title(f'estimator:{estimator_name} dataset:{dataset_name}', style='italic')
    plt.legend()
    # plt.ylim(min(min(train_scores_mean), min(validation_scores_mean))-0.2,
    #          max(max(train_scores_mean), max(validation_scores_mean))+0.2)
    # plt.ylim(-0.1, 0.8)
    # plt.show()
    return figure

