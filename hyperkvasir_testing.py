import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from preprocess_tools import ds_store_removal
from model_optimization_tools import confusion_matrix_heatmap

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import matthews_corrcoef, accuracy_score, f1_score, roc_auc_score


cwd = os.getcwd()

df_train = pd.read_excel('hyperkvasir_features_train_090.xlsx')
df_train = df_train.sample(frac=1, random_state=13)

df_valid = pd.read_excel('hyperkvasir_features_validation_090.xlsx')
df_valid = df_valid.sample(frac=1, random_state=13)

df_train_valid = df_train.append(df_valid, ignore_index=True)
x_train_valid = df_train_valid.iloc[:, :-1]  # features
y_train_valid = df_train_valid['label']  # labels

test_path = 'HyperKvasir Test with PCA'
test_list = sorted(os.listdir(os.path.join(cwd, f'{test_path}')))
ds_store_removal(kappa=test_list)
# print(f'test list: {test_list}')
test_names = ['Z-Line (i)', 'Z-Line (ii)', 'Z-Line (iii)',
              'Retroflex - Stomach (i)', 'Retroflex - Stomach (ii)',
              'Pylorus (i)', 'Pylorus (ii)', 'Pylorus (iii)', 'Pylorus (iv)',
              'Cecum (i)', 'Cecum (ii)', 'Cecum (iii)']

class_names = ['background', 'z-line', 'retr/stomach', 'pylorus', 'cecum', 'retr/rectum']

# KNN Classifier
clf1 = KNeighborsClassifier(n_neighbors=3, algorithm='kd_tree', weights='distance', leaf_size=40)
# RFC Classifier
clf2 = RandomForestClassifier(n_estimators=350, max_features=20, criterion='gini', random_state=13)
# SVC Classifier
clf3 = SVC(C=1000,
           class_weight='balanced',
           gamma=0.0001,
           kernel='rbf',
           probability=True,
           random_state=13)

# Voting Classifier.
clf_soft = VotingClassifier(estimators=[('knn', clf1), ('rfc', clf2), ('svc', clf3)],
                            voting='soft', weights=[0.1, 0.2, 0.3])

estimator = clf3
estimator.fit(x_train_valid, y_train_valid)

for i, file in enumerate(test_list):
    # print(f'file: {file}, i:{i}')
    dataframe = pd.read_excel(os.path.join(cwd, f'{test_path}', file))
    dataframe['label'] = int(file[0])*np.ones(shape=len(dataframe))
    # dataframe.to_excel(f'{file[:-5]}_with_label.xlsx')
    x_test = dataframe.iloc[:, :-1]  # features
    y_test = dataframe['label']  # labels
    y_predicted = estimator.predict(x_test)
    y_proba = estimator.predict_proba(x_test)

    m = matthews_corrcoef(y_true=y_test, y_pred=y_predicted)
    a = accuracy_score(y_true=y_test, y_pred=y_predicted)
    f = f1_score(y_true=y_test, y_pred=y_predicted, average='macro')
    print(f'file: {file}')
    print(f'mcc: {m:.4f}')
    print(f'accuracy: {a:.4f}')
    print(f'f1_score: {f:.4f}')

print('So far so Good.')


"""
for i, test_file in enumerate(test_list):
    print(f'{test_names[i]}, {test_file}')
    df_test = pd.read_excel(os.path.join(cwd, f'{test_path}', test_file))
    x_test = df_test
    test_name = [f'{test_names[i]}', f'{test_file[0]}']

    figure = plt.figure()
    y_predicted = estimator.predict(x_test)
    plt.style.use('seaborn')
    plt.xlabel('Number of Frames', fontsize=18, labelpad=6)
    plt.yticks(ticks=[1, 2, 3, 4, 5, 6], labels=labels)
    plt.ylabel('Labels', fontsize=18, labelpad=6)
    plt.suptitle(f'Video: {test_name[0]}  Classifier: {estimator.__class__.__name__}', fontsize=25)
    ax = figure.gca()
    ax.set_ylim(0.5, 6.5)
    plt.scatter(np.arange(len(y_predicted)), y_predicted, s=0.9)
    plt.show()
"""
