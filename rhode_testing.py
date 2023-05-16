import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier

from model_optimization_tools import save_metrics, confusion_matrix_heatmap, plot_roc_curves
from sklearn.metrics import matthews_corrcoef, f1_score, accuracy_score, roc_auc_score


df_train = pd.read_excel('rhode_features_train_090.xlsx')
df_train = df_train.sample(frac=1, random_state=13)
# x_train = df_train.iloc[:, :-1]  # features
# y_train = df_train['label']  # labels

df_valid = pd.read_excel('rhode_features_validation_090.xlsx')
df_valid = df_valid.sample(frac=1, random_state=13)
# x_valid = df_valid.iloc[:, :-1]  # features
# y_valid = df_valid['label']  # labels

df_train_valid = df_train.append(df_valid, ignore_index=True)
x_train_valid = df_train_valid.iloc[:, :-1]  # features
y_train_valid = df_train_valid['label']  # labels
print(f'train valid shape:{df_train_valid.shape}.')

df_test = pd.read_excel('rhode_features_test_090.xlsx')
df_test = df_test.sample(frac=1, random_state=13)
print(f'test shape:{df_test.shape}.')

x_test = df_test.iloc[:, :-1]  # features
y_test = df_test['label']  # labels
class_name = ['esophagus', 'stomach', 'small_bowel', 'colon']

# KNN Classifier
clf1 = KNeighborsClassifier(n_neighbors=7, algorithm='kd_tree', weights='distance', leaf_size=40)
# RFC Classifier
clf2 = RandomForestClassifier(n_estimators=350, max_features=20, criterion='log_loss', random_state=13)
# SVC Classifier
clf3 = SVC(C=100,
           class_weight='balanced',
           gamma=0.001,
           kernel='rbf',
           probability=True,
           random_state=13)
# SoftVote Classifier
clf_soft = VotingClassifier(estimators=[('svc', clf1), ('rfc', clf2), ('knn', clf3)],
                            voting='soft', weights=[0.1, 0.2, 0.3])

"""
estimator = clf3
roc = plot_roc_curves(labels=y_train_valid, train_x=x_train_valid, test_x=x_test,
                      train_y=y_train_valid, test_y=y_test, class_name=class_name, estimator=estimator)

"""

estimator = clf3
estimator.fit(x_train_valid, y_train_valid)
y_predict = estimator.predict(x_test)
# y_proba = estimator.predict_proba(x_test)
conf = confusion_matrix_heatmap(test_y=y_test, predicted_y=y_predict, class_names=class_name,
                                estimator_name=estimator.__class__.__name__, dataset_name='Rhode Island')

plt.show()


print(f'MCC:{matthews_corrcoef(y_true=y_test, y_pred=y_predict):.4f}')
print(f'ACC:{accuracy_score(y_true=y_test, y_pred=y_predict):.4f}')
# auc = roc_auc_score(y_true=y_test, y_score=y_proba, average='macro', multi_class='ovr')
# print(f'AUC:{auc:.4f}')
f1 = f1_score(y_true=y_test, y_pred=y_predict, average='macro')
print(f'F1:{f1:.4f}')
