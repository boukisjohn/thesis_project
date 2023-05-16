import pandas as pd
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV
from model_optimization_tools import save_metrics, plot_learning_curves, confusion_matrix_heatmap, plot_roc_curves

# Dataset Load
df_train = pd.read_excel('rhode_features_train_090.xlsx')
# df_train = pd.read_excel('rhode_features_train_080.xlsx')
df_train = df_train.sample(frac=1, random_state=13)
x_train = df_train.iloc[:, :-1]  # features
y_train = df_train['label']  # labels

df_valid = pd.read_excel('rhode_features_validation_090.xlsx')
# df_valid = pd.read_excel('rhode_features_validation_080.xlsx')
df_valid = df_valid.sample(frac=1, random_state=13)
x_valid = df_valid.iloc[:, :-1]  # features
y_valid = df_valid['label']  # labels

# kNN GridSearchCV.
"""
grid_knn = {'n_neighbors': [5*i for i in range(1, 5)],
            'weights': ['distance'],
            'algorithm': ['kd_tree'],
            'leaf_size': [40 + 20*j for j in range(3)]}

search_knn = GridSearchCV(KNeighborsClassifier(), param_grid=grid_knn,
                          scoring='matthews_corrcoef', cv=3, refit=True, verbose=2)

search_knn.fit(x_train, y_train)
print(f'kNN Results:\n')
print(f'{search_knn.best_estimator_} \n {search_knn.best_params_}.\n ')
"""

# SVC GridSearchCV.
"""
grid_svc = {'C': [pow(10, i) for i in range(2, 5)],
            'gamma': [pow(0.1, i) for i in range(2, 5)],
            'kernel': ['rbf']}

search_svc = GridSearchCV(SVC(probability=True, random_state=13), param_grid=grid_svc,
                          scoring='matthews_corrcoef', cv=3, refit=True, verbose=2)
search_svc.fit(x_train, y_train)
print(f'SVC Results:\n')
print(f'{search_svc.best_estimator_} \n {search_svc.best_params_}.\n ')
"""

# RFC GridSearchCV.
"""
grid_rfc = {'n_estimators': [100 + 50 * i for i in range(11)],
            'max_features': [20, 40, 60],
            'criterion': ['log_loss', 'gini']}

search_rfc = GridSearchCV(RandomForestClassifier(random_state=13), param_grid=grid_rfc,
                          scoring='matthews_corrcoef', cv=3, refit=True, verbose=2)

search_rfc.fit(x_train, y_train)
print(f'RFC Results:\n')
print(f'{search_rfc.best_estimator_} \n {search_rfc.best_params_}.\n ')
"""

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
estimator = clf1
estimator.fit(x_train, y_train)
y_predict = estimator.predict(x_valid)
"""

classifiers = [clf1, clf2, clf3, clf_soft]
classifier_names = ['knn', 'rfc', 'svc', 'soft vote']
save_metrics(train_x=x_train, train_y=y_train, valid_x=x_valid, valid_y=y_valid,
             clf=classifiers, clf_names=classifier_names, dataset_name='rhode', save=False)

# Confusion Matrix
"""
class_names = ['esophagus', 'stomach', 'small bowel', 'colon']
fig = confusion_matrix_heatmap(test_y=y_valid, predicted_y=y_predict, class_names=class_names,
                               estimator_name=f'{estimator.__class__.__name__}', dataset_name='HyperKvasir')

plt.show()
"""

# Plot learning curves.
"""
learn_curves = plot_learning_curves(features=x_train, labels=y_train, estimator=estimator,
                                    estimator_name=f'{estimator.__class__.__name__}', dataset_name='Rhode Island')

plt.show()
"""
# Plot roc auc.
"""
class_names = ['esophagus', 'stomach', 'small bowel', 'colon']
roc = plot_roc_curves(labels=y_train, train_x=x_train, test_x=x_valid,
                      train_y=y_train, test_y=y_valid, estimator=estimator, class_name=class_names)
plt.show()

"""