import os
import pandas as pd
import numpy as np
from rhode_preprocess_tools import ds_store_removal
from feature_extraction_tools import lbp, histogram_2d, pca_scree_plot
from cv2 import cv2
import time
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import matthews_corrcoef, log_loss, accuracy_score
from my_functions import plot_learning_curves
from matplotlib import pyplot as plt
from feature_extraction_tools import features_extraction
from feature_extraction_tools import features_extraction_one_folder
from preprocess_tools import ds_store_removal

cwd = os.getcwd()

# Train/ Valid feature extraction.
"""
start = time.time()

source_path = os.path.join(cwd, 'HyperKvasir Dataset')
features_extraction(src_path=source_path, lbp_version=2, excel_name='hyperkvasir_features',
                    current_work_dir=cwd, label=True)

"""

# Test feature extraction (extract features from folders with video frames).
"""
test_folder = 'HyperKvasir Test'
cases_list = sorted(os.listdir(os.path.join(cwd, test_folder)))
ds_store_removal(kappa=cases_list)
print(f'{cases_list}')

for case in cases_list:
    sourcePath = os.path.join(cwd, test_folder, case)
    print(f'Source Path: {sourcePath}')
    features_extraction_one_folder(src_path=sourcePath, lbp_version=2,
                                   excel_name=f'{case}_features', current_work_dir=cwd)

end = time.time()
print('time: ', (end-start)/3600, 'hours.')
"""

# PCA

df = pd.read_excel('hyperkvasir_features.xlsx')
df = df.sample(frac=1, random_state=13)
x = df.iloc[:, :-1]  # features
y = df['label']  # labels

x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, random_state=13)

# -------------------------------------------------
# Training set.
# -------------------------------------------------

df_train = x_train
df_train_color = df_train.iloc[:, :136]
df_train_texture = df_train.iloc[:, 136:-1]
y_train = np.reshape(y_train.to_numpy(), (-1, 1))

# Color
color_scaler = StandardScaler()
x_train_color = df_train_color
x_train_scaled_color = color_scaler.fit_transform(x_train_color)
number_of_components_color = .90  # keep 90% of information
pca_color = PCA(n_components=number_of_components_color)
pca_train_features_color = pca_color.fit_transform(x_train_scaled_color)
print(f'{pca_train_features_color.shape}')

# Texture
texture_scaler = StandardScaler()
x_train_texture = df_train_texture
x_train_scaled_texture = texture_scaler.fit_transform(x_train_texture)
number_of_components_texture = .90  # keep 90% of information
pca_texture = PCA(n_components=number_of_components_texture)
pca_train_features_texture = pca_texture.fit_transform(x_train_scaled_texture)
print(f'{pca_train_features_texture.shape}')

# Combine
train_features = np.concatenate((pca_train_features_color, pca_train_features_texture), axis=1)
new_train = np.concatenate((train_features, y_train), axis=1)
new_train = pd.DataFrame(data=new_train, columns=['pc_%d' % i for i in range(1, len(new_train[0]) + 1)])
new_train.columns = [*new_train.columns[:-1], 'label']
print(f'Train PCA:\n {new_train.head()}')
# new_train.to_excel('hyperkvasir_features_train_090.xlsx', index=False)

pca_scree_plot(scaled_array=x_train_scaled_color, feature_name='Color Feature')
pca_scree_plot(scaled_array=x_train_scaled_texture, feature_name='Texture Feature')

"""
# -------------------------------------------------
# Validation set.
# -------------------------------------------------

df_valid = x_valid
df_valid_color = df_valid.iloc[:, :136]
df_valid_texture = df_valid.iloc[:, 136:-1]
y_valid = np.reshape(y_valid.to_numpy(), (-1, 1))

# Color
x_validation_color = df_valid_color
x_validation_scaled_color = color_scaler.transform(x_validation_color)
pca_validation_features_color = pca_color.transform(x_validation_scaled_color)

# Texture
x_validation_texture = df_valid_texture
x_validation_scaled_texture = texture_scaler.transform(x_validation_texture)
pca_validation_features_texture = pca_texture.transform(x_validation_scaled_texture)

# Combine
validation_features = np.concatenate((pca_validation_features_color, pca_validation_features_texture), axis=1)
new_validation = np.concatenate((validation_features, y_valid), axis=1)
new_validation = pd.DataFrame(data=new_validation, columns=['pc_%d' % i for i in range(1, len(new_validation[0]) + 1)])
new_validation.columns = [*new_validation.columns[:-1], 'label']
print(f'Validation PCA:\n {new_validation.head()}')
# new_validation.to_excel('hyperkvasir_features_validation_090.xlsx', index=False)

"""

"""
# -------------------------------------------------
# Test set.
# -------------------------------------------------
'''
sourcePath = os.path.join(cwd, 'HyperKvasir Test Excel')
test_df_list = sorted(os.listdir(sourcePath))
ds_store_removal(kappa=test_df_list)

for test_df in test_df_list:
    print(test_df)
    df = pd.read_excel(os.path.join(sourcePath, test_df))
    df_test_color = df.iloc[:, :136]
    df_test_texture = df.iloc[:, 136:-1]

    # Color
    x_test_color = df_test_color
    x_test_scaled_color = color_scaler.transform(x_test_color)
    pca_test_features_color = pca_color.transform(x_test_scaled_color)

    # Texture
    x_test_texture = df_test_texture
    x_test_scaled_texture = texture_scaler.transform(x_test_texture)
    pca_test_features_texture = pca_texture.transform(x_test_scaled_texture)

    # Combine
    new_test = np.concatenate((pca_test_features_color, pca_test_features_texture), axis=1)
    new_test = pd.DataFrame(data=new_test, columns=['pc_%d' % i for i in range(1, len(new_test[0]) + 1)])
    print(f'Test PCA:\n {new_test.head()}')
    print(f'{test_df[:-5]}_90.xlsx')
    new_test.to_excel(f'{test_df[:-5]}_90.xlsx', index=False)
'''
"""

# Mini Validation
"""
# Validate
df = pd.read_excel('hyperkvasir_features_090.xlsx')
df = df.sample(frac=1, random_state=13)

x = df.iloc[:, :-1]  # features
y = df['label']  # labels

x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, random_state=13)

clf1 = KNeighborsClassifier(n_neighbors=7, algorithm='ball_tree', weights='distance')
clf2 = RandomForestClassifier(n_estimators=550, max_features=20, criterion='gini', random_state=13)

clf3 = SVC(C=100,
           class_weight='balanced',
           gamma=0.0001,
           kernel='rbf',
           probability=True,
           random_state=13)

clf = VotingClassifier(estimators=[('knn', clf1), ('rfc', clf2), ('svc', clf3)],
                       voting='soft', weights=[0.1, 0.2, 0.4])

clf.fit(x_train, y_train)
y_predict = clf.predict(x_valid)
y_proba = clf.predict_proba(x_valid)
print(f'accuracy score:{accuracy_score(y_true=y_valid, y_pred=y_predict)}.')
print(f'mcc:{matthews_corrcoef(y_true=y_valid, y_pred=y_predict)}.')
a = plot_learning_curves(features=x_train_valid, labels=y_train_valid, estimator=clf,
                         estimator_name='rfc', dataset_name='hyperKvasir')

plt.show()
"""


