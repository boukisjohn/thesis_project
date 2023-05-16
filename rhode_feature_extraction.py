import os
import pandas as pd
from cv2 import cv2
import numpy as np
import time
from rhode_preprocess_tools import ds_store_removal
from rhode_feature_extraction_tools import histogram_2d, lbp
from feature_extraction_tools import features_extraction, pca_scree_plot
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


cwd = os.getcwd()
# Train, Validation and Test set feature extraction.
"""
start = time.time()

source_path_train_ds = os.path.join(cwd, 'rhode dataset split/train_ds')
features_extraction(src_path=source_path_train_ds, lbp_version=2, excel_name='rhode_features_train_ds',
                    current_work_dir=cwd, label=True)

source_path_valid_ds = os.path.join(cwd, 'rhode dataset split/validation_ds')
features_extraction(src_path=source_path_valid_ds, lbp_version=2, excel_name='rhode_features_validation_ds',
                    current_work_dir=cwd,  label=True)

source_path_test_ds = os.path.join(cwd, 'rhode dataset split/test_ds')
features_extraction(src_path=source_path_test_ds, lbp_version=2, excel_name='rhode_features_test_ds',
                    current_work_dir=cwd,  label=True)
                    
end = time.time()
print('time: ', (end-start)/3600, 'hours.')
"""

# PCA

# -------------------------------------------------
# Training set.
# -------------------------------------------------

df_train = pd.read_excel('rhode_features_train_ds.xlsx')
df_train = df_train.sample(frac=1, random_state=13)
df_train_color = df_train.iloc[:, :136]
df_train_texture = df_train.iloc[:, 136:-1]
y_train = np.reshape(df_train['label'].to_numpy(), (-1, 1))

# Color
color_scaler = StandardScaler()
x_train_color = df_train_color
x_train_scaled_color = color_scaler.fit_transform(x_train_color)
number_of_components_color = .90  # keep 90% of information
pca_color = PCA(n_components=number_of_components_color)
pca_train_features_color = pca_color.fit_transform(x_train_scaled_color)
print(f'color pca shape:{pca_train_features_color.shape}.')

# Texture
texture_scaler = StandardScaler()
x_train_texture = df_train_texture
x_train_scaled_texture = texture_scaler.fit_transform(x_train_texture)
number_of_components_texture = .90  # keep 90% of information
pca_texture = PCA(n_components=number_of_components_texture)
pca_train_features_texture = pca_texture.fit_transform(x_train_scaled_texture)
print(f'texture pca shape:{pca_train_features_texture.shape}.')

pca_scree_plot(scaled_array=x_train_scaled_color, feature_name='Color Feature')
pca_scree_plot(scaled_array=x_train_scaled_texture, feature_name='Texture Feature')

# Combine
train_features = np.concatenate((pca_train_features_color, pca_train_features_texture), axis=1)
new_train = np.concatenate((train_features, y_train), axis=1)
new_train = pd.DataFrame(data=new_train, columns=['pc_%d' % i for i in range(1, len(new_train[0]) + 1)])
new_train.columns = [*new_train.columns[:-1], 'label']
print(new_train.head())
# new_train.to_excel('rhode_features_train_090.xlsx', index=False)

'''
"""
# -------------------------------------------------
# Validation set.
# -------------------------------------------------

df_valid = pd.read_excel('rhode_features_validation_ds.xlsx')
df_valid = df_valid.sample(frac=1, random_state=13)
df_valid_color = df_valid.iloc[:, :136]
df_valid_texture = df_valid.iloc[:, 136:-1]
y_valid = np.reshape(df_valid['label'].to_numpy(), (-1, 1))

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
print(new_validation.head())
new_validation.to_excel('rhode_features_validation_090.xlsx', index=False)


# -------------------------------------------------
# Test set.
# -------------------------------------------------

df_test = pd.read_excel('rhode_features_test_ds.xlsx')
df_test = df_test.sample(frac=1, random_state=13)
df_test_color = df_test.iloc[:, :136]
df_test_texture = df_test.iloc[:, 136:-1]
y_test = np.reshape(df_test['label'].to_numpy(), (-1, 1))

# Color
x_test_color = df_test_color
x_test_scaled_color = color_scaler.transform(x_test_color)
pca_test_features_color = pca_color.transform(x_test_scaled_color)

# Texture
x_test_texture = df_test_texture
x_test_scaled_texture = texture_scaler.transform(x_test_texture)
pca_test_features_texture = pca_texture.transform(x_test_scaled_texture)

# Combine
test_features = np.concatenate((pca_test_features_color, pca_test_features_texture), axis=1)
new_test = np.concatenate((test_features, y_test), axis=1)
new_test = pd.DataFrame(data=new_test, columns=['pc_%d' % i for i in range(1, len(new_test[0]) + 1)])
new_test.columns = [*new_test.columns[:-1], 'label']
print(new_test.head())
new_test.to_excel('rhode_test_features_090.xlsx', index=False)
"""
'''