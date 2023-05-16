import os
import pandas as pd
import numpy as np
from cv2 import cv2
import math
from scipy import fft
from skimage.feature import local_binary_pattern
from preprocess_tools import ds_store_removal
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA


def rgb_to_hsi(image):
    """
    rgb_to_hsi : It's a function that transforms an image from BGR format to HSI (Hue-Saturation-Intensity) format.
    ------------
    Parameters:
    image: numpy array (MxNx3) (cv2.imread())

    Returns:
    hsi:   numpy array (MxNx3) : The original image transformed into HSI color model.
    ------------
    Original File: converter_rgb2hsi.py
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        # Represent Image in range [0, 1].
        bgr = np.float32(image) / 255

        # Separate color channels.
        blue = bgr[:, :, 0]
        green = bgr[:, :, 1]
        red = bgr[:, :, 2]

        # Hue Calculation.
        def calc_hue(r, b, g):
            hue = np.copy(r)
            # sum = 0
            for i in range(0, b.shape[0]):
                for j in range(0, b.shape[1]):
                    a = 0.5 * ((r[i][j] - g[i][j]) + (r[i][j] - b[i][j]))
                    c = math.sqrt(
                        (r[i][j] - g[i][j]) ** 2 + ((r[i][j] - b[i][j]) * (g[i][j] - b[i][j])))
                    c += 0.001  # To avoid b = 0.
                    hue[i][j] = a / c
                    hue[i][j] = math.acos(hue[i][j])

                    if b[i][j] <= g[i][j]:
                        hue[i][j] = hue[i][j]
                    else:
                        hue[i][j] = ((360 * math.pi) / 180.0) - hue[i][j]
            return hue

        # Saturation Calculation.
        def calc_saturation(r, g, b):
            minimum = np.minimum(np.minimum(r, g), b)
            saturation = 1 - (3 * minimum) / (r + g + b + 0.001)
            return saturation

        # Intensity Calculation.
        def calc_intensity(r, g, b):
            intensity = np.divide(r + g + b, 3)
            return intensity

        hsi = cv2.merge((calc_hue(red, blue, green), calc_saturation(red, blue, green),
                         calc_intensity(red, blue, green)))
        return hsi


def histogram_2d(image, dct=True):
    """
    histogram_2d: 2D Hue-Saturation histogram calculation function.
    ------------
    Parameters:
    image:    numpy array (MxNx3/ RGB)
    dct:      bool (default = True)

    Returns:
    hist_2d_ravel: numpy array (1024x1)
    OR
    hist_2d:  numpy array (32x32)
    If dct = True (default) then the function returns a compressed version of 2D Hue-Saturation histogram in 1D form.
    If dct = False then the function returns a 32-bin 2D HS histogram, ready to plot.
    ------------
    Original File: filter.py
    """
    image = cv2.resize(image, [256, 256])
    hsi = rgb_to_hsi(image)
    hist_2d = cv2.calcHist([hsi], [0, 1], None, [32, 32], [0, 2 * math.pi, 0, 1])
    hist_2d_ravel = np.ravel(hist_2d)
    hist_dct = fft.dct(hist_2d_ravel, n=136)
    if dct:
        return hist_dct
    else:
        return hist_2d


def lbp(image, version):
    """
    lbp : It's a function that extracts the texture histogram of an image based on Local Binary Pattern algorithm.
    ------------
    Parameters:
    image:    numpy array
    version:  int         - =1 for 1d 3-channel lbp (1x7 Array).
                          - =2 for 3d 3-channel lbp (1x343 Array).

    Returns:
    hist_lbp: numpy array - if version = 1: 1x21 Array (1D Array from 3x1D Arrays (separate channels LBP histograms)).
    (7+7+7)
                          - if version = 2: 1x343 Array (1D Array from a 3D LBP Histogram).(7*7*7)
                          ------------
    Original File: local_binary_function.py
    """
    # image was resized from [512, 512]
    image = cv2.resize(image, [256, 256])
    blue, green, red = cv2.split(image)

    radius = 1
    points = 8 * radius

    lbp_blue = local_binary_pattern(blue, P=points, R=radius, method='uniform')
    lbp_green = local_binary_pattern(green, P=points, R=radius, method='uniform')
    lbp_red = local_binary_pattern(red, P=points, R=radius, method='uniform')

    lbp_img = cv2.merge([lbp_blue, lbp_green, lbp_red])

    if version == 1:
        hist_b = cv2.calcHist([lbp_img.astype(np.uint8)], [0], None, [7], [0, 9])
        hist_g = cv2.calcHist([lbp_img.astype(np.uint8)], [1], None, [7], [0, 9])
        hist_r = cv2.calcHist([lbp_img.astype(np.uint8)], [2], None, [7], [0, 9])

        hist_b = np.ravel(hist_b)
        hist_g = np.ravel(hist_g)
        hist_r = np.ravel(hist_r)

        hist_lbp = np.concatenate((hist_b, hist_g, hist_r), axis=0)
        return hist_lbp
    elif version == 2:
        hist_lbp = cv2.calcHist([lbp_img.astype(np.uint8)], [0, 1, 2], None, [7, 7, 7], [0, 9, 0, 9, 0, 9])
        hist_lbp = np.ravel(hist_lbp)
        return hist_lbp
    else:
        print("error")
        pass
    return 0


# Used in both datasets.
def features_extraction(src_path, lbp_version, excel_name, current_work_dir, label=True):
    """
    This is a function that extracts color and texture features of images from a given path. The selected path is the
    one that contains the landmarks and must be sorted in order to label each feature vector correctly.
    :param src_path: the current working directory + where the landmarks are located.
    :param lbp_version: 1 or 2 (1 for 1d histogram kai 2 for 3d histogram).
    :param excel_name: the name of the Excel file that contains the feature array.
    :param current_work_dir: current working directory
    :param label: Bool, if True each vector will have an extra column with its corresponding label.
    :return: the function doesn't return anything other than a saved Excel file.
    """

    landmark_list = os.listdir(os.path.join(current_work_dir, src_path))
    ds_store_removal(kappa=landmark_list)

    features_list = []
    feature_label_list = []
    for i, landmark in enumerate(landmark_list):
        image_list = sorted(os.listdir(os.path.join(src_path, landmark)))
        ds_store_removal(kappa=image_list)
        for image in image_list:
            img = cv2.imread(os.path.join(src_path, landmark, image))
            color_hist = histogram_2d(image=img)
            text_hist = lbp(image=img, version=lbp_version)
            features = np.hstack((color_hist, text_hist))
            features_list.append(features)
            label_value = i + 1
            features_label = np.append(features, label_value)
            feature_label_list.append(features_label)

    features_no_label_array = np.asarray(features_list)
    features_label_array = np.asarray(feature_label_list)
    if label:
        features_array = features_label_array
        dataframe = pd.DataFrame(features_array)
        dataframe.columns = ['feature_' + str(i + 1) for i in range(len(dataframe.columns))]
        dataframe.columns = [*dataframe.columns[:-1], 'label']
    else:
        features_array = features_no_label_array
        dataframe = pd.DataFrame(features_array)
        dataframe.columns = ['feature_' + str(i + 1) for i in range(len(dataframe.columns))]

    dataframe.to_excel(excel_name + '.xlsx', index=False)
    print(dataframe.head())

    return None


# Used in HyperKvasir.
def features_extraction_one_folder(src_path, lbp_version, excel_name, current_work_dir):
    """
    This is a function that extracts color and texture features of images from a given path. The selected path is the
    one that contains the landmarks and must be sorted in order to label each feature vector correctly.
    :param src_path: the current working directory + where the landmarks are located.
    :param lbp_version: 1 or 2 (1 for 1d histogram kai 2 for 3d histogram).
    :param excel_name: the name of the Excel file that contains the feature array.
    :param current_work_dir: current working directory
    :return: the function doesn't return anything other than a saved Excel file.
    """
    image_list = os.listdir(os.path.join(current_work_dir, src_path))
    ds_store_removal(kappa=image_list)
    image_list = sorted(sorted(image_list), key=len)

    features_list = []
    for image in image_list:
        img = cv2.imread(os.path.join(src_path, image))
        color_hist = histogram_2d(image=img)
        text_hist = lbp(image=img, version=lbp_version)
        features = np.hstack((color_hist, text_hist))
        features_list.append(features)

    features_no_label_array = np.asarray(features_list)
    features_array = features_no_label_array
    dataframe = pd.DataFrame(features_array)
    dataframe.columns = ['feature_' + str(i + 1) for i in range(len(dataframe.columns))]

    dataframe.to_excel(excel_name + '.xlsx', index=False)
    print(dataframe.head())

    return None


# Used in both datasets.
def pca_scree_plot(scaled_array, feature_name):
    """
    A function that creates the scatter plot.
    :param scaled_array: feature array but scaled
    :param feature_name: color or texture
    :return: None
    """
    pca = PCA(n_components=len(scaled_array[0]))
    pca.fit(scaled_array)
    plt.grid()
    plt.plot(np.cumsum(pca.explained_variance_ratio_ * 100))
    plt.xlabel('Number of Components')
    plt.ylabel('Explained Variance')
    plt.title(f'{feature_name}')
    plt.show()
    return None

