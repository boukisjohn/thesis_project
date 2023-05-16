import os
import pandas as pd
import numpy as np
from cv2 import cv2
import math
import shutil
import random


# Used in Rhode Island Dataset.
def ds_store_removal(kappa):
    """
    a function that removes metadata files from a list (typically an os.listdir() list)
    :param kappa: a random list
    :return: a list without extra files
    """
    if '.DS_Store' in kappa:
        kappa.remove('.DS_Store')
    if 'new_.DS_Store' in kappa:
        kappa.remove('new_.DS_Store')


# Used in HyperKvasir Dataset.
def video_to_frames(current_work_dir, source_path, images_name, folder_path):
    """
    A function that creates images from video frames.
    :param current_work_dir: os.getcwd()
    :param source_path: where the video is stored ('/video_test_frames/z_line_video_test/z_line_video_test_2.avi')
    :param images_name: how the new names will be called (z_line_1_-i-.jpg)
    :param folder_path: the path of the folder to save the generated images.
    :return:
    """
    video_path = os.path.join(current_work_dir, source_path)
    video_cap = cv2.VideoCapture(video_path)
    success, image = video_cap.read()
    count = 0
    while success:
        cv2.imwrite(os.path.join(folder_path, f'{images_name}_{count}.jpg'), image)  # save frame as JPEG file
        success, image = video_cap.read()
        print('Read a new frame: ', success)
        count += 1
    return None


# Used in Rhode Island Dataset.
def numbers2excel(list_to_convert, excel_name, label, return_list=False):
    """
    a function that returns the corresponding case values from a given list
    :param list_to_convert: a list with integers
    :param excel_name: string, how the xlsx file will be called
    :param label: string, label for the xlsx file (train, valid, test)
    :param return_list: True for list return
    :return:
    """
    name_list = []
    for i in list_to_convert:
        if len(str(i)) == 1:
            a = 's00' + str(i)
        elif len(str(i)) == 2:
            a = 's0' + str(i)
        elif len(str(i)) == 3:
            a = 's' + str(i)
        else:
            print('something went wrong!')
            break
        name_list.append(a)
    dataframe = pd.DataFrame(name_list, columns=[label])
    dataframe.to_excel(f'{excel_name}.xlsx', index=False)
    if return_list:
        return name_list
    else:
        return None


# Used in Rhode Island Dataset.
def data_split(source_path, current_working_directory):
    """
    A function that produces train, validation and test set based on the No of different patients/cases.
    :param source_path: where the patients/cases are stored
    :param current_working_directory: os.getcwd()
    :return: three lists, one of
    """
    list_of_files = os.listdir(os.path.join(current_working_directory, source_path))
    ds_store_removal(kappa=list_of_files)
    number_of_files = len(list_of_files)
    list_of_numbers = list(range(1, number_of_files+1))
    random.seed(13)
    random.shuffle(list_of_numbers)
    print(list_of_numbers, '\n')

    train_valid_value = math.floor(0.8 * len(list_of_numbers))

    train_value = math.floor(0.8 * train_valid_value)
    valid_value = math.ceil(0.2 * train_valid_value)
    test_value = math.ceil(0.2 * len(list_of_numbers))

    if train_value + valid_value + test_value == len(list_of_numbers):
        print(f'train set len: {train_value}, validation set len:{valid_value},  test set len: {test_value}.')
        print('splitting was successful!')
        pass
    else:
        print('there was a problem!')
        pass

    train_list = list_of_numbers[:train_value]
    valid_list = list_of_numbers[train_value:train_value + valid_value]
    test_list = list_of_numbers[train_value + valid_value:train_value + valid_value + test_value]

    print(f'train set: {train_list}, validation list: {valid_list}, and test set: {test_list}.')
    return train_list, valid_list, test_list


# Used in Rhode Island Dataset.
def copy_to_folder(destination_folder, list_to_copy, current_working_directory, source_path, destination_path):
    """
    A function that copies/ moves images from specific files based on a list
    to the necessary folder (train, validation, test) and renames them.
    :param destination_folder: string train, validation ot test
    :param list_to_copy: a list with elements like s002 extracted from excel
    :param current_working_directory: os.getcwd()
    :param source_path: the source path
    :param destination_path: the destination path
    :return:
    """
    source_directory = os.path.join(current_working_directory, source_path)
    patient_list = sorted(os.listdir(source_path))
    ds_store_removal(patient_list)
    for patient in patient_list:
        if patient in list_to_copy:
            source_directory_2 = os.path.join(source_directory, patient)
            landmark_list = os.listdir(source_directory_2)
            ds_store_removal(landmark_list)
            for landmark in landmark_list:
                source_directory_3 = os.path.join(source_directory_2, landmark)
                image_list = sorted(os.listdir(source_directory_3))
                ds_store_removal(image_list)
                for n_img, image in enumerate(image_list):
                    source = os.path.join(source_directory_3, image)
                    new_image_name = landmark[2:] + '_' + patient + '_%d.jpeg' % n_img
                    destination = os.path.join(current_working_directory, destination_path,
                                               destination_folder, landmark, new_image_name)
                    # shutil.copy(source, destination)
                    shutil.move(source, destination)
                    # print('source:', source, '\ndestination:', destination)
                    # print('\n')
        else:
            pass
    print('Tread softly because you tread on my dreams.')


# vres allo onoma! # Used in Rhode Island Dataset.
def minimum_images(current_work_directory, src_path):
    """
    A function that find which class/ landmark has the fewer images and how many are them.
    :param current_work_directory: os.getcwd()
    :param src_path: where the patients/ cases are stored. (dataset_split/train or train_validation etc.)
    :return: int, the minimum number of images in any class.
    """
    landmark_list = sorted(os.listdir(os.path.join(current_work_directory, src_path)))
    ds_store_removal(kappa=landmark_list)
    images_per_class = []
    for landmark in landmark_list:
        image_list = os.listdir(os.path.join(current_work_directory, src_path, landmark))
        ds_store_removal(kappa=image_list)
        print(f'landmark: {landmark}, images: {len(image_list)}.')
        n_images = len(image_list)
        images_per_class.append(n_images)
    print('\n')
    # minimum = min(images_per_class)
    images_per_class = np.asarray(images_per_class)
    minimum = np.min(images_per_class[np.nonzero(images_per_class)])
    landmarks_name = ['1_esophagus', '2_stomach', '3_small_bowel', '4_colon']
    print(f'minimum: {minimum}, landmark: {landmarks_name[np.argmin(images_per_class[np.nonzero(images_per_class)])]}.')
    return minimum


# vres allo onoma! Used in Rhode Island Dataset.
def copy_some_images(current_work_directory, src_path, dst_path, minimum, move=True):
    landmark_list = sorted(os.listdir(os.path.join(current_work_directory, src_path)))
    ds_store_removal(kappa=landmark_list)
    for landmark in landmark_list:
        # print(f'landmark: {landmark}.')
        image_list = os.listdir(os.path.join(current_work_directory, src_path, landmark))
        ds_store_removal(kappa=image_list)
        random.seed(13)
        image_list = random.sample(image_list, minimum)
        for image in image_list:
            # print(f'image: {image}.')
            source_img = os.path.join(current_work_directory, src_path, landmark, image)
            new_landmark = f'{landmark}_ds'
            # print(f'new landmark: {new_landmark}.')
            destination_img = os.path.join(current_work_directory, dst_path, new_landmark, image)
            if move:
                shutil.move(source_img, destination_img)
            else:
                shutil.copy(source_img, destination_img)
    return None
