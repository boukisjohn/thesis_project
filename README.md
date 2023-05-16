# thesis_project
My thesis project on Automatic Classification of Anatomical Landmarks from Gastrointestinal Track in Endoscopy Images using Machine Learning Technics.

- This repository includes code files used for 1)data/ file preprocess, 2)feature extraction, 3)model optimization, 4)train and 5)final evaluation of the trained model for 2 seperate datasets I)HyperKvasir Dataset and II) Rhode Island Dataset.
- For HyperKvasir:
a) download the dataset from .
b) unzip the labeled and unlabeled images files.
c) merge the anatomical landmarks of upper and lower GI to one common file alongside with a file called background which consists of 1000 random images from unlabeled images.

- For Rhode Island Dataset
a) download the dataset from .
b) unzip all the cases and landmarks.
c) make a folder with all 474 cases in it.
d) create folders (directories) for i)train, validation, test and ii)train_downsample, validation_downsample, test_downsample.

- For conducting any experiment with the given code make sure that each directory matches with yours. 
