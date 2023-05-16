import os
from preprocess_tools import video_to_frames
from preprocess_tools import ds_store_removal

cwd = os.getcwd()


# -------------------------------------------------
# Upper GI.
# -------------------------------------------------

upper_gi_path = 'HyperKvasir Videos/labeled-videos/upper-gi-tract'
landmark_upper_list = sorted(os.listdir(os.path.join(cwd, upper_gi_path)))
ds_store_removal(kappa=landmark_upper_list)
# print(landmark_upper_list)

for landmark_upper in landmark_upper_list:
    video_upper_list = sorted(os.listdir(os.path.join(cwd, upper_gi_path, landmark_upper)))
    ds_store_removal(kappa=video_upper_list)
    for video in video_upper_list:
        imageName = f'{video[:-4]}'
        sourcePath = os.path.join(cwd, upper_gi_path, landmark_upper, video)
        folderPath = os.path.join(cwd, 'HyperKvasir Test', imageName)
        video_to_frames(current_work_dir=cwd, source_path=sourcePath, images_name=imageName, folder_path=folderPath)

# -------------------------------------------------
# Lower GI.
# -------------------------------------------------

lower_gi_path = 'HyperKvasir Videos/labeled-videos/lower-gi-tract'
landmark_lower_list = sorted(os.listdir(os.path.join(cwd, lower_gi_path)))
ds_store_removal(kappa=landmark_lower_list)
# print(landmark_lower_list)

for landmark_lower in landmark_lower_list:
    video_lower_list = sorted(os.listdir(os.path.join(cwd, lower_gi_path, landmark_lower)))
    ds_store_removal(kappa=video_lower_list)
    for video in video_lower_list:
        imageName = f'{video[:-4]}'
        sourcePath = os.path.join(cwd, lower_gi_path, landmark_lower, video)
        folderPath = os.path.join(cwd, 'HyperKvasir Test', imageName)
        video_to_frames(current_work_dir=cwd, source_path=sourcePath, images_name=imageName, folder_path=folderPath)
