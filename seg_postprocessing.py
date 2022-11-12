"""
This python file is used to further process saved hdf5 files.
For example if a new sized images are needed, it is possible to load original segmented images and then
process it according to desired properties.
"""
import os
from seg_utils import save_as_hdf5
from Postprocessing import PostProcessing

# Folder where images to be segmented are:
PATH_T0_IMAGES = "Data"

# Folder where segmented images are:
PATH_TO_SEGMENTED_IMAGES = "Segmented"

# Set it to True if already-combined hdf5 file is going to be used for post-processing, else Set it to False to combine
# hdf5 files in DATASETS:
IS_ALREADY_COMBINED = False

"""
Name of the combined hdf5 file. There are 3 options:

OPTION 1: If IS_ALREADY_COMBINED = False, saves combined hdf5 files with this name.

OPTION 2: Else if IS_ALREADY_COMBINED = True, finds the already-combined hdf5 file named as COMBINED_FILE_NAME
in PATH_TO_SEGMENTED_IMAGES.
"""
COMBINED_FILE_NAME = ["combined_images"]
"""
OPTION 3: Else if IS_ALREADY_COMBINED = True, finds the already-combined hdf5 files named as COMBINED_FILE_NAME
in PATH_TO_SEGMENTED_IMAGES. It is a list of names of already-combined hdf5 files, the last one ine the name of 
combined hdf5 file that is going to be saved:
"""
# COMBINED_FILE_NAME = [ "combined_images1",
#                        "combined_images2",
#                        "combined_images_final"]
"""
Below is relevant only if hdf5 files are going to be combined (no already-combined hdf5 files is going to be used, 
only if IS_ALREADY_COMBINED = False)
"""
# Which datasets are going to be used for combining and processing. Relevant only if IS_ALREADY_COMBINED = False:
DATASETS = ["BinRushed"]

# Extension name of hdf5 files in DATASETS that are going to be combined in a single hdf5.
# Relevant only if IS_ALREADY_COMBINED = False :
ENDSWITH = "original"

"""
If there are incorrectly-segmented images, they will be ignored while combining hdf5 files. There are three options:

Option 1: Finds the csv file named as IGNORED_IMAGES which is generated automatically while running
segmentation algorithm in seg_main.py. It contains ignored image indices along with their corresponding dataset.
By reading that csv file, it finds ignored images:
"""
# IGNORED_IMAGES = "ignored_images.csv"
"""
# Option 2: Ignored image indices is written manually along with their corresponding dataset:
"""
# IGNORED_IMAGES = {"BinRushed": [1],
#                   "Messidor": [2,3]
#                   }
"""
Option 3: All images are correctly segmented:
"""
IGNORED_IMAGES = None

"""
This part is related to processing:
"""
# Replaces gray images with RGB ones. If hdf5 files contain already RGB images, set it to 0:
REPLACE_GRAY_IMAGES_WITH_RGB = 0

# Full size images are cropped or downsampled to RESCALE_SIZE:
RESCALE_SIZE = 512

# Downsamples images (same field-of-view, lower resolution),
# either set DOWNSAMPLE_IMAGE or CROP_IMAGE_WITHOUT_DOWNSAMPLING to 1, both cannot be set to 1:
DOWNSAMPLE_IMAGE = 0

# Crops image without downsampling (smaller field-of-view, same resolution):
CROP_IMAGE_WITHOUT_DOWNSAMPLING = 0

# Normalizes image:
NORMALIZE_IMAGE = 0

# Omits background class in segmentations:
OMIT_BACKGROUND_CLASS = 0

# Combined images and labels are processed and then they are saved as PROCESSED_FILE_NAME:
PROCESSED_FILE_NAME = "processed_combined"

# Name of the csv file which maps image index of combined images to corresponding image in dataset.
# It is automatically generated while combining hdf5 files. It is used to replace gray images with RGB
# ones via postprocessing.change_gray_with_RGB:
DICT_NAME = "combined_2_single.csv"

# Initializes postprocessing object:
postprocessing = PostProcessing(path_to_images=PATH_T0_IMAGES, path_to_segmented_images=PATH_TO_SEGMENTED_IMAGES,
                                datasets=DATASETS, is_already_combined=IS_ALREADY_COMBINED,
                                combined_file_name=COMBINED_FILE_NAME, endswith=ENDSWITH, dictionary_name=DICT_NAME,
                                ignore_these=IGNORED_IMAGES, rescale_size=RESCALE_SIZE)

"""
Comment below line if there will be no transformation:
"""
transformed_images, transformed_labels = postprocessing.process_data(REPLACE_GRAY_IMAGES_WITH_RGB,
                                                                     DOWNSAMPLE_IMAGE,
                                                                     CROP_IMAGE_WITHOUT_DOWNSAMPLING,
                                                                     NORMALIZE_IMAGE,
                                                                     OMIT_BACKGROUND_CLASS)

# Saves processed images and labels:
save_as_hdf5(transformed_images,transformed_labels,
             save_path=os.path.join(os.getcwd(),PATH_TO_SEGMENTED_IMAGES),file_name=PROCESSED_FILE_NAME)

