"""
This python files is the main file for running segmentation algorithm.
"""
from Segmentation import Segmentation

# Folder where image datasets are:
DATA_FOLDER = "Data"

# Which dataset to run the algorithm:
DATASET = "BinRushed"

# Folder to save segmented images:
SAVE_FOLDER = "Segmented"

# If prime images are desired to be saved as RGB:
SAVE_AS_RGB = True

# Name of the csv file containing ignored image indices along with their corresponding dataset. It is generated
# automatically while running segmentation algorithm:
IGNORED_IMAGES_CSV = "ignored_images.csv"

"""
The algorithm runs the segmentation starting from FIRST_IMAGE to LAST_IMAGE. If specific images are desired to be run
use SPECIF_IMAGE, and set FIRST_IMAGE and LAST_IMAGE to None.

Below use cases are only relevant when running segmentation algorithm. When debugging, they are irrelevant.
Use Case 1.1: Runs segmentation algorithm for all images between 5 and 10.  
    FIRST_IMAGE = 5
    LAST_IMAGE = 10
    SPECIFIC_IMAGE = None

Use Case 1.2: Runs segmentation algorithm only for image3, image7 and image11.  
    FIRST_IMAGE = None
    LAST_IMAGE = None
    SPECIFIC_IMAGE = [3,7,11]
    
Use Case 1.3: Runs segmentation algorithm only for specfic images that are read from IGNORED_IMAGES_CSV.  
    FIRST_IMAGE = None
    LAST_IMAGE = None
    SPECIFIC_IMAGE = IGNORED_IMAGES_CSV
"""
# Starting image for the "for loop" (not relevant for debugging):
FIRST_IMAGE = 1

# End image for the "for loop" (not relevant for debugging):
LAST_IMAGE = 2

# The segmentation algorithm runs only for specified images. It is beneficial especially when trying to segment
# incorrectly segmented images. If FIRST_IMAGE and LAST_IMAGE are used, set SPECIFIC_IMAGE as None:
SPECIFIC_IMAGE = None

# Below values might be fine-tuned for better segmentations:
# Threshold value for inner edge detection algorithm:
INNER_EDGE_THRESHOLD = 20

# Threshold area value while removing small areas in labeled images
AREA_THRESHOLD = 350

# Initializes segmentation object
segmentation = Segmentation(data_folder=DATA_FOLDER, dataset=DATASET, save_folder=SAVE_FOLDER,
                            save_as_RGB=SAVE_AS_RGB, ignored_images_csv=IGNORED_IMAGES_CSV)

"""
Uncomment below for debugging. 

Use Case 2.1: Debugs all 6 different segmentations of image23 => image23-1, image23-2, ..., image23-6.
            DEBUG_IMAGE = "23"
Use Case 2.2: Debugs only the specified segmentation of image23, i.e image23-4
            DEBUG_IMAGE = "23-4"
"""
# DEBUG_IMAGE = "1-1"
# segmentation.debug_segmentation_algorithm(INNER_EDGE_THRESHOLD,AREA_THRESHOLD,DEBUG_IMAGE)

"""
Uncomment below for running the segmentation algorithm. 

It segments all images defined by either FIRST_IMAGE and LAST_IMAGE or SPECIFIC_IMAGE.
Then it saves each segmented images as a jpg file which helps to debug visually. 
Finally it combines all segmentations corresponding to prime image and saves them in a hdf5 file. 
"""
segmentation.execute_segmentation_algorithm(first_image=FIRST_IMAGE, last_image=LAST_IMAGE, specific_images=SPECIFIC_IMAGE,
                                            inner_edge_threshold=INNER_EDGE_THRESHOLD, area_threshold=AREA_THRESHOLD)

