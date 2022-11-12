"""
This python file contains the Segmentation class used for segmentation algorithm.
"""
import os
import cv2
import numpy as np
from skimage.morphology import (closing, square, opening, erosion, dilation)
from skimage.measure import (label, regionprops)
from scipy import ndimage as ndi
from seg_utils import (save_as_hdf5,
                       save_dictionary_as_csv,
                       load_image,
                       plot_image,
                       read_csv_as_dictionary)

class Segmentation:

    def __init__(self, data_folder, dataset, save_folder, save_as_RGB, ignored_images_csv, class_number=4):
        self.__data_folder = data_folder
        self.__dataset = dataset
        self.__output_path = save_folder
        self.__save_folder = os.path.join(save_folder, dataset)
        self.__image_indices = [1, 2, 3, 4, 5, 6] # 6 different segmentations
        self.__save_as_RGB = save_as_RGB
        self.__ignored_images_csv = ignored_images_csv
        self.__class_number = class_number
        self.__flagged_segmentation = False

    def __inner_edge_detection(self, prime_image, image, threshold, image_name, debug, save_folder, image_index, verbose=True):
        """
        Algorithm to detect inner edge.
        First, it subtracts prime image and image.
        Then it applies binary threshold to detect inner edge.
        Finally it applies closing operation to have a continuous edge.

        :param prime_image: Unsegmented image
        :param image: Segmented image by doctors
        :param threshold: Threshold value
        :param image_name: Name of the image whose inner edge is detected
        :param debug: If True, will plot the resulting image, else it will save the result to the save_folder
        :param save_folder: The folder where the result will be saved
        :param verbose: If True, it will plot whole steps, else it plots only resulting image
        :return: binary image where inner edge pixels are set to 1
        """
        print("         Running inner edge detection algorithm...")
        difference = cv2.subtract(prime_image, image)  # subtracts to images
        if verbose: plot_image(difference, image_name + "_difference", debug)

        # Applies binary thresholding, sets whole pixels bigger than threshold to 1
        _, inner_edge = cv2.threshold(difference, threshold, 1, cv2.THRESH_BINARY)

        # Applies closing operation
        inner_edge = closing(inner_edge, square(3))

        plot_image(inner_edge, image_name + "_inner_edge", debug, save_folder=save_folder)
        return inner_edge

    def __outer_edge_detection(self, prime_image, image_name, debug, save_folder, threshold=5, verbose=True):
        """
        Algorithm to detect outer edge.
        First sets all nonblack pixels to 1 to have a disk.
        Then applies opening operation to get rid of small noisy pixels around the edge
        Then obtains smaller disk by applying erosion operation.
        Finally, obtains outer edge by subtracting smaller disk from bigger disk and applying closing operation

        :param prime_image: Unsegmented image
        :param image_name:  Name of the image whose outer edge is detected
        :param debug: If True, will plot the resulting image, else it will save the result to the save_folder
        :param save_folder: The folder where the result will be saved
        :param threshold: Threshold for nonblack pixel, by default it is set to 3
        :param verbose: If True, it will plot whole steps, else it plots only resulting image
        :return: binary image where outer edge pixels are set to 1
        """
        print("     Running outer edge detection algorithm...\n")
        # Sets all nonblack pixels to 1 (threshold = 1)
        disk = (prime_image > threshold) * 1
        # disk = closing(disk,square(7)) # COMMENT!: used in MagrabiaFemale-image23 to get rid of noisy fundus
        if verbose: plot_image(disk, image_name + " outer disk", debug)

        # Applies opening operation to get rid of small noisy pixels around the edge
        opened_disk = opening(disk, square(3))
        if verbose: plot_image(opened_disk, image_name + " opened outer disk", debug)

        # Obtains smaller disk by applying erosion operation
        smaller_disk = erosion(opened_disk, square(3))
        if verbose: plot_image(smaller_disk, image_name + " smaller outer disk", debug)

        # Detects outer edge by subtracting smaller disk from bigger disk.
        # Also applies closing operation to obtain continuous edge
        outer_edge = closing(cv2.subtract(opened_disk, smaller_disk), square(2))
        # If outer edge is too thin, uncomment below line to correctly segment the image:
        outer_edge = dilation(outer_edge, square(2))
        # outer_edge = closing(outer_edge, square(3))

        plot_image(outer_edge, image_name + "_outer_edge", debug, save_folder=save_folder)

        return outer_edge

    def __label_image(self, merged_edges, image_name, area_threshold=500):
        """
        Labels a binary image where all edge pixels are set to 1 by using skimage.measure.label
        If number of labels detected is not 3 (background is not counted by default), it removed small regions.

        :param merged_edges: Binary image where all edge pixels are set to 1
        :param area_threshold: Areas below this threshold are removed
        :return labeled image and number of labels
        """
        print("         Running labeling algorithm...")
        image = np.copy(merged_edges)
        labeled_image, N = label(image, return_num=True, connectivity=1)

        # N should be 3 since: (label function does not count background class therefore N=4-1=3)
        # Labels: 0 - background, 1 - outer edge, 2 - outer inner edge, 3 - inner inner edge

        # If due to noise small regions are labeled, it removes those regions according to their respective area.
        if (N != 3):
            # print("For {}:".format(image_name))
            print('         Not 4 classes are detected. Now removing small regions...')
            for prop in regionprops(labeled_image):
                if (prop.area < area_threshold):
                    for coord in prop.coords:
                        image[coord[0], coord[1]] = 0
            labeled_image, N = label(image, return_num=True, connectivity=2)

        return labeled_image, N

    def __find_seed_point(self, image, label):
        """
        For a given image which is processed to find edges, finds a single seed point inside the region.
        It randomly selects 2 points from points belonging to the given label. Since circular region is convex, a line
        connecting any two points on the boundary will be inside the region.Therefore a new point (candidate seed point) is
        calculated by averaging these randomly selected 2 points. If the pixel label at the new point is 0, it returns the
        seed point. Otherwise it continues on searching.

        :param image: edge detected image
        :param label: class of the region
        :return: seed point
        """
        labelIndices = np.where(image == label)  # points belonging the label
        max_iteration = 750
        iter = 0
        while (1):
            index1 = np.random.randint(0, labelIndices[0].shape)  # randomly selects a point belonging to given label
            index2 = np.random.randint(0, labelIndices[0].shape)  # randomly selects a point belonging to given label
            # Row index, found by averaging row indices of 2 randomly selected points belonging to the label:
            row = int((labelIndices[0][index1] + labelIndices[0][index2]) / 2)
            # Column index, found by averaging column indices of 2 randomly selected points belonging to the label:
            col = int((labelIndices[1][index1] + labelIndices[1][index2]) / 2)
            if (image[
                row, col] == 0 and iter <= max_iteration):  # if the pixel label at the calculated candidate seed point is 0, returns the point as seed
                return [(row, col)]
            elif (iter > max_iteration):
                return -1
            iter += 1

    def __region_growing_algorithm(self, image, label, seeds):
        """
        Applie region growing algorithm to the given image.

        :param image: labeled image
        :param label: which label to apply region growing. i.e: for outer edge label = 1
        :param seeds: a single pixel coordinate inside the region where whole other inside pixels will be labeled
        :return: the image where all pixels inside the given region labeled with given label
        """
        # Neighboring directions the seed point:
        directs = [(-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0)]
        # Stores if the pixel is visited:
        visited = np.zeros(shape=(image.shape), dtype=np.uint8)
        height = image.shape[0]
        width = image.shape[1]
        while len(seeds):
            seed = seeds.pop(0)
            row = seed[0]
            col = seed[1]
            # Visit point (x,y)
            image[row, col] = label  # labels the seed point with given label
            # Stores that pixel is visited so that the algorithm will not visit the same pixel again
            visited[row][col] = 1
            # Neighboring pixel coordinate
            for direct in directs:
                cur_row = row + direct[0]
                cur_col = col + direct[1]
                # If pixel coordinate that is going to be visited is not legal, continues with next
                if cur_col < 0 or cur_row < 0 or cur_row >= height or cur_col >= width:
                    continue
                # The current pixel is not visited and its label is 0 (background label)
                if (not visited[cur_row][cur_col]) and (image[cur_row][cur_col] == 0):
                    image[cur_row][cur_col] = label  # labels the current pixel with given label
                    visited[cur_row][cur_col] = 1
                    # Add pixel coordinates to the seed so that in the next iterations its neighbors will be checked
                    seeds.append((cur_row, cur_col))

    def __segmented_incorrectly(self, message, image, image_name, prime_index, debug):
        """
        If an image is segmented incorrectly, prints "message", saved an empty image into segmented folder such that
        it is easy to detect visually, and adds image index into csv file where ignored image indices are stored.
        """
        print(message)
        plot_image(np.zeros_like(image), image_name + "_segmented_image", debug,
                        RGB=True, save_folder=self.__save_folder)
        # If the image is already flagged as incorrectly-segmented or if in debug mode, it does not add the image index
        # to the csv file:
        if not (self.__flagged_segmentation or debug):
            save_dictionary_as_csv(save_folder=self.__output_path,
                                   csv_file_name=self.__ignored_images_csv,
                                   dictionary={self.__dataset: prime_index},
                                   mode='a')
            # Flags the image as incorrectly segmented:
            self.__flagged_segmentation = True

    def __is_segmented_correctly(self, segmented_image):
        """
        Checks if the image is correctly segmented.
        """
        # If the maximum iteration during segmentation algorithm is reached, it returns an int.
        if type(segmented_image) is int:
            return 0
        list_segmented = segmented_image.flatten().tolist()
        set_segmented = set(list_segmented)
        # Number of distinct labels in the segmented image should be equal to class number (4)
        return (len(set_segmented) == self.__class_number)

    def __run_region_growth_algorithm(self, labeled_image):
        """
        Applies region growing algorithm to labeled image
        :param labeled_image: Labeled image to be segmented
        :return: Segmented image
        """
        print("         Running region growing algorithm...")
        segmented_image = np.copy(labeled_image)
        labels = [3, 2, 1]
        for label in labels:
            # Finds a single seed point corresponding to label:
            seeds = self.__find_seed_point(image=segmented_image,
                                    label=label)
            if (seeds == -1):
                return -1
            # From the found single seed points, region starts to grow by labeling all pixel inside the circle whose label is 0 as label
            self.__region_growing_algorithm(segmented_image, label, seeds)
        return segmented_image

    def __run_segmentation_algorithm(self, inner_edge_threshold, area_threshold, debug=False):
        """
        Runs the segmentation algorithm.

        :param inner_edge_threshold: threshold value used during inner edge detection
        :param area_threshold: threshold area value while removing small areas in labeled images
        """
        path_to_data = os.path.join(self.__data_folder, self.__dataset)
        if not os.path.exists(self.__save_folder): os.makedirs(self.__save_folder)

        for prime_index in self.__prime_indices:

            self.__flagged_segmentation = False
            print("Working on image{}prime\n".format(prime_index))
            prime_image_name = os.path.join(path_to_data, "image" + str(prime_index) + "prime")
            # Loads prime image and converts it to gray image:
            prime_image = load_image(prime_image_name, debug, verbose=True)
            # Detects outer edge:
            outer_edge = self.__outer_edge_detection(prime_image, prime_image_name, debug, save_folder=None, verbose=True)
            # Empy list to store different labeled versions of the prime image:
            labels = []
            for image_index in self.__image_indices:
                image_name = os.path.join(path_to_data, "image" + str(prime_index) + "-" + str(image_index))
                print("     Working on image{}-{}".format(prime_index, image_index))
                # Loads image and converts it to gray image:
                image = load_image(image_name, debug, verbose=True)
                # Detects inner edge:
                inner_edge = self.__inner_edge_detection(prime_image, image, inner_edge_threshold, image_name, debug,
                                                         save_folder=None, image_index=image_index, verbose=True)
                merged_edges = outer_edge + inner_edge
                # If want to save intermediate results use like below:
                # plot_image(merged_edges, image_name + "_merged_edges", DEBUG, save_folder=SAVE_FOLDER)
                plot_image(merged_edges, image_name + "_merged_edges", debug, save_folder=None)
                # Labels binary image:
                labeled_image, N = self.__label_image(merged_edges, image_name, area_threshold=area_threshold)
                # If the number of detected classes are not 3, it skips to other image since there is a problem.
                if (N != 3):
                    # Number of classes are not 4. It still requires an improvement.
                    self.__segmented_incorrectly(message="         Number of classes are not 4. Couldn't segment, skipping to the next image!",
                                                 image=merged_edges, image_name=image_name, prime_index=prime_index, debug=debug)
                    continue
                plot_image(labeled_image, image_name + "_labeled_image", debug, RGB=True, save_folder=None)
                # Obtains segmented image where all pixels are labeled:
                segmented_image = self.__run_region_growth_algorithm(labeled_image)
                # Checks if the image is segmented correctly:
                if not self.__is_segmented_correctly(segmented_image):
                    self.__segmented_incorrectly(
                        message="         The image is not segmented correctly, skipping to the next image!",
                        image=merged_edges, image_name=image_name, prime_index=prime_index, debug=debug)
                    continue
                plot_image(segmented_image, image_name + "_segmented_image", debug, RGB=True,
                           save_folder=self.__save_folder)
                # Adds segmented image to the list of possible segmentations of the original image:
                labels.append(segmented_image)
            # Saves the image and its possible segmentations as a hdf5 file. Each prime image is saved as
            # a hdf5 file separately. Then they are combined in a single hdf5 file later:
            if not debug:
                print("\n         Saving original-sized files...\n")
                # If images are desired to be saved as RGB:
                if self.__save_as_RGB:
                    prime_image = load_image(prime_image_name, debug, return_RGB=True)
                save_as_hdf5(prime_image, labels, self.__save_folder, file_name="image" + str(prime_index) + "_original")


    def debug_segmentation_algorithm(self, inner_edge_threshold, area_threshold, image):
        """
        Debugs the segmentation algorithm on a specific image.

        :param inner_edge_threshold: threshold value used during inner edge detection
        :param area_threshold: threshold area value while removing small areas in labeled images
        :param image: index of the prime image that is desired to be debugged. If it is a single number, i.e. "23",
                      it debugs all segmentations of image23. Else if it has a specified segmentation, i.e. "23-4", it
                      debugs only image23-4.
        """
        image_name_splitted = image.split("-")
        debug_image = int(image_name_splitted[0])
        debug_index = None
        # If there is a specified segmentation that is desired to be debugged:
        if (len(image_name_splitted) == 2): debug_index = int(image_name_splitted[1])
        self.__prime_indices = np.array([debug_image])
        if debug_index is not None: self.__image_indices = np.array([debug_index])
        self.__run_segmentation_algorithm(inner_edge_threshold, area_threshold, debug=True)

    def execute_segmentation_algorithm(self, first_image, last_image, specific_images, inner_edge_threshold, area_threshold):
        """
        Executes the segmentation algorithm.

        :param first_image: start index for segmentation algorithm
        :param last_image: end index for segmentation algorithm
        :param inner_edge_threshold: threshold value used during inner edge detection
        :param area_threshold: threshold area value while removing small areas in labeled images
        """
        assert (((first_image is None) and (last_image is None) and (specific_images is not None)) or
                ((first_image is not None) and (last_image is not None) and (specific_images is None))), \
            "\nPlease either set SPECIFIC_IMAGES to None and use FIRST_IMAGE and LAST_IMAGE (see Use Case 1.1); " \
            "or use SPECIFIC_IMAGES and set FIRST_IMAGE and LAST_IMAGE to None (see Use Case 1.2 and 1.3). "

        if first_image is not None: self.__prime_indices = np.arange(first_image,last_image+1)
        elif specific_images is not None:
            if type(specific_images) is list: self.__prime_indices = np.asarray(specific_images)
            elif type(specific_images) is str:
                dict = read_csv_as_dictionary(os.path.join(self.__output_path,self.__ignored_images_csv))
                self.__prime_indices = np.asarray(dict[self.__dataset], dtype="int32")

        self.__run_segmentation_algorithm(inner_edge_threshold, area_threshold)



