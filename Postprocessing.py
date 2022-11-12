import os
import cv2
import torch
import numpy as np

from seg_utils import (read_csv_as_dictionary,
                       save_dictionary_as_csv,
                       load_image,
                       read_hdf5,
                       save_as_hdf5,
                       combine_combined_hdf5_files
                       )


class ChangeGrayWithRGB:
    """
     Changes gray scale images in a hdf5 file to RGB version of the same image in the corresponding dataset
    """
    def __init__(self, input_path, dict_path):
        self.input_path = input_path
        self.dict_path = dict_path

    def __call__(self, image_index, **kwargs):
        dict = read_csv_as_dictionary(self.dict_path)
        dataset, image_name = dict.get(str(image_index)).split("_")
        image_path = os.path.join(self.input_path,dataset,"{}prime".format(image_name))
        RGB_image = load_image(image_path,debug=False,return_RGB=True)
        return RGB_image


class OmitBackgroundClass:
    """
    Omits background class and makes background and fundus as same class.
    Originally there were 4 classes (0:background, 1:fundus, 2:disc, 3:cup); however after this function =>
    there are 3 classes(0:background and fundus, 1:disc, 2:cup).
    """
    def __call__(self, image, **kwargs):
        image[image <= 1] = 1
        return image - 1


class NormalizeImage:
    """
    Normalizes the image.
    """
    def __call__(self, image, axis=None, **kwargs):
        # If image has RGB channels, each channel is normalize within themselves:
        if (len(image.shape) == 3):
            axis = (0,1)
        mean = np.mean(image, axis=axis, keepdims=True)
        std = np.std(image, axis=axis, keepdims=True)
        return (image - mean) / std


class CropImageWRTCenter:
    """
    Crops image with respect to calculated center of disc such that the disc is positioned in the middle of the image.
    """
    def __init__(self, rescale_size, combined_labels):
        self.__rescale_size = rescale_size
        self.centroids = self.calculate_centroids(combined_labels)

    def __call__(self, image, image_index, label_index=None, **kwargs):
        # If labeled image is going to be cropped, it uses its own calculated disc center:
        if label_index is not None:
            centroid = self.__centroids[image_index][label_index]
        # If prime image is going to be cropped, it uses mean of disc centers in 6 segmentations since it is hard to
        # calculate disc center in prime image
        else:
            centroid = np.mean(self.__centroids[image_index], axis=0)
        return self.crop_image_wrt_disc_center(image,centroid)

    def localize_disc_center(self, image, is_background_omitted=False):
        """
        Localizes the center of the disc and returns the centroid.
        """
        # Makes pixels whose labels are 1 or 2 as 1, others zero:
        if is_background_omitted: binary_mask = np.ma.masked_where(image >= 1, image)
        # Makes pixels whose labels are 2 or 3 as 1, others zero:
        else: binary_mask = np.ma.masked_where(image >= 2, image)

        # Calculates the centroid by calculating the mean of coordinate of pixels set to 1 with previous line of code:
        centroid = np.mean(np.argwhere(binary_mask.mask), axis=0)
        return [int(centroid[0]), int(centroid[1])]

    def calculate_centroids(self, data):
        """
        Calculates centroids of 6 segmented images.
        """
        all_centroids = []
        for dat in data:
            centroids = []
            for sub_dat in dat:
                centroid = self.localize_disc_center(sub_dat)
                centroids.append(centroid)
            all_centroids.append(centroids)
        return np.asarray(all_centroids)

    def crop_image_wrt_disc_center(self, image, centroid):
        """
        Crops the image with respect to disc center.
        It returns an image with size rescale_size X rescale_size centered whose center is the disc center.

        Note: May need to decentralize ROI!
        """
        # If ROI is close to the edge, in order not to get negative index max (a,0) is used:
        row = max(int(centroid[0]) - self.__rescale_size // 2, 0)
        col = max(int(centroid[1]) - self.__rescale_size // 2, 0)

        return image[row:row + self.__rescale_size, col:col + self.__rescale_size]


class CropImageWRTHeight:
    """
    Crops image with respect to height in order to get square image.
    """
    def __call__(self, image, **kwargs):
        height, width = image.shape[0], image.shape[1]
        start_indx, end_indx = int(width/2 - height/2), int(width/2 + height/2)
        assert (end_indx-start_indx == height)
        return image[:,start_indx:end_indx]


class DownsampleImage:
    """
    Downsamples image with same field-of-view but with lower resolution.
    At the end a rectangular image is obtained since height of the image is considered while downsampling.
    """
    def __init__(self, rescale_size):
        self.__rescale_size = rescale_size

    def __call__(self, image, label_index=None, **kwargs):
        # If labeled image (segmentation) is going to be downsampled, it uses cv2.INTER_NEAREST as interpolation func:
        if label_index is not None:
            interpolation_func = cv2.INTER_NEAREST
        # If prime image is going to be downsampled, it uses cv2.INTER_CUBIC as interpolation func
        else:
            interpolation_func = cv2.INTER_CUBIC
        height, width = image.shape[0], image.shape[1]
        rescaled_height = self.__rescale_size
        rescale_factor = height/rescaled_height
        rescaled_width = int(width // rescale_factor)
        resized_img = cv2.resize(image,
                                 dsize=(rescaled_width, rescaled_height),
                                 interpolation=interpolation_func)
        return resized_img

class ChangeOrderOfAxes:
    def __call__(self, image, **kwargs):
        return image.transpose(2,0,1)

class ToTensor:
    def __call__(self, image, **kwargs):
        return torch.from_numpy(image)

class ComposeTransform:
    """
    Transform class that combines multiple other transforms into one
    """
    def __init__(self, transforms):
        """
        :param transforms: transforms to be combined
        """
        self.__transforms = transforms

    def insert(self, index, transform):
        self.__transforms.insert(index, transform)

    def append(self, transform):
        self.__transforms.append(transform)

    def __call__(self, **kwargs):
        img = kwargs.get("image")
        kwargs.pop("image")
        for transform in self.__transforms:
            img = transform(image=img,**kwargs)
        return img


class PostProcessing:

    def __init__(self, path_to_images, path_to_segmented_images, datasets, is_already_combined,
                 combined_file_name, endswith, dictionary_name, ignore_these, rescale_size):

        self.__path_to_code = os.getcwd()
        self.__path_to_images = os.path.join(self.__path_to_code, path_to_images)
        self.__path_to_segmented_images = os.path.join(self.__path_to_code, path_to_segmented_images)
        self.__datasets = datasets
        self.__combined_file_name = combined_file_name
        self.__endswith = endswith
        self.__dictionary_name = dictionary_name
        # If ignore_these = -1, finds ignored image indices by reading ignored_images_csv:
        self.ignore_these = ignore_these if (type(ignore_these) is not str) else self.find_ignored_images(ignore_these)

        self.combined_images = None
        self.combined_labels = None
        self.size = None

        # If already-combined hdf5 file is going to be used:
        if is_already_combined: self.initialize_from_combined_hdf5()
        # If seperate hdf5 files are going to be combined:
        else:
            assert (len(self.__combined_file_name) == 1), \
                "If seperate hdf5 files are going to be combined (IS_ALREADY_COMBINED = 0), enter the COMBINED_FILE_NAME as " \
            "a single name, i.e COMBINED_FILE = [\"procesed\"]. Yours have the length of {}. Look at Option 1 and 2"\
                    .format(len(self.__combined_file_name))
            self.combine_hdf5()

        # Initializes transform functions:
        self.change_gray_with_RGB = ChangeGrayWithRGB(input_path=path_to_images,
                                                      dict_path=os.path.join(path_to_segmented_images,dictionary_name))
        self.omit_background_class = OmitBackgroundClass()
        self.normalize_image = NormalizeImage()
        self.downsample_image = DownsampleImage(rescale_size=rescale_size)
        self.crop_image_wrt_height = CropImageWRTHeight()
        self.crop_image_wrt_center = CropImageWRTCenter(rescale_size=rescale_size,combined_labels=self.combined_labels)

    def find_ignored_images(self,csv_file):
        """
        Find ignored image indices from the csv file such that those images are ignored while combining hdf5 files.

        :param csv_file: name of the csv file containing ignored image indices along with corresponding dataset
        """
        mydict = read_csv_as_dictionary(path_to_file=os.path.join(self.__path_to_segmented_images,csv_file))
        # Converts string values into integer values:
        ignored_images_dict = {}
        for key, val in mydict.items():
            if (type(val) is list):
                ignored_images_dict[key] = list(map(int, val))
            elif (type(val) is str):
                temp_list = []
                temp_list.append(int(val))
                ignored_images_dict[key] = temp_list
        return ignored_images_dict

    def initialize_from_combined_hdf5(self):
        """
        If is_already_combined = True,
        """
        if len(self.__combined_file_name) == 1:
            self.combined_images, self.combined_labels = read_hdf5(os.path.join(self.__path_to_segmented_images,self.__combined_file_name[0]))
        else:
            self.combined_images, self.combined_labels = combine_combined_hdf5_files(folder_path=self.__path_to_segmented_images,
                                                                                     files_to_be_combined=self.__combined_file_name,
                                                                                     file_name=self.__combined_file_name.pop())
        self.size = self.combined_images.shape[0]

    def combine_hdf5(self):
        """
        Combines single hdf5 files whose ending is same as endswith given in folder_path.
        It loops over datasets list to find hdf5 files in folder_path/dataset ending with endswith and combines them.
        Some images may not be segmented correctly, if they are provided as ignore_these. While combining, it ignores those images.
        Then saves it as file_name.
        """
        all_images = []
        all_labels = []
        count = 0
        dict = {}
        print("Combining hdf5 files:")
        for dataset in self.__datasets:
            print("\n{}:\n".format(dataset))
            for file in os.listdir(os.path.join(self.__path_to_segmented_images, dataset)):
                if file.endswith(".hdf5"):  # if the file is a hdf5 file
                    split = file.split("_")
                    if (self.__endswith in split[1]):  # if the file ends with provided endswith
                        # If some images are provided to be ignored for the dataset
                        if (self.ignore_these is not None and self.ignore_these.get(dataset) is not None):
                            image_number = int(split[0].replace("image", ""))
                            if (image_number in self.ignore_these.get(dataset)):
                                print("     {} is ignored".format(file))
                                continue
                        dict[count] = dataset + '_' + split[0]
                        image, labels = read_hdf5(os.path.join(self.__path_to_segmented_images, dataset, file.replace(".hdf5", '')))
                        count += 1

                        all_images.append(image)
                        all_labels.append(labels)
        # for index, element in enumerate(all_images):
        #     print("image index:{}, image shape:{}, data type:{}".format(index, element.shape, element.dtype))
        # for index, element in enumerate(all_labels):
        #     print("label index:{}, image shape:{}, data type:{}".format(index, element.shape, element.dtype))
        self.combined_images = np.asarray(all_images)
        self.combined_labels = np.asarray(all_labels)
        print("\nSaving combined hdf5 files in a single one...")
        save_as_hdf5(self.combined_images, self.combined_labels, self.__path_to_segmented_images, self.__combined_file_name[0])
        save_dictionary_as_csv(save_folder=self.__path_to_segmented_images,csv_file_name=self.__dictionary_name,
                               dictionary=dict, mode='w')
        self.size = self.combined_images.shape[0]

    def transform_prime_images(self, prime_image_transforms):
        """
        Applies transforms in image_transforms to images.
        """
        transformed_images = []
        for index, img in enumerate(self.combined_images):
            data = prime_image_transforms(image=img, image_index=index)
            transformed_images.append(data)
        return np.asarray(transformed_images)

    def transform_labeled_images(self, labeled_image_transforms):
        """
        Applies transforms in label_transforms to segmentations.
        """
        transformed_labels = []
        for image_idx, label in enumerate(self.combined_labels):
            transformed_label = []
            for label_idx, sub_label in enumerate(label):
                data = labeled_image_transforms(image=sub_label, image_index=image_idx, label_index=label_idx)
                transformed_label.append(data)
            transformed_labels.append(transformed_label)
        return np.asarray(transformed_labels)

    def process_data(self, change_gray_with_RGB, downsample_image, crop_image, normalize_image, omit_background_class):
        """
        Below transform functions that are implemented in postprocessing are listed and example use cases are showed afterwards:

        Transforms:
        Image-specific transforms (cannot apply to labels):
            postprocessing.change_gray_with_RGB: replaces gray images in combined hdf5 file with rgb ones
            postprocessing.normalize_image: normalizes the image
        Label-specific transforms (cannot apply to prime images):
            postprocessing.omit_background_class: omits background class such that 0: background and fundus, 1: disc, 2: cup
        For both:
            postprocessing.downsample_image: downsamples image such that same field-of-view with lower resolution
            postprocessing.crop_image_wrt_height: crops image with respect to image height to get square image
            postprocessing.crop_image_wrt_center: crops image with respect to calculated centroid of disc

        ComposeTransform includes the transforms that are going to be applied according to given order. It is possible to add
        or remove transforms as desired.
        i.e: First gray images are replaced with RGB ones, then images are downsampled to RESCALE_SIZE, and finally the images are
        cut with respect to height to get a square image
        prime_image_transforms = ComposeTransform([
            postprocessing.change_gray_with_RGB,
            postprocessing.downsample_image,
            postprocessing.crop_image_wrt_height
        ])
        """
        assert not (downsample_image and crop_image), \
            "You cannot both do downsampling and cropping without downsampling at the same time. " \
            "Set either of them or both of them to 0."

        if downsample_image:
            prime_image_transforms = ComposeTransform([
                self.crop_image_wrt_height,
                self.downsample_image
            ])
            labeled_image_transforms = ComposeTransform([
                self.crop_image_wrt_height,
                self.downsample_image
            ])

        elif crop_image:
            prime_image_transforms = ComposeTransform([
                self.crop_image_wrt_center
            ])
            labeled_image_transforms = ComposeTransform([
                self.crop_image_wrt_center
            ])

        if change_gray_with_RGB:
            prime_image_transforms.insert(0,self.change_gray_with_RGB)

        if normalize_image:
            if not downsample_image and not crop_image:
                prime_image_transforms = ComposeTransform([
                    self.normalize_image
                ])
            else:
                prime_image_transforms.append(self.normalize_image)

        if omit_background_class:
            if not downsample_image and not crop_image:
                labeled_image_transforms = ComposeTransform([
                    self.omit_background_class
                ])
            else:
                labeled_image_transforms.insert(0,self.omit_background_class)

        return self.transform_prime_images(prime_image_transforms), self.transform_labeled_images(labeled_image_transforms)





