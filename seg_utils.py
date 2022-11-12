"""
This python file contains utility function.
"""
import os
import h5py
import cv2
import csv
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.distance import euclidean

def save_as_hdf5(images, labels, save_path, file_name):
    """
    Saves images and segmented labels as hdf5 file.
    """
    file = h5py.File(os.path.join(save_path, file_name + ".hdf5"), "w")
    images = np.asarray(images)
    file.create_dataset(
        name="image",
        data=images,
        shape=images.shape)
    labels = np.asarray(labels)
    file.create_dataset(
        name="labels",
        data=labels,
        shape=labels.shape)
    file.close()

def read_hdf5(file_path):
    """
    Reads an hdf5 file given in file_path
    """
    with h5py.File(file_path + ".hdf5", 'r') as hf:
        image = np.array(hf["image"], dtype='float64')
        labels = np.array(hf["labels"], dtype="uint8")
    return image, labels

def combine_combined_hdf5_files(folder_path,files_to_be_combined,file_name):
    """
    Combines already combined hdf5 files.
    :param folder_path: The folder where combined files are
    :param files_to_be_combined: Name of the files to be combined as a list
    :param file_name: Saving name of the combine hdf5 file
    """
    combined_images = []
    combined_labels = []
    for file in files_to_be_combined:
        images, labels = read_hdf5(os.path.join(folder_path, file))
        for image in images:
            combined_images.append(image)
        for label in labels:
            combined_labels.append(label)

    all_images = np.asarray(combined_images)
    all_labels = np.asarray(combined_labels)
    print("\nSaving combined hdf5 files in a single one...")
    save_as_hdf5(all_images, all_labels, folder_path, file_name)
    return all_images, all_labels

def save_dictionary_as_csv (save_folder:str, csv_file_name:str, dictionary:dict, mode:str, verbose:bool=True):
    """
    Saves dictionary as a csv file.

    :param save_folder: folder to save the csv file
    :param csv_file_name: name of the csv file to be saved
    :param dictionary: dictionary
    :param mode: writing mode; if mode="a", appends to already-existing csv file:
                 else if mode="w", write over already-existing csv file
    :param verbose: verbosity
    """
    if verbose:print("         Saving the dictionary as {} in {}..".format(csv_file_name,save_folder))
    with open(os.path.join(save_folder,csv_file_name), mode) as f:
        for key in dictionary.keys():
            f.write("%s, %s\n" % (key, dictionary[key]))

def read_csv_as_dictionary(path_to_file:str):
    """
    Reads csv file as a dictionary.

    :param path_to_file: path to csv file that is going to be read.
    """
    with open(path_to_file, mode='r') as infile:
        reader = csv.reader(infile, skipinitialspace=True)
        mydict = {}
        for rows in reader:
            key, val = rows[0], rows[1]
            if key not in mydict:
                mydict[key] = val
            elif type(mydict[key]) == list:
                if val not in mydict[key]: mydict[key].append(val)
            else:
                if val not in mydict[key]: mydict[key] = [mydict[key], val]
        infile.close()
    return mydict

def plot_image(image, title, debug=False, RGB=False, save_folder=None):
    """
    Function to plot images.

    :param image: Image to be plotted
    :param title: Title of the image
    :param debug: If in debug mode: it shows the image, else: it does not show the image but
                save it
    :param RGB: If TRUE: plots the image as a RGB image, else: it uses cmap="gray"
    :param save_folder: Folder name that the image will be saved
    """
    img = np.copy(image)
    if RGB:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap="gray")

    plt.title(title)
    ax = plt.gca()
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    # If in debug mode, shows the image; else saves the image without showing
    if debug:
        plt.show()
    else:
        folders = title.split("\\")
        if save_folder is not None: plt.savefig(os.path.join(save_folder, folders[2] + ".jpg"))
        # if save_folder is not None: plt.savefig(os.path.join(save_folder+".jpg"))

def load_image(image_path, debug, verbose=False, return_RGB=False):
    """
    Loads images given image_path.
    Applies defined image processing.

    :param image_path: Path to the image that is going to be loaded.
    :param debug: If True, will plot the resulting image, else it will save the result to the save_folder
    :param verbose: If True, plots resulting images.
    """
    if os.path.exists(image_path + ".jpg"):
        image = cv2.imread(image_path + ".jpg")  # reads the image
    else:
        image = cv2.imread(image_path + ".tif")
    image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # If verbose True, shows read images.
    if verbose: plot_image(image_RGB, title=image_path, debug=debug, RGB=True)

    # Converts the RGB image to gray scale image:
    image_gray = cv2.cvtColor(image_RGB, cv2.COLOR_RGB2GRAY)
    # If verbose True, shows gray scale images.
    if verbose: plot_image(image_gray, title=image_path + "-gray", debug=debug)
    if return_RGB:
        return image_RGB
    else:
        return image_gray


def create_masked_image(segmented_image,label):
    """
    Creates masked image for the given label.
    """
    mask = np.zeros_like(segmented_image)
    mask[segmented_image == label] = 1
    return np.ma.masked_where(mask == 0, mask)

def plot_overlay_image(image,title,segmented_image,alpha=0.3):
    """
    Plots overlaid image of segmented version and original version to compare.
    """
    plt.imshow(image.astype(np.uint8), cmap="gray", interpolation='none')
    plt.title(title)
    labels = {"0":"jet",
              "1":"prism",
              "2":"ocean"}
    for key in labels:
        masked_image = create_masked_image(segmented_image,int(key))
        plt.imshow(masked_image, cmap=labels[key], interpolation="none", alpha=alpha)

    ax = plt.gca()
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    plt.show()

