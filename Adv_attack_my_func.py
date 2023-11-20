import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data.dataset import ConcatDataset, Subset
from PIL import Image
import argparse
import random
import Backbones
import matplotlib.pyplot as plt


def load_and_split_mnist(list_split):
    """ loading MNIST and retrieving train, val and test indices to create training, val and test sets.

     Args:
       list_split:

    Returns:
        4 objects (_MNISTDataset, train_indices, val_indices, test_indices): the last three are  list of lists,
        containing the indices of the corresponding features fetched randomly for each class
    """

    # Getting and preparing MNIST dataset
    _mnist_train_val_data = datasets.MNIST(
        root="data",  # root folder where we're going to store the images
        train=True,  # loading the training set
        download=True,  # downloading the training set, if "False" obtaining the test set (10000 examples)
        transform=transforms.Compose([transforms.ToTensor()])
    )

    _mnist_test_data = datasets.MNIST(root="data", train=False, download=True,
                                      transform=transforms.Compose([transforms.ToTensor()]))

    # Merge the two datasets to create a larger one
    _MNISTDataset = ConcatDataset([_mnist_train_val_data, _mnist_test_data])

    # splitting dataset
    train_indices, val_indices, test_indices = create_splits(_MNISTDataset, list_split)
    return _MNISTDataset, train_indices, val_indices, test_indices


def plot_single_feature(dataset, feat_num):
    """Display a feature from the MNIST dataset.

    Args:
        dataset: MNIST dataset.
        feat_num: image to plot in the dataset.
    """

    if not isinstance(feat_num, int) and feat_num <= 0:
        raise ValueError("Invalid feat_num is provided (it must be a positive integer)")

    # Get a specific feature and label from the dataset and plot it
    feature = dataset[feat_num][0]  # get the feature (via [0]): it's a 3D tensor with 1 RGB channel
    label = dataset[feat_num][1]  # get the label (via [1])

    # Display the image using matplotlib, by accessing the first "channel" of the image (which is a 2D tensor)
    plt.imshow(feature[0], cmap='gray')  # use 'gray' colormap for grayscale images
    plt.title(f"Label: {label}")
    plt.show()  # Show the image


def undersampling(data_per_class, random_seed):
    """Balance imbalanced datasets by reducing the number of samples in the
     majority class to match the number of samples in the minority class

     Args:
        data_per_class: list of lists associated to each class containing indices of the data samples
        random_seed: random seed value

    Returns:
        undersampled_indices: list of lists that stores the indices of data samples after undersampling
        num_samples_to_keep: the number of samples of the minority class
    """

    # count the number of elements in each class
    element_counts = [len(sublist) for sublist in data_per_class]
    # print("\nNumber of elements for each class:", element_counts)

    # identify the minority class
    num_samples_to_keep = min(element_counts)
    minority_class = element_counts.index(num_samples_to_keep)

    # perform undersampling
    undersampled_indices = []

    for label, indices in enumerate(data_per_class):
        if label == minority_class:
            undersampled_indices.append(indices)
        else:
            if random_seed is not None:
                random.seed(random_seed)  # set the random seed if provided

            # shuffle and select a subset of samples from other classes
            random.shuffle(indices)
            undersampled_indices.append(indices[:num_samples_to_keep])
    return undersampled_indices, num_samples_to_keep


def create_splits(dataset, proportions: list):
    """Create 3 splits from the current dataset.

    Args:
        dataset: reference dataset.
        proportions: list with the fractions of data to store in each split (they must sum to 1).

    Returns:
        3 objects (train_indices, val_indices, test_indices): each of them is a list of lists, containing the indices of
        the corresponding features fetched randomly for each class
    """

    # checking argument
    p = 0
    invalid_prop_found = False
    for prop in proportions:
        if prop <= 0.0:
            invalid_prop_found = True
            break
        p += prop
    if p != 1.0 or invalid_prop_found or len(proportions) == 0:
        raise ValueError("Invalid proportions were provided (they must be positive and sum to 1)")

    # dividing data with respect to the main classes
    data_per_class = []

    for j in range(0, 10):  # j is the index of the main class
        data_per_class.append([])

    for i in range(0, len(dataset)):  # iterate over all the data samples in the dataset
        data_per_class[dataset[i][1]].append(i)  # It creates a list of lists where each sub-list contains the indices
        # of data samples belonging to a specific class

    random_seed = 42  # for reproducibility of the shuffled indices, otherwise comment it

    undersampled_indices_per_class, number_of_data_per_class = undersampling(data_per_class, random_seed)
    num_splits = len(proportions)

    # initialize the start index
    start = 0  # index of the first element to consider

    # Initialize the training,validation and test lists of indices
    train_indices = []
    val_indices = []
    test_indices = []

    for i in range(0, num_splits):
        p = proportions[i]
        n = int(p * number_of_data_per_class)  # number of element to consider for each class
        end = start + n if i < num_splits - 1 else number_of_data_per_class  # last (excluded) element to consider

        # Append the indices to the respective split list
        if i == 0:
            train_indices = [sublist[start:end] for sublist in undersampled_indices_per_class]
        elif i == 1:
            val_indices = [sublist[start:end] for sublist in undersampled_indices_per_class]
        elif i == 2:
            test_indices = [sublist[start:end] for sublist in undersampled_indices_per_class]

        # update the start index for the next split
        start = end

    return train_indices, val_indices, test_indices


def plot_accuracy(min_batch_train_acc, accs_val, backbone):
    """Plot the accuracy on the mini batch training data and the validation accuracy at each epoch

    Args:
        min_batch_train_acc: list with the training accuracies.
        accs_val: list with the validation accuracies computed at each epoch
        backbone:  architectures/classifiers used for training

    """

    plt.figure()
    plt.plot(min_batch_train_acc, label='Training Data')
    plt.plot(accs_val, label='Validation Data')
    plt.ylabel('Accuracy %')
    plt.xlabel('Mini-batches')
    plt.ylim((5, 102))
    plt.legend(loc='lower right')
    plt.savefig(backbone+'_training_stage.pdf')

