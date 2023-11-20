import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import models, transforms, datasets
from torch.utils.data import Dataset
from torch.utils.data.dataset import Subset
from PIL import Image
import argparse
import random
from classifier import Classifier
import matplotlib.pyplot as plt
import Backbones
import Adv_attack_my_func as MyFunc
from autoattack import AutoAttack
import numpy as np


class DatasetFromSubset(Dataset):
    """Class that models a generic dataset from a Subset object of MNIST images and targets"""

    def __init__(self, subset_obj, custom_operation=None):
        """Create a dataset.

        Args:
            subset_obj: subset object created after splitting MNIST dataset
            custom_operation: preprocessing operation
        """
        # dataset attributes
        self.subset_obj = subset_obj
        self.preprocess = custom_operation
        self.length = None

        list_indices = self.get_indices_and_shuffle()

        # dictionary where the values are the original indices and the keys are their positions (enumerated order)
        self.index_dict = {i: list_indices[i] for i in range(len(list_indices))}

    def __len__(self):
        """The total number of examples in this dataset."""
        self.length = self.subset_obj.__len__()*len(self.subset_obj.indices[0])
        return self.length

    def get_indices_and_shuffle(self):
        """Return the list of indices, shuffled, of the original MNIST examples present in MY_Dataset """

        list_of_indices = []
        for classes in self.subset_obj.indices:
            for index in classes:
                list_of_indices.append(index)

        # At this moment, list_of_indices is a sequence of indices fetched in an ordered manner from the subset
        #  => since I have an ordered sequence of features per each class, I shuffle them

        rand_seed = 0
        random.seed(rand_seed)
        random.shuffle(list_of_indices)

        return list_of_indices

    def __getitem__(self, index):
        if index > self.__len__():
            raise IndexError("list index out of range")
        new_index = self.index_dict[index]
        feature = self.subset_obj.dataset[new_index][0]  # fetch the feature
        feature = transforms.ToPILImage()(feature)  # Convert the feature tensor to PIL Image for preprocessing purposes
        label = self.subset_obj.dataset[new_index][1]  # fetch the label
        if self.preprocess is not None:
            feature = self.preprocess(feature)  # applying the preprocess operation on the single element
        else:
            transform = transforms.Compose([
                transforms.ToTensor()
            ])
            feature = transform(feature)
        return feature, label


class AdversarialDataset(Dataset):
    """Class that models an adversarial dataset from tensor objects of MNIST images and targets"""

    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        feature = self.features[index]
        label = self.labels[index]
        return feature, label


def parse_command_line_arguments():
    """Parse command line arguments, checking their values."""

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('mode', choices=['train', 'eval', 'adv_attacks_generation', 'eval_other_adv_attacks'],
                        help='train, evaluate the classifier or generate adversarial attacks; '
                             'also evaluate the robustness on other adversarial datasets')
    parser.add_argument('--splits', type=str, default='0.7-0.15-0.15',
                        help='fraction of data to be used in train, val, test set (default: 0.7-0.15-0.15)')
    parser.add_argument('--backbone', type=str, default='simplecnn', choices=['resnet18', 'simplecnn', 'mlp1', 'mlp2'],
                        help='backbone network for feature extraction (default: simplecnn)"')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='mini-batch size (default: 64)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of training epochs (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (Adam) (default: 0.001)')
    parser.add_argument('--device', default='cpu', type=str,
                        help='device to be used for computations (in {cpu, cuda:0, cuda:1, ...}, default: cpu)')

    parsed_arguments = parser.parse_args()

    # converting split fraction string to a list of floating point values ('0.7-0.15-0.15' => [0.7, 0.15, 0.15])
    splits_string = str(parsed_arguments.splits)
    fractions_string = splits_string.split('-')
    if len(fractions_string) != 3:
        raise ValueError("Invalid split fractions were provided. Required format (example): 0.7-0.15-0.15")
    else:
        splits = []
        frac_sum = 0.
        for fraction in fractions_string:
            try:
                splits.append(float(fraction))
                frac_sum += splits[-1]
            except ValueError:
                raise ValueError("Invalid split fractions were provided. Required format (example): 0.7-0.15-0.15")
        if frac_sum != 1.0:
            raise ValueError("Invalid split fractions were provided. They must sum to 1.")

    # updating the 'splits' argument
    parsed_arguments.splits = splits

    return parsed_arguments


def create_adversarial_examples(eps, args, _classifier):
    """Create and save adversarial examples for a specific architecture."""

    # Preparing the data:
    # loading MNIST and retrieving test indices to create test set
    _MNISTDataset, _, _, test_indices = MyFunc.load_and_split_mnist(args.splits)
    test_subset = Subset(_MNISTDataset, test_indices)  # instance of the test Subset class on MNIST dataset

    # generate dataset object from subset
    _test_set = DatasetFromSubset(test_subset)

    # Use AutoAttack for generating adversarial examples
    # initialize AutoAttack
    if args.backbone == 'simplecnn':
        adversary = AutoAttack(_classifier.net, norm='Linf', eps=eps, version='rand', device=args.device)
        batch_size = 100  # (customizable) it's very time-consuming considering all the original test set cardinality
    elif args.backbone == 'resnet18':
        adversary = AutoAttack(_classifier.net, norm='Linf', eps=eps, version='standard', device=args.device)
        batch_size = 1000  # (customizable)
    else:
        adversary = AutoAttack(_classifier.net, norm='Linf', eps=eps, version='standard', device=args.device)
        batch_size = int(len(_test_set))

    # create DataLoader object
    test_dl = DataLoader(_test_set, batch_size=batch_size, shuffle=None)

    i = 0
    with torch.no_grad():
        for X, Y in test_dl:

            if args.backbone == 'simplecnn' or args.backbone == 'resnet18':
                i += 1
                X = X.to(args.device)
                if i == 2:  # I create only a subset of adversarial examples for these classifiers
                    break

            # apply the standard evaluation, where the attacks are run sequentially on batches
            x_adv = adversary.run_standard_evaluation(X, Y, bs=batch_size)
            # , return_labels=True)  # uncomment to see the output (in general wrong) of the NN after perturbation
            # (see the 'autoattack' code in this case)

            # convert the tensors to a numpy arrays
            adv_numpy_array = x_adv.numpy()
            orig_label = Y.numpy()

        # save the numpy arrays
        np.save('adv_' + args.backbone + '_array.npy', adv_numpy_array)
        np.save('labels_' + args.backbone + '_array.npy', orig_label)


def plot_adversarial_example(backbone_item, index):

    # Display a generic feature from the adversarial dataset
    _features_loaded_array = np.load('adv_' + backbone_item + '_array.npy')
    _labels = np.load('labels_' + backbone_item + '_array.npy')

    # convert features array to tensor
    _adv_features_tensor = torch.from_numpy(_features_loaded_array)

    # create adversarial dataset for backbone architecture
    adv_dataset = AdversarialDataset(_adv_features_tensor, _labels)
    MyFunc.plot_single_feature(adv_dataset, index)


# entry point
if __name__ == "__main__":

    args = parse_command_line_arguments()  # command line parse routine

    for k, v in args.__dict__.items():  # for loop that prints the args to screen
        print(k + '=' + str(v))

    if args.mode == 'train':  # if the selected operational mode is "train" then the training routine starts

        # creating a new classifier
        _classifier = Classifier(args.backbone, args.device)  # I create an object of class Classifier

        # total_params = sum(p.numel() for p in _classifier.net.parameters())  # count the number of parameters
        # print(total_params)

        # loading MNIST and retrieving train, val and test indices to create training, val and test sets
        _MNISTDataset, train_indices, val_indices, test_indices = MyFunc.load_and_split_mnist(args.splits)

        # Display a generic feature from the MNIST dataset
        # MyFunc.plot_single_feature(_MNISTDataset, 63312)

        # create subset
        train_subset = Subset(_MNISTDataset, train_indices)  # instance of the training Subset class on MNIST dataset
        val_subset = Subset(_MNISTDataset, val_indices)  # instance of the validation MNIST...
        test_subset = Subset(_MNISTDataset, test_indices)  # instance of the test...

        # generate dataset object from subset
        _train_set = DatasetFromSubset(train_subset, _classifier.preprocess_train)
        _val_set = DatasetFromSubset(val_subset)
        _test_set = DatasetFromSubset(test_subset)

        # training the classifier
        _classifier.train_classifier(_train_set, _val_set, args.batch_size, args.lr, args.epochs, args.backbone)

        # loading the model that yielded the best results in the validation data (during the training epochs)
        print("Training complete, loading the best found model...")
        _classifier.load('classifier_'+args.backbone+'.pth')

        # computing the performance of the final model in the prepared data splits
        print("Evaluating the classifier...")
        _train_acc = _classifier.eval_classifier(_train_set, batch_size=args.batch_size)
        _val_acc = _classifier.eval_classifier(_val_set, batch_size=args.batch_size)
        _test_acc = _classifier.eval_classifier(_test_set, batch_size=args.batch_size)

        print("train set:\tacc={0:.2f}".format(_train_acc))
        print("val set:\tacc={0:.2f}".format(_val_acc))
        print("test set:\tacc={0:.2f}".format(_test_acc))

    elif args.mode == 'eval':
        print("Evaluating the classifier...")

        # creating a new classifier
        _classifier = Classifier(args.backbone, args.device)

        # loading the classifier
        _classifier.load('classifier_'+args.backbone+'.pth')

        # Preparing the data:
        # loading MNIST and retrieving test indices to create test set
        _MNISTDataset, _, _, test_indices = MyFunc.load_and_split_mnist(args.splits)
        test_subset = Subset(_MNISTDataset, test_indices)  # instance of the test Subset class on MNIST dataset

        # generate dataset object from subset
        _test_set = DatasetFromSubset(test_subset)

        # computing the classifier performance
        _acc = _classifier.eval_classifier(_test_set, batch_size=args.batch_size)

        print("acc={0:.2f}".format(_acc))

    elif args.mode == 'adv_attacks_generation':
        print("Generating the adversarial examples...")

        # creating a new classifier
        _classifier = Classifier(args.backbone, args.device)

        # loading the classifier
        _classifier.load('classifier_' + args.backbone + '.pth')

        epsilon = 0.3  # bound on the norm of the adversarial perturbations

        # Generate adversarial examples
        create_adversarial_examples(epsilon, args, _classifier)

    elif args.mode == 'eval_other_adv_attacks':
        print("Evaluating "+args.backbone+" architecture on other adversarial datasets")

        # creating a new classifier
        _classifier = Classifier(args.backbone, args.device)

        # loading the classifier
        _classifier.load('classifier_' + args.backbone + '.pth')

        # plot_adversarial_example('mlp1', 0)

        backbone_list = ['mlp1', 'mlp2', 'simplecnn', 'resnet18']

        for backbone in backbone_list:

            print('Accuracy of the', args.backbone, 'model on the', backbone, 'adversarial dataset:')

            # load the numpy arrays to a tensor
            features_loaded_array = np.load('adv_'+backbone+'_array.npy')
            labels = np.load('labels_'+backbone+'_array.npy')

            # convert features array to tensor
            adv_features_tensor = torch.from_numpy(features_loaded_array)

            # create adversarial dataset for backbone architecture
            adv_dataset_ = AdversarialDataset(adv_features_tensor, labels)

            acc = _classifier.eval_classifier(adv_dataset_, batch_size=len(adv_dataset_))
            print(acc)
            print("")















