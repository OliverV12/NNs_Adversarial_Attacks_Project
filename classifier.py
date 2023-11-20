import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import Backbones
from torch.utils.data import DataLoader
from torchvision import models, transforms
import Adv_attack_my_func as MyFunc


class Classifier:
    """Classifier that makes predictions on numbers (categories) of the MNIST database."""

    def __init__(self, backbone="simplecnn", device="cpu"):
        """Create an untrained classifier.

        Args:
            backbone: the string ("MLP" or "simplecnn") that indicates which backbone network will be used.
            device: the string ("cpu", "cuda:0", "cuda:1", ...) that indicates the device to use.
        """
        # class attributes
        self.num_outputs = 10  # in our problem we have 10 total classes [0-9]
        self.net = None  # the neural network: backbone + additional projection to the output space, initialized to None
        self.device = torch.device(device)  # the device (object) on which data will be moved
        self.preprocess_train = None  # image preprocessing operations (when training the classifier)
        self.preprocess_eval = None  # image preprocessing operations (when evaluating the classifier)

        # creating the network
        if backbone is not None and backbone == "resnet18":
            self.net = models.resnet18(weights=None)
            self.net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2)
            # self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=(3, 3), bias=False)

            # adding a new (learnable) final layer (transfer learning)
            self.net.fc = nn.Linear(512, self.num_outputs)

            # preprocessing operations on the input images
            self.preprocess_train = transforms.Compose([transforms.RandomRotation(45), transforms.ToTensor()])
            self.preprocess_eval = transforms.Compose([transforms.ToTensor()])

        # creating the network
        elif backbone is not None and backbone == "simplecnn":

            # case 1: SimpleCNN-based network
            simple_cnn = Backbones.SimpleCNN()  # create an instance of SimpleCNN class, importing Backbones module
            self.net = simple_cnn.net

            # preprocessing operations on the input images
            self.preprocess_train = simple_cnn.preprocess_train  # call preprocess_train attribute from Backbones module
            self.preprocess_eval = simple_cnn.preprocess_eval

        elif backbone is not None and backbone == "mlp1":

            # case 1: MLP-based network
            mlp = Backbones.MLP1()
            self.net = mlp.net

            # preprocessing operations on the input images
            self.preprocess_train = mlp.preprocess_train  # call preprocess_train attribute from Backbones module
            self.preprocess_eval = mlp.preprocess_eval

        elif backbone is not None and backbone == "mlp2":

            # case 1: MLP-based network
            mlp = Backbones.MLP2()
            self.net = mlp.net

            # preprocessing operations on the input images
            self.preprocess_train = mlp.preprocess_train  # call preprocess_train attribute from Backbones module
            self.preprocess_eval = mlp.preprocess_eval

        else:
            if backbone is not None:
                raise ValueError("Unknown backbone: " + str(backbone))
            else:
                raise ValueError("Specify a backbone network!")

        # moving the network to the right device memory
        self.net.to(self.device)

    def save(self, file_name):
        """Save the classifier (network)."""

        torch.save(self.net.state_dict(), file_name)

    def load(self, file_name):
        """Load the classifier (network)."""

        self.net.load_state_dict(torch.load(file_name, map_location=self.device))

    def forward(self, x):  # 'x' is a minibatch of images
        """Compute the output of the network."""

        logits = self.net(x)  # outputs before applying the activation function (activation scores)
        outputs = torch.nn.functional.softmax(logits, dim=1)  # to have probabilities we use the softmax function

        return outputs, logits  # 'outputs' is needed for performance, 'logits' for the loss

    @staticmethod
    def decision(outputs):  # decision function that tells which is the class
        """Given the tensor with the net outputs, compute the final decision of the classifier (class label).

        Args:
            outputs: the 2D tensor with the outputs of the net (each row is about an example).

        Returns:
            1D tensor with the main class IDs (for each example).
        """

        # the decision on main classes is given by the winning class (since they are mutually exclusive)
        main_class_ids = torch.argmax(outputs, dim=1)  # 'argmax' computes the unit with the strongest activation

        return main_class_ids  # class id (index of the winning class)

    def train_classifier(self, train_set, validation_set, batch_size, lr, epochs, backbone):

        # initializing some elements
        best_val_acc = -1.  # the best accuracy computed on the validation data (main classes)
        best_epoch = -1  # the epoch in which the best accuracy above was computed

        # ensuring the classifier is in 'train' mode (pytorch)
        self.net.train()

        # creating the optimizer
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.net.parameters()), lr)

        # Define DataLoader object on the training and validation set
        train_dl = DataLoader(train_set, batch_size=batch_size, shuffle=True)

        # Display parameters
        min_batch_accs_train = []
        accs_val = []  # list of accuracies (val_acc) computed at each epoch
        val_acc = 0

        for e in range(epochs):
            epoch_train_acc = 0.0
            epoch_train_loss = 0.0
            epoch_num_train_examples = 0
            i = 0  # count the mini-batches
            for X, Y in train_dl:  # loop on mini-batches

                i += 1

                # counting
                batch_num_train_examples = X.shape[0]  # mini-batch size
                epoch_num_train_examples += batch_num_train_examples

                # moving mini-batch data to the right device
                X = X.to(self.device)
                Y = Y.to(self.device)

                # computing the network output on the current mini-batch
                outputs, logits = self.forward(X)

                # computing the loss function (by passing the activation scores and the targets)
                loss = Classifier.__loss(logits, Y)

                # computing gradients and updating the network weights
                optimizer.zero_grad()  # zeroing the memory areas that were storing previously computed gradients
                loss.backward()  # computing gradients
                optimizer.step()  # updating weights

                # computing the performance of the net on the current training mini-batch
                with torch.no_grad():  # keeping these operations out of those for which we will compute the gradient
                    self.net.eval()  # switching to eval mode

                    # Compute performance
                    batch_train_acc = self.__performance(outputs, Y)
                    min_batch_accs_train.append(batch_train_acc)

                    # accumulating performance measures to get a final estimate on the whole training set
                    epoch_train_acc += batch_train_acc * batch_num_train_examples

                    # accumulating other stats
                    epoch_train_loss += loss.item() * batch_num_train_examples

                    self.net.train()

                    # Print mini-batch related stats
                    print(" mini-batch {0}:  loss={1:.4f}, tr_acc={2:.2f}".format(i, loss.item(), batch_train_acc))

                    if i == len(X):
                        continue

                    accs_val.append(val_acc)

            val_acc = self.eval_classifier(validation_set, batch_size)
            accs_val.append(val_acc)

            # Save the model if the validation accuracy increases
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = e + 1
                self.save("classifier_"+backbone+".pth")

            epoch_train_loss /= epoch_num_train_examples

            # printing (epoch related) stats on screen
            print(("epoch={0}/{1}:\tloss={2:.4f}, tr_acc={3:.2f}, val_acc={4:.2f}"
                   + (", BEST!" if best_epoch == e + 1 else ""))
                  .format(e + 1, epochs, epoch_train_loss,
                          epoch_train_acc / epoch_num_train_examples, val_acc))

        # display the accuracy on the mini-batch training data and validation accuracy
        MyFunc.plot_accuracy(min_batch_accs_train, accs_val, backbone)

    def eval_classifier(self, _data_set, batch_size):
        """Evaluate the classifier on the given data set."""

        # checking if the classifier is in 'eval' or 'train' mode (in the latter case, we have to switch state)
        training_mode_originally_on = self.net.training
        if training_mode_originally_on:
            self.net.eval()  # enforcing evaluation mode

        data_dl = DataLoader(_data_set, batch_size=batch_size, shuffle=False)

        # lists to accumulate network outputs and labels
        cpu_batch_outputs = []
        cpu_batch_labels = []

        with torch.no_grad():  # keeping off the autograd engine
            for X, Y in data_dl:

                X = X.to(self.device)

                # Computing the network output on the current mini-batch
                outputs, _ = self.forward(X)
                cpu_batch_outputs.append(outputs.cpu())
                cpu_batch_labels.append(Y)

        # Concatenating network outputs and labels
        all_outputs = torch.cat(cpu_batch_outputs, dim=0)
        all_labels = torch.cat(cpu_batch_labels, dim=0)

        # Computing the performance of the net on the whole dataset
        acc = self.__performance(all_outputs, all_labels)

        if training_mode_originally_on:
            self.net.train()  # restoring the training state, if needed

        return acc

    @staticmethod
    def __loss(logits, labels):
        """Compute the loss function of the classifier.

        Args:
            logits: the (partial) outcome of the forward operation.
            labels: 1D tensor with the class labels.

        Returns:
            The value of the loss function.
        """

        tot_loss = F.cross_entropy(logits, labels, reduction="mean")
        return tot_loss

    def __performance(self, outputs, labels):
        """Compute the accuracy in predicting the main classes.

        Args:
            outputs: the 2D tensor with the network outputs for a batch of samples (one example per row).
            labels: the 1D tensor with the expected labels.

        Returns:
            The accuracy in predicting the main classes.
        """

        # taking a decision
        main_class_ids = self.decision(outputs)

        # computing the accuracy on main classes
        right_predictions_on_main_classes = torch.eq(main_class_ids, labels)
        acc_main_classes = torch.mean(right_predictions_on_main_classes.to(torch.float) * 100.0).item()

        return acc_main_classes
