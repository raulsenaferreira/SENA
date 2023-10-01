import os

import numpy as np
import requests
import tarfile

import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import TensorDataset

from Params.params_dataset import *


class Dataset:
    """
    Load and configure a dataset. The valid values for the arguments can be found in Params/params_dataset.py.
    The first time a specific dataset is called, it is downloaded and saved locally.

    Args:
        name (str): Name of the dataset.
        split (str): Split to load ("train" or "test").
        network (str): Network that will be used to process the dataset.
        additional_transform (str or None): transform applied to images before going through the network.
        adversarial_attack (str or None): attack applied to images before going through the network.
        batch_size (int): Batch size used to process the dataset.

    Attributes (public):
        name (str): Name.
        split (str): Split.
        network (str): Network.
        additional_transform (str or None): Transform.
        adversarial_attack (str or None): Attack.
        batch_size (int): Batch size.
        dataloader (torch dataloader): Dataloader object used by the neural network to process the dataset
    """

    def __init__(self, name, split, network, additional_transform=None, adversarial_attack=None, batch_size=1000):
        """Initializes dataset."""

        # Public attributes
        self.name = name
        self.split = split
        self.network = network
        self.batch_size = batch_size
        self.additional_transform = additional_transform
        self.adversarial_attack = adversarial_attack

        # Create Data folder if required
        if not os.path.exists(datasets_path):
            os.makedirs(datasets_path)

        # Ensure dataset is valid
        self._check_accepted_dataset()
        self._check_accepted_network()
        self._check_accepted_transforms()
        self._check_accepted_attack()

        # Private attributes
        self._mean_transform = mean_transform[network]
        self._std_transform = std_transform[network]
        self._set_transforms()

        # Create dataloader
        self._load_dataset()
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size,
                                                      shuffle=False, num_workers=2)

    def _check_accepted_dataset(self):
        """Ensures that the queried dataset is valid."""
        data_split = self.name + "_" + self.split
        if data_split not in accepted_datasets:
            raise ValueError("Accepted dataset/split pairs are: %s" % str(accepted_datasets)[1:-1])

    def _check_accepted_network(self):
        """Ensures that the queried neural network is valid."""
        accepted_networks = list(mean_transform.keys())
        if self.network not in accepted_networks:
            raise ValueError("Accepted network architectures are: %s" % str(accepted_networks)[1:-1])

    def _check_accepted_transforms(self):
        """Ensures that the queried transform is valid."""
        accepted_transforms = list(additional_transforms.keys()) + [None]
        if self.additional_transform not in accepted_transforms:
            raise ValueError("Accepted data transforms are: %s" % str(accepted_transforms)[1:-1])

    def _check_accepted_attack(self):
        """Ensures that the queried adversarial attack is valid."""
        if self.adversarial_attack not in accepted_attacks + [None]:
            raise ValueError("Accepted attacks are: %s" % str(accepted_attacks)[1:-1])

    def _set_transforms(self):
        """Sets the transforms."""
        transform_list = [transforms.ToTensor(),
                          transforms.Normalize(self._mean_transform, self._std_transform)]
        if self.additional_transform is not None:
            transform_list.insert(0, additional_transforms[self.additional_transform])

        self._transform = transforms.Compose(transform_list)

    def _load_dataset(self):
        """Loads the dataset."""
        if self.name == "cifar10":
            self._load_cifar10()
        elif self.name == "cifar100":
            self._load_cifar100()
        elif self.name == "svhn":
            self._load_svhn()
        elif self.name == "imagenet":
            self._load_imagenet()
        elif self.name == "tiny_imagenet":
            self._load_tinyimagenet()
        elif self.name == "lsun":
            self._load_lsun()
        elif self.name == "mnist":
            self._load_mnist()
        elif self.name == "fashion_mnist":
            self._load_fashion_mnist()
        elif self.name == "emnist":
            self._load_emnist()
        elif self.name == "asl_mnist":
            self._load_asl_mnist()
        elif self.name == "simpsons_mnist":
            self._load_simpsons_mnist()
        elif self.name == "gtsrb":
            self._load_gtsrb()
        elif self.name == "fractal":
            self._load_fractal()
        else:
            pass

    def _load_mnist(self):
        """Load MNIST."""
        is_train = self.split == "train"
        mnist_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        self.dataset = torchvision.datasets.MNIST(root=datasets_path, train=is_train,
                                                  download=True, transform=mnist_transform)

    def _load_fashion_mnist(self):
        """Load Fashion MNIST."""
        is_train = self.split == "train"
        mnist_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        self.dataset = torchvision.datasets.FashionMNIST(root=datasets_path, train=is_train,
                                                         download=True, transform=mnist_transform)

    def _load_emnist(self):
        """Load E-MNIST."""
        is_train = self.split == "train"
        mnist_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        self.dataset = torchvision.datasets.EMNIST(root=datasets_path, train=is_train, split="letters",
                                                   download=True, transform=mnist_transform)

    def _load_asl_mnist(self):
        mnist_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)),
                                              transforms.Grayscale(num_output_channels=1)
                                              ])
        '''
        # execute this just once to convert the csv data to png
        import pandas as pd
        train_data = pd.read_csv(path_asl_mnist + 'sign_mnist_test.csv')
        images_csv = train_data.iloc[:, 1:].to_numpy()
        targets_csv = train_data['label'].to_numpy()
        channels, batch, size = 1, np.shape(images_csv)[0], np.shape(images_csv)[1]
        images_csv = np.reshape(images_csv, (batch, 28, 28))
        print(np.shape(images_csv), np.shape(targets_csv))
        i = 0

        for img in images_csv:
            PIL_image = Image.fromarray(np.uint8(img * 255), 'L')
            PIL_image.save(path_asl_mnist + 'sign_mnist_test/{}.jpg'.format(i))
            i += 1
        '''

        self.dataset = torchvision.datasets.ImageFolder(path_asl_mnist, transform=mnist_transform)

    def _load_cifar10(self):
        """Load CIFAR10."""
        is_train = self.split == "train"
        self.dataset = torchvision.datasets.CIFAR10(root=datasets_path, train=is_train,
                                                    download=True, transform=self._transform)

    def _load_cifar100(self):
        """Load CIFAR100."""
        is_train = self.split == "train"
        self.dataset = torchvision.datasets.CIFAR100(root=datasets_path, train=is_train,
                                                     download=True, transform=self._transform)

    def _load_svhn(self):
        """Load SVHN."""
        self.dataset = torchvision.datasets.SVHN(root=datasets_path, split=self.split,
                                                 download=True, transform=self._transform)

    def _load_imagenet(self):
        """Load Imagenet."""
        if self.split == 'test':
            self.split = 'val'
        self.dataset = torchvision.datasets.ImageNet(root=datasets_path, split=self.split,
                                                     download=True, transform=self._transform)

    def _load_tinyimagenet(self):
        """Load Tiny ImageNet."""
        '''
        if not os.path.exists(path_tinyImagenet):
            r = requests.get(downloadUrl_tinyImagenet, allow_redirects=True)
            open(path_tinyImagenet[:-1] + ".tar.gz", 'wb').write(r.content)
            tar = tarfile.open(path_tinyImagenet[:-1] + ".tar.gz")
            tar.extractall(path=datasets_path)
            tar.close()
        '''
        self.dataset = torchvision.datasets.ImageFolder(path_tinyImagenet,
                                                        transform=self._transform)

    def _load_lsun(self):
        """Load LSUN."""
        '''
        if not os.path.exists(path_lsun):
            r = requests.get(downloadUrl_lsun, allow_redirects=True)
            open(path_lsun[:-1] + ".tar.gz", 'wb').write(r.content)
            tar = tarfile.open(path_lsun[:-1] + ".tar.gz", 'r:')
            tar.extractall(path=datasets_path)
            tar.close()
        '''
        path_lsun_folder = "Data/LSUN_resize2/"
        self.dataset = torchvision.datasets.ImageFolder(path_lsun_folder,  # path_lsun
                                                        transform=self._transform)

        # self.dataset = torchvision.datasets.LSUN(root=datasets_path, classes='test_lmdb',
        #                                         transform=self._transform)

    def _load_fractal(self):
        fractal_transform = transforms.Compose([self._transform, transforms.Resize((32, 32))])
        self.dataset = torchvision.datasets.ImageFolder(path_fractal, transform=fractal_transform)

    def _load_gtsrb(self):
        """Load GTSRB."""
        gtsrb_transform = transforms.Compose([self._transform, transforms.Resize((32, 32))])
        self.dataset = torchvision.datasets.GTSRB(root=datasets_path, split="train",
                                                  download=True, transform=gtsrb_transform)

    def _load_simpsons_mnist(self):
        # https://github.com/alexattia/SimpsonRecognition
        mnist_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)),
                                              transforms.Grayscale(num_output_channels=1)
                                              ])
        self.dataset = torchvision.datasets.ImageFolder(path_simpsons_mnist, transform=mnist_transform)
