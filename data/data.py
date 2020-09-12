import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import pandas as pd


def getDataset(dataset):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        ])

    if dataset == 'CIFAR10':
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        num_classes = 10
        inputs = 3

    elif dataset == 'CIFAR100':
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
        num_classes = 100
        inputs = 3
        
    elif dataset == 'MNIST':
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        num_classes = 10
        inputs = 1

    elif dataset == 'EEG':
        # Read Training Data
        train_data = pd.read_csv('./data/EEG/training_set.csv', header=None)
        train_data = np.array(train_data).astype('float32')
        [num_train_data, _] = np.shape(train_data)
        train_data = np.reshape(train_data, [num_train_data, 32, 20])
        train_data = train_data[:, np.newaxis, :, :]

        # Read Training Labels
        train_labels = pd.read_csv('./data/EEG/training_label.csv', header=None)
        train_labels = np.array(train_labels).astype('float32')
        train_labels = np.squeeze(train_labels)

        # Read Testing Data
        test_data = pd.read_csv('./data/EEG/test_set.csv', header=None)
        test_data = np.array(test_data).astype('float32')
        [num_test_data, _] = np.shape(test_data)
        test_data = np.reshape(test_data, [num_test_data, 32, 20])
        test_data = test_data[:, np.newaxis, :, :]

        # Read Testing Labels
        test_labels = pd.read_csv('./data/EEG/test_label.csv', header=None)
        test_labels = np.array(test_labels).astype('float32')
        test_labels = np.squeeze(test_labels)

        trainset = torch.utils.data.TensorDataset(torch.tensor(train_data), torch.tensor(train_labels))
        testset  = torch.utils.data.TensorDataset(torch.tensor(test_data), torch.tensor(test_labels))
        num_classes = 4
        inputs = 1

    return trainset, testset, inputs, num_classes


def getDataloader(trainset, testset, valid_size, batch_size, num_workers):
    num_train = len(trainset)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    # train_idx, valid_idx = indices[split:], indices[:split]
    train_idx, valid_idx = indices[:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
    # valid_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers)
    # test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=num_workers)
    test_loader  = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=num_workers)

    return train_loader, valid_loader, test_loader