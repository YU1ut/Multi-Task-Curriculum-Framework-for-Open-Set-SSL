import numpy as np
from PIL import Image
import os

import torchvision
import torch

class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2

def get_cifar10(root, n_labeled, out_dataset, start_label=0,
                 transform_train=None, transform_val=None,
                 download=True):

    base_dataset = torchvision.datasets.CIFAR10(root, train=True, download=download)
    train_labeled_idxs, train_unlabeled_idxs, val_idxs = train_val_split(base_dataset.targets, int(n_labeled/10))

    train_labeled_dataset = CIFAR10_labeled(root, train_labeled_idxs, train=True, transform=transform_train)

    train_unlabeled_dataset = CIFAR10_unlabeled(root, train_unlabeled_idxs, out_path=f'./data/{out_dataset}', train=True, transform=TransformTwice(transform_train))
    train_dataset = CIFAR10_cocnat(root, train_labeled_idxs, train_unlabeled_idxs, start_label, out_path=f'./data/{out_dataset}', train=True, transform=transform_train)

    val_dataset = CIFAR10_labeled(root, val_idxs, train=True, transform=transform_val, download=True)
    test_dataset = CIFAR10_labeled(root, train=False, transform=transform_val, download=True)

    print (f"#Labeled: {len(train_labeled_idxs)} #Unlabeled: {len(train_unlabeled_idxs)} #OOD: {len(train_unlabeled_dataset.outpath)} #Val: {len(val_idxs)}")

    return train_labeled_dataset, train_unlabeled_dataset, train_dataset, val_dataset, test_dataset

def train_val_split(labels, n_labeled_per_class):
    labels = np.array(labels)
    train_labeled_idxs = []
    train_unlabeled_idxs = []
    val_idxs = []

    for i in range(10):
        idxs = np.where(labels == i)[0]
        np.random.shuffle(idxs)
        train_labeled_idxs.extend(idxs[:n_labeled_per_class])
        train_unlabeled_idxs.extend(idxs[n_labeled_per_class:-500])
        val_idxs.extend(idxs[-500:])
    np.random.shuffle(train_labeled_idxs)
    np.random.shuffle(train_unlabeled_idxs)
    np.random.shuffle(val_idxs)

    return train_labeled_idxs, train_unlabeled_idxs, val_idxs

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)

def pad(x, border=4):
    return np.pad(x, [(0,0), (border, border), (border, border), (0, 0)])

def normalise(x, mean=cifar10_mean, std=cifar10_std):
    x, mean, std = [np.array(a, np.float32) for a in (x, mean, std)]
    x -= mean*255
    x *= 1.0/(255*std)
    return x

def transpose(x, source='NHWC', target='NCHW'):
    return x.transpose([source.index(d) for d in target]) 

class CIFAR10_labeled(torchvision.datasets.CIFAR10):

    def __init__(self, root, indexs=None, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(CIFAR10_labeled, self).__init__(root, train=train,
                 transform=transform, target_transform=target_transform,
                 download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
        self.data = transpose(normalise(pad(self.data)))

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class CIFAR10_unlabeled(CIFAR10_labeled):

    def __init__(self, root, indexs, out_path, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(CIFAR10_unlabeled, self).__init__(root, indexs, train=train,
                 transform=transform, target_transform=target_transform,
                 download=download)
        if out_path == './data/none':
             self.outpath = []
        else:
            self.outpath = np.load(out_path+'.npy')
            self.outpath = transpose(normalise(pad(self.outpath)))
        self.targets = np.array(self.targets.tolist() + [-1]*len(self.outpath))

    def __len__(self):
        return len(self.data) + len(self.outpath)

    def __getitem__(self, index):
        if index < len(self.data):
            img, target = self.data[index], self.targets[index]
        else:
            img, target = self.outpath[index - len(self.data)], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class CIFAR10_cocnat(torchvision.datasets.CIFAR10):

    def __init__(self, root, labeled_indexs, unlabeled_indexs, start_label, out_path, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(CIFAR10_cocnat, self).__init__(root, train=train,
                 transform=transform, target_transform=target_transform,
                 download=download)

        self.data_x = self.data[labeled_indexs]
        self.data_x = transpose(normalise(pad(self.data_x)))
        self.targets_x = np.array(self.targets)[labeled_indexs]

        self.data_u = self.data[unlabeled_indexs]
        self.data_u = transpose(normalise(pad(self.data_u)))
        self.targets_u = np.array(self.targets)[unlabeled_indexs]
        
        if out_path == './data/none':
             self.outpath = []
        else:
            self.outpath = np.load(out_path+'.npy')
            self.outpath = transpose(normalise(pad(self.outpath)))

        self.soft_labels = np.zeros((len(self.data_x)+len(self.data_u)+len(self.outpath)), dtype=np.float32)
        for idx in range(len(self.data_x)+len(self.data_u)+len(self.outpath)):
            if idx < len(self.data_x):
                self.soft_labels[idx] = 1.0
            else:
                self.soft_labels[idx] = start_label
        self.prediction = np.zeros((len(self.data_x)+len(self.data_u)+len(self.outpath), 10), dtype=np.float32)
        self.prediction[:len(self.data_x),:] = 1.0
        self.count = 0

    def __len__(self):
        return len(self.data_x) + len(self.data_u) + len(self.outpath)

    def label_update(self, results):
        self.count += 1

        # While updating the noisy label y_i by the probability s, we used the average output probability of the network of the past 10 epochs as s.
        idx = (self.count - 1) % 10
        self.prediction[len(self.data_x):, idx] = results[len(self.data_x):]

        if self.count >= 10:
            self.soft_labels = self.prediction.mean(axis=1)

    def __getitem__(self, index):
        if index < len(self.data_x):
            img, target = self.data_x[index], self.targets_x[index]

        elif index < len(self.data_x) + len(self.data_u):
            img, target = self.data_u[index-len(self.data_x)], self.targets_u[index-len(self.data_x)]

        else:
            img, target = self.outpath[index - len(self.data_x) - len(self.data_u)], -1

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, self.soft_labels[index], index