import logging.config
import os
from typing import List, Optional, Callable
import time

import PIL
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from datasets import *
from time import perf_counter

logger = logging.getLogger()

class ImageDataset(Dataset):
    def __init__(self, data, transform=None, cls_list=None, data_dir=None,
                 preload=False, device=None, transform_on_gpu=False):
        inputs,gt = data
        self.images, self.labels = [], []
        for x,y in zip(inputs,gt):
            self.images.append(x)
            self.labels.append(y)
        # self.dataset = dataset
        self.transform = transform
        self.cls_list = cls_list
        self.data_dir = data_dir
        self.preload = preload
        self.device = device
        self.transform_on_gpu = transform_on_gpu

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]
    
class StreamDataset(Dataset):
    def __init__(self, sample, transform :Optional[Callable]=None, cls_list=None):

        self.images     = []
        self.labels     = []
        self.cls_list   = cls_list
        self.transform  = transform

        for _, (image, label) in enumerate(sample):
            for img in image:
                self.images.append(img)
            for lbl in label:
                self.labels.append(self.cls_list.index(lbl.item()))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        sample  = dict()
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image   = self.images[idx]
        label   = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        sample["image"] = image
        sample["label"] = label
        return sample

    @torch.no_grad()
    def get_data(self):
        data = dict()
        images = []
        labels = []
        for i, image in enumerate(self.images):
            image = transforms.ToPILImage()(image)
            images.append(self.transform(image))
            labels.append(self.labels[i])
        data['image'] = torch.stack(images)
        data['label'] = torch.LongTensor(labels)
        return data

class MemoryDataset(Dataset):
    def __init__(self, transform=None, test_transform=None, cls_list=None, save_test=None, keep_history=False):
        
        self.datalist = []
        self.labels = []
        self.images = []
        
        self.transform = transform
        self.cls_list = cls_list
        self.cls_dict = {cls_list[i]:i for i in range(len(cls_list))}
        self.cls_count = []
        self.cls_idx = []
        self.cls_train_cnt = np.array([])
        self.score = []
        self.others_loss_decrease = np.array([])
        self.previous_idx = np.array([], dtype=int)
        self.test_transform = test_transform
        self.keep_history = keep_history

        self.save_test = save_test
        if self.save_test is not None:
            self.device_img = []

    def __len__(self):
        return len(self.images)

    def add_new_class(self, cls_list):
        self.cls_list = cls_list
        self.cls_count.append(0)
        self.cls_idx.append([])
        self.cls_dict = {self.cls_list[i]:i for i in range(len(self.cls_list))}
        self.cls_train_cnt = np.append(self.cls_train_cnt, 0)

    def __getitem__(self, idx):
        sample = dict()
        if torch.is_tensor(idx):
            idx = idx.value()
        label = self.labels[idx]
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)
        sample["image"] = image
        sample["label"] = label
        return sample

    def update_gss_score(self, score, idx=None):
        if idx is None:
            self.score.append(score)
        else:
            self.score[idx] = score

    def replace_sample(self, sample, idx=None):
        x, y = sample
        y = y.item()
        self.cls_count[self.cls_dict[y]] += 1

        if idx is None:
            self.cls_idx[self.cls_dict[y]].append(len(self.images))
            self.datalist.append({'image':x,'label':self.cls_dict[y]})
            self.images.append(x)
            self.labels.append(self.cls_dict[y])
            if self.save_test:
                self.device_img.append(self.test_transform(transforms.ToPILImage()(x)).unsqueeze(0))
            if self.cls_count[self.cls_dict[y]] == 1:
                self.others_loss_decrease = np.append(self.others_loss_decrease, 0)
            else:
                self.others_loss_decrease = np.append(self.others_loss_decrease, np.mean(self.others_loss_decrease[self.cls_idx[self.cls_dict[y]][:-1]]))
        else:
            self.cls_count[self.labels[idx]] -= 1
            self.cls_idx[self.labels[idx]].remove(idx)
            self.datalist[idx] = {'image':x,'label':self.cls_dict[y]}
            self.cls_idx[self.cls_dict[y]].append(idx)
            self.images[idx] = x
            self.labels[idx] = self.cls_dict[y]
            if self.save_test:
                self.device_img[idx] = self.test_transform(transforms.ToPILImage()(x)).unsqueeze(0)
            if self.cls_count[self.cls_dict[y]] == 1:
                self.others_loss_decrease[idx] = np.mean(self.others_loss_decrease)
            else:
                self.others_loss_decrease[idx] = np.mean(self.others_loss_decrease[self.cls_idx[self.cls_dict[y]][:-1]])

    def get_weight(self):
        weight = np.zeros(len(self.images))
        for i, indices in enumerate(self.cls_idx):
            weight[indices] = 1/self.cls_count[i]
        return weight

    @torch.no_grad()
    def get_batch(self, batch_size, use_weight=False, transform=None):
        if use_weight:
            weight = self.get_weight()
            indices = np.random.choice(range(len(self.images)), size=batch_size, p=weight/np.sum(weight), replace=False)
        else:
            indices = np.random.choice(range(len(self.images)), size=batch_size, replace=False)
        data = dict()
        images = []
        labels = []
        for i in indices:
            if transform is None:
                images.append(self.transform(transforms.ToPILImage()(self.images[i])))
            else:
                images.append(transform(transforms.ToPILImage()(self.images[i])))
            labels.append(self.labels[i])
            self.cls_train_cnt[self.labels[i]] += 1
        data['image'] = torch.stack(images)
        data['label'] = torch.LongTensor(labels)
        if self.keep_history:
            self.previous_idx = np.append(self.previous_idx, indices)
        return data

    def update_loss_history(self, loss, prev_loss, ema_ratio=0.90, dropped_idx=None):
        if dropped_idx is None:
            loss_diff = np.mean(loss - prev_loss)
        elif len(prev_loss) > 0:
            mask = np.ones(len(loss), bool)
            mask[dropped_idx] = False
            loss_diff = np.mean((loss[:len(prev_loss)] - prev_loss)[mask[:len(prev_loss)]])
        else:
            loss_diff = 0
        difference = loss_diff - np.mean(self.others_loss_decrease[self.previous_idx]) / len(self.previous_idx)
        self.others_loss_decrease[self.previous_idx] -= (1 - ema_ratio) * difference
        self.previous_idx = np.array([], dtype=int)

    def get_two_batches(self, batch_size, test_transform):
        indices = np.random.choice(range(len(self.images)), size=batch_size, replace=False)
        data_1 = dict()
        data_2 = dict()
        images = []
        labels = []
        for i in indices:
            if self.transform_on_gpu:
                images.append(self.transform_gpu(self.images[i].to(self.device)))
            else:
                images.append(self.transform(self.images[i]))
            labels.append(self.labels[i])
        data_1['image'] = torch.stack(images)
        data_1['label'] = torch.LongTensor(labels)
        images = []
        labels = []
        for i in indices:
            images.append(self.test_transform(self.images[i]))
            labels.append(self.labels[i])
        data_2['image'] = torch.stack(images)
        data_2['label'] = torch.LongTensor(labels)
        return data_1, data_2

    def make_cls_dist_set(self, labels, transform=None):
        if transform is None:
            transform = self.transform
        indices = []
        for label in labels:
            indices.append(np.random.choice(self.cls_idx[label]))
        indices = np.array(indices)
        data = dict()
        images = []
        labels = []
        for i in indices:
            images.append(transform(self.images[i]))
            labels.append(self.labels[i])
        data['image'] = torch.stack(images)
        data['label'] = torch.LongTensor(labels)
        return data

    def make_val_set(self, size=None, transform=None):
        if size is None:
            size = int(0.1*len(self.images))
        if transform is None:
            transform = self.transform
        size_per_cls = size//len(self.cls_list)
        indices = []
        for cls_list in self.cls_idx:
            if len(cls_list) >= size_per_cls:
                indices.append(np.random.choice(cls_list, size=size_per_cls, replace=False))
            else:
                indices.append(np.random.choice(cls_list, size=size_per_cls, replace=True))
        indices = np.concatenate(indices)
        data = dict()
        images = []
        labels = []
        for i in indices:
            images.append(transform(self.images[i]))
            labels.append(self.labels[i])
        data['image'] = torch.stack(images)
        data['label'] = torch.LongTensor(labels)
        return data

    def is_balanced(self):
        mem_per_cls = len(self.images)//len(self.cls_list)
        for cls in self.cls_count:
            if cls < mem_per_cls or cls > mem_per_cls+1:
                return False
        return True


def get_train_datalist(dataset, n_tasks, m, n, rnd_seed, cur_iter: int) -> List:
    if n == 100 or m == 0:
        n = 100
        m = 0
    return pd.read_json(
        f"collections/{dataset}/{dataset}_split{n_tasks}_n{n}_m{m}_rand{rnd_seed}_task{cur_iter}.json"
    ).to_dict(orient="records")

def get_test_datalist(dataset) -> List:
    return pd.read_json(f"collections/{dataset}/{dataset}_val.json").to_dict(orient="records")


def get_statistics(dataset: str):
    """
    Returns statistics of the dataset given a string of dataset name. To add new dataset, please add required statistics here
    """
    if dataset == 'imagenet':
        dataset = 'imagenet1000'
    assert dataset in [
        "mnist",
        "FashionMNIST",
        "SVHN",
        "cifar10",
        "cifar100",
        "CINIC10",
        "imagenet100",
        "imagenet1000",
        "tinyimagenet",
        "imagenet-r",
        "core50", 
        "imagenet_sketch", 
    ]
    mean = {
        "mnist": (0.1307,),
        "FashionMNIST": (0.1307,),
        "SVHN": (0.4377, 0.4438, 0.4728),
        "cifar10": (0.4914, 0.4822, 0.4465),
        "cifar100": (0.5071, 0.4867, 0.4408),
        "CINIC10": (0.47889522, 0.47227842, 0.43047404),
        "tinyimagenet": (0.4802, 0.4481, 0.3975),
        "imagenet100": (0.485, 0.456, 0.406),
        "imagenet1000": (0.485, 0.456, 0.406),
        "imagenet-r": (0.485, 0.456, 0.406),
        "core50": (0.6003, 0.5684, 0.5414), 
        "imagenet_sketch": (0.485, 0.456, 0.406), 
    }

    std = {
        "mnist": (0.3081,),
        "FashionMNIST": (0.3081,),
        "SVHN": (0.1969, 0.1999, 0.1958),
        "cifar10": (0.2023, 0.1994, 0.2010),
        "cifar100": (0.2675, 0.2565, 0.2761),
        "CINIC10": (0.24205776, 0.23828046, 0.25874835),
        "tinyimagenet": (0.2302, 0.2265, 0.2262),
        "imagenet100": (0.229, 0.224, 0.225),
        "imagenet1000": (0.229, 0.224, 0.225),
        "imagenet-r": (0.229, 0.224, 0.225),
        "core50": (0.1785, 0.1912, 0.2008), 
        "imagenet_sketch": (0.229, 0.224, 0.225),
    }

    classes = {
        "mnist": 10,
        "FashionMNIST": 10,
        "SVHN": 10,
        "cifar10": 10,
        "cifar100": 100,
        "CINIC10": 10,
        "tinyimagenet": 200,
        "imagenet100": 100,
        "imagenet1000": 1000,
        "imagenet-r": 200,
        "core50": 50, 
        "imagenet_sketch": 1000, 
    }

    in_channels = {
        "mnist": 1,
        "FashionMNIST": 1,
        "SVHN": 3,
        "cifar10": 3,
        "cifar100": 3,
        "CINIC10": 3,
        "tinyimagenet": 3,
        "imagenet100": 3,
        "imagenet1000": 3,
        "imagenet-r": 3,
        "core50": 3, 
        "imagenet_sketch": 3, 
    }

    inp_size = {
        "mnist": 28,
        "FashionMNIST": 28,
        "SVHN": 32,
        "cifar10": 32,
        "cifar100": 32,
        "CINIC10": 32,
        "tinyimagenet": 64,
        "imagenet100": 224,
        "imagenet1000": 224,
        "imagenet-r": 224,
        "core50": 128, 
        "imagenet_sketch": 224, 
    }
    return (
        mean[dataset],
        std[dataset],
        classes[dataset],
        inp_size[dataset],
        in_channels[dataset],
    )