from typing import Callable, Optional
import os

from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_url
import torchvision.transforms as transforms

# TinyImageNet dataset class
# Download code from https://github.com/JH-LEE-KR/ContinualDatasets/blob/main/continual_datasets/continual_datasets.py
# by JH-LEE-KR

class TinyImageNet(ImageFolder):
    def __init__(self, 
                 root             : str, 
                 train            : bool, 
                 transform        : Optional[Callable] = None, 
                 target_transform : Optional[Callable] = None, 
                 download         : bool = False
                 ) -> None:
        
        self.root = os.path.expanduser(root)
        self.url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
        self.filename = 'tiny-imagenet-200.zip'

        fpath = os.path.join(self.root, self.filename)
        if not os.path.isfile(fpath):
            if not download:
               raise RuntimeError('Dataset not found. You can use download=True to download it')
            else:
                print('Downloading from '+ self.url)
                download_url(self.url, self.root, filename=self.filename)
        if not os.path.exists(os.path.join(self.root, 'tiny-imagenet-200')):
            import zipfile
            zip_ref = zipfile.ZipFile(fpath, 'r')
            zip_ref.extractall(os.path.join(self.root))
            zip_ref.close()
            # self.split()

        self.path = self.root + '/tiny-imagenet-200/'
        if train:
            super().__init__(self.path + "train", transform=transforms.ToTensor() if transform is None else transform, target_transform=target_transform)
            self.classes = []
            with open(self.path + "wnids.txt", 'r') as f:
                for id in f.readlines():
                    self.classes.append(id.split("\n")[0])
            self.class_to_idx = {clss: idx for idx, clss in enumerate(self.classes)}
            self.targets = []
            for idx, (path, _) in enumerate(self.samples):
                self.samples[idx] = (path, self.class_to_idx[path.split("/")[-3]])
                self.targets.append(self.class_to_idx[path.split("/")[-3]])

        else:
            super().__init__(self.path + "val", transform=transforms.ToTensor() if transform is None else transform, target_transform=target_transform)
            self.classes = []
            with open(self.path + "wnids.txt", 'r') as f:
                for id in f.readlines():
                    self.classes.append(id.split("\n")[0])
            self.class_to_idx = {clss: idx for idx, clss in enumerate(self.classes)}
            self.targets = []
            with open(self.path + "val/val_annotations.txt", 'r') as f:
                file_to_idx = {line.split('\t')[0] : self.class_to_idx[line.split('\t')[1]] for line in f.readlines()}
                for idx, (path, _) in enumerate(self.samples):
                    self.samples[idx] = (path, file_to_idx[path.split("/")[-1]])
                    self.targets.append(file_to_idx[path.split("/")[-1]])
