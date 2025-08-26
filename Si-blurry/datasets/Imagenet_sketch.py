from typing import Callable, Optional
import os

import torch
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_url
import torchvision.transforms as transforms

class Imagenet_Sketch(ImageFolder):
    def __init__(self, 
                 root             : str, 
                 train            : bool, 
                 transform        : Optional[Callable] = None, 
                 target_transform : Optional[Callable] = None, 
                 download         : bool = False
                 ) -> None:
        
        self.root = os.path.expanduser(root)
        self.url = "https://drive.google.com/file/d/1Mj0i5HBthqH1p_yeXzsg22gZduvgoNeA/view"
        self.filename = 'ImageNet-Sketch.zip'

        fpath = os.path.join(self.root, self.filename)
        if not os.path.isfile(fpath):
            if not download:
               raise RuntimeError('Dataset not found. You can use download=True to download it')
            else:
                print('Downloading from '+ self.url)
                download_url(self.url, self.root, filename=self.filename)
        if not os.path.exists(os.path.join(self.root, 'imagenet-s')):
            import tarfile
            tar = tarfile.open(fpath, 'r')
            tar.extractall(os.path.join(self.root))
            tar.close()

        self.path = self.root + '/imagenet-s/'
        super().__init__(self.path, transform=transforms.Compose([transforms.Resize(256), transforms.RandomCrop(224)]) if transform is None else transforms.Compose([transforms.Resize(256), transforms.RandomCrop(224),transform]), target_transform=target_transform)
        generator = torch.Generator().manual_seed(0)
        len_train = int(len(self.samples) * 0.8)
        len_test = len(self.samples) - len_train
        self.train_sample = torch.randperm(len(self.samples), generator=generator)
        self.test_sample = self.train_sample[len_train:].sort().values.tolist()
        self.train_sample = self.train_sample[:len_train].sort().values.tolist()

        if train:
            self.classes = [i for i in range(1000)]
            self.class_to_idx = [i for i in range(1000)]
            samples = []
            for idx in self.train_sample:
                samples.append(self.samples[idx])
            self.targets = [s[1] for s in samples]
            self.samples = samples

        else:
            self.classes = [i for i in range(1000)]
            self.class_to_idx = [i for i in range(1000)]
            samples = []
            for idx in self.test_sample:
                samples.append(self.samples[idx])
            self.targets = [s[1] for s in samples]
            self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)