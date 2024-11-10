from typing import Callable, Optional

import os
import torch
from torch.utils.data import Dataset, random_split
from torchvision.datasets.utils import download_url
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms

# CUB200 dataset class
# Download code from https://github.com/JH-LEE-KR/ContinualDatasets/blob/main/continual_datasets/continual_datasets.py
# by JH-LEE-KR

class CUB200(Dataset):
    def __init__(self, 
                 root             : str, 
                 train            : bool, 
                 transform        : Optional[Callable] = None, 
                 target_transform : Optional[Callable] = None, 
                 download         : bool = False
                 ) -> None:
        super().__init__()

        self.root = os.path.expanduser(root)
        self.url = 'https://data.deepai.org/CUB200(2011).zip'
        self.filename = 'CUB200(2011).zip'

        fpath = os.path.join(self.root, self.filename)
        if not os.path.isfile(fpath):
            if not download:
                raise RuntimeError('Dataset not found. You can use download=True to download it')
            else:
                print('Downloading from '+self.url)
                download_url(self.url, self.root, filename=self.filename)
        if not os.path.exists(os.path.join(self.root, 'CUB_200_2011')):
            import zipfile
            zip_ref = zipfile.ZipFile(fpath, 'r')
            zip_ref.extractall(self.root)
            zip_ref.close()
            import tarfile
            tar_ref = tarfile.open(os.path.join(self.root, 'CUB_200_2011.tgz'), 'r')
            tar_ref.extractall(self.root)
            tar_ref.close()
    
        self.dataset = ImageFolder(self.root + '/CUB200-2011/images', transforms.ToTensor() if transform is None else transform, target_transform)
        len_train    = int(len(self.dataset) * 0.8)
        len_val      = len(self.dataset) - len_train
        train, test  = random_split(self.dataset, [len_train, len_val], generator=torch.Generator().manual_seed(42))
        self.dataset = train if train else test
        self.classes = self.dataset.dataset.classes
        self.targets = []
        for i in self.dataset.indices:
            self.targets.append(self.dataset.dataset.targets[i])
        pass
    
    def __getitem__(self, index):
        return self.dataset.__getitem__(index)

    def __len__(self):
        return len(self.dataset)
