from typing import Callable, Optional

import torch
from torch.utils.data import Dataset, random_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms

class NotMNIST(Dataset):
    def __init__(self, 
                 root             : str, 
                 train            : bool, 
                 transform        : Optional[Callable] = None, 
                 target_transform : Optional[Callable] = None, 
                 download         : bool = False
                 ) -> None:
        super().__init__()
        self.dataset = ImageFolder(root + '/notMNIST_large/', transforms.ToTensor() if transform is None else transform, target_transform)
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
        image, label = self.dataset.__getitem__(index)
        return image.expand(3,-1,-1), label

    def __len__(self):
        return len(self.dataset)
