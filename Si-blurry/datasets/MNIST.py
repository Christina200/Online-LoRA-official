# Wraping for the SHVN dataset

from typing import Callable, Optional

from torch.utils.data import Dataset
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

class MNIST(Dataset):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:

        super().__init__()
        self.dataset = MNIST(root, train, transforms.ToTensor() if transform is None else transform, target_transform, download)
        
        self.classes = [str(i) for i in range(10)]
        self.targets = []
        for cls in self.dataset.labels:
            self.targets.append(int(cls))

    def __getitem__(self, index):
        image, label = self.dataset.__getitem__(index)
        return image.expand(3,-1,-1), label
        
    def __len__(self):
        return len(self.dataset)