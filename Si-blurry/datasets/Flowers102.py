# Wraping for the SHVN dataset

from typing import Callable, Optional

from torch.utils.data import Dataset
from torchvision.datasets import Flowers102
from torchvision.transforms import transforms

class Flowers102(Dataset):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:

        super().__init__()
        self.dataset = Flowers102(root, "train" if train else "test", transforms.ToTensor() if transform is None else transform, target_transform, download)
        
        self.classes = [str(i) for i in range(102)]
        self.targets = []
        for cls in self.dataset._labels:
            self.targets.append(int(cls))

    def __getitem__(self, index):
        image, label = self.dataset.__getitem__(index)
        return image.mean(dim=0, keepdim=True).expand(3,-1,-1), label

    def __len__(self):
        return len(self.dataset)