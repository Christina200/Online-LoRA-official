# Multi datasets for continual learning
# All datasets needs to be in the same format.
# have targets and classes within the dataset.

from typing import Callable, Optional, Iterable
from torch.utils.data import Dataset

class multiDatasets(Dataset):
    def __init__(
        self,
        datasets: Iterable[Dataset],
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:

        super().__init__()
        self.datasets = []
        self.dataset_lengths = []
        self.classes = []

        for dataset in datasets:
            if not isinstance(dataset, Dataset):
                raise TypeError("dataset should be a Dataset object")
            self.datasets.append(dataset(root, train, transform, target_transform, download))
            self.dataset_lengths.append(len(self.datasets[-1]))
            self.classes += len(self.classes)

        self.classes = [str(i) for i in range(self.classes)]
        self.targets = []

        for i, dataset in enumerate(self.datasets):
            for cls in dataset.targets:
                self.targets.append(int(cls) + sum(self.classes[:i]))


    def __getitem__(self, index):
        target = self.targets[index]
        for i, dataset in enumerate(self.datasets):
            if index < self.dataset_lengths[i]:
                return dataset[index], target
            index -= self.dataset_lengths[i]

    def __len__(self):
        return len(self.targets)