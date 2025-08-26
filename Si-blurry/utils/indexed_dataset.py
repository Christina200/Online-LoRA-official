from torch.utils.data import Dataset

class IndexedDataset(Dataset):
    def __init__(self, dataset:Dataset) -> None:
        super().__init__()
        self.dataset = dataset
        self.classes = dataset.classes
        self.targets = dataset.targets
    def __getitem__(self, index):
        return *self.dataset.__getitem__(index), index
    def __len__(self):
        return len(self.dataset)