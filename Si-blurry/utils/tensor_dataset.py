import torch
from torch.utils.data import Dataset

class TensorDataset(Dataset):
    def __init__(self, data, label, transform=None):
        self.data = data
        self.label = label
        self.transform = transform
        self.length = len(data)
        self.classes = torch.unique(label).tolist()
        self.targets = label.tolist()
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.label[index]
        if self.transform:
            x = self.transform(x)
        return x, y, index
    
    def __len__(self):
        return self.length