import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define the path to your dataset
dataset_path = '/home/xw6956/l2p-pytorch/datasets/core50_128x128'

# Define the transformation. No normalization is applied here as we are calculating the mean and std.
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to a common size
    transforms.ToTensor(),  # Convert images to tensors
])

# Load the dataset
dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

# Create a DataLoader
dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

# Function to calculate mean and std
def calculate_mean_std(loader):
    mean = 0.
    std = 0.
    total_images_count = 0
    for images, _ in loader:
        batch_samples = images.size(0)  # Batch size (the last batch can have smaller size!)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += batch_samples

    mean /= total_images_count
    std /= total_images_count

    return mean, std

# Calculate mean and std
mean, std = calculate_mean_std(dataloader)

print(f'Mean: {mean}')
print(f'Std: {std}')
