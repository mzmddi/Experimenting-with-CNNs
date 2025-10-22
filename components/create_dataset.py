# ---NOTES---
"""
input params: 
    - batch_size=2: int
return:
    - train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader
"""
# ---IMPORTS---
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
from torch.utils.data import random_split
# ---CODE---

def get_dataset(batch_size=16):
    
    transform = transforms.Compose([
        transforms.Resize((512,512)),
        transforms.ToTensor(),
        transforms.Normalize(
        mean=[0.4883, 0.4553, 0.4170],
        std=[0.2276, 0.2230, 0.2232]
    )
    ])
    # this is the transformed used for the dataset
    # resizes all images to 512x512
    # toTensor()
    # normalises with those values
    # the mean and std values are found by calculating them before hand
    
    dataset = datasets.ImageFolder(root="./training_dataset", transform=transform)
    
    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)
    val_size = int(0.1 * dataset_size)
    test_size = dataset_size - train_size - val_size
    # creating the sizes of each set of data
    
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    # using random_split(), the original dataset is split into 3 dinstinct sets
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    # creates a loader for each one of the datasets
    
    return train_loader, val_loader, test_loader