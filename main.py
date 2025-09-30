"""
./main.py
entry-way to perform the experimentations

// Setting up the virtual environment
    ** at the root directory **
    (1) python3 -m venv .venv
    (2) source .venv/bin/activate
    ** if (some or all) requirements are not installed yet **
    (3) pip3 install -r requirements.txt
"""

# ---IMPORTS-----
from py_log.logger import genlogger
import os
from torch.utils.data import Dataset
from torch.nn import Sequential
from dataset_loading import TrainingDataset
import torch
# ---------------


# genlogger.info("This is a test logging message")

# ---INPUT-PrePROCESSING-----

genlogger.debug("Starting -INPUT-PrePROCESSING- section")

train_img_dir = "training_datasets/ShanghaiTech/part_A/train_data/images"
train_gt_dir  = "training_datasets/ShanghaiTech/part_A/train_data/ground_truth"

train_dataset = TrainingDataset(train_img_dir, train_gt_dir)
genlogger.debug("Train_dataset resolved")
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
genlogger.debug("train_loader resolved")

count = 0
for images, counts in train_loader:
    print("Batch images:", images.shape)   # e.g. [16, 3, 256, 256]
    print("Batch counts:", counts)         # e.g. [16, 1]
    count += 1
    if count == 3:
        break

device = torch.device("mps" if torch.mps.is_available() else "cpu")
print("Using device:", device)

# ---MODEL-CREATION-----

