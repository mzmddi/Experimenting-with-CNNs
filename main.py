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
import torch.nn as nn
# ---------------


# genlogger.info("This is a test logging message")

# ---INPUT-PrePROCESSING-----

genlogger.debug("Starting to process the input...")

train_img_dir = "training_datasets/ShanghaiTech/part_B/train_data/images"
train_gt_dir  = "training_datasets/ShanghaiTech/part_B/train_data/ground_truth"

train_dataset = TrainingDataset(train_img_dir, train_gt_dir)
genlogger.debug("Train_dataset resolved")
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
genlogger.debug("train_loader resolved")

counting = 0
for images, counts in train_loader:
    print("Batch images:", images.shape)
    counting += 1
    if counting == 1:
        break


device = torch.device("mps" if torch.mps.is_available() else "cpu")
print(f"Device configured to : {device}")

# ---MODEL-CREATION-----

genlogger.debug("Starting to create the model...")

"""
This section will follow the model creation listed in the history_of_models.txt file.
Once the model is created is created and trained, a new model is created, so the previous one will be overwritten in the code.
This means that this section is constantly being overwritten, but a copy is kept in the txt file above.
"""

model = nn.Sequential(
    # convolution component
    # convolution block #1
    nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    
    ## neural network section
    nn.Flatten(),
    
    nn.Linear(32*128*128,128),
    nn.ReLU(),
    nn.Linear(128, 1)
)
genlogger.debug("Model created")

print(f"\nHere is the architecture of the model created: \n{model}")

# ---TRAINING-THE-MODEL---

genlogger.debug("Starting training the model...")

model  = model.to(device=device)
genlogger.debug(f"Model set to run on device ({device})")

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

epochs = 6
print(f"Numner of epochs defined: {epochs}")

size_of_train = len(train_loader)


for e in range(epochs):
    running_loss = 0.0
    counter = 0
    print("\n")
    for img, count in train_loader:
        img, count = img.to(device), count.to(device)
        
        optimizer.zero_grad()
        outputs = model(img)
        loss = criterion(outputs, count)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        counter += 1
        
        print(f"Epoch {e+1} in progress: {(counter/size_of_train)*100:.1f}%", end="\r")
        
        
    
    print(f"\nEpoch {e+1} out of {epochs} DONE, Loss: {running_loss/len(train_loader):.4f}")
    
# ---SAVING-THE-MODEL----
# genlogger.debug("Saving the model...")

# torch.save(model.state_dict(), "models/first_model.pth")
# print("\nModel Saved!")
# genlogger.debug(f"Model saved!")

# ---TESTING-THE-MODEL----
genlogger.debug("Starting Testing the model...")

test_img_dir = "training_datasets/ShanghaiTech/part_B/test_data/images"
test_gt_dir  = "training_datasets/ShanghaiTech/part_B/test_data/ground_truth"

test_dataset = TrainingDataset(train_img_dir, train_gt_dir)
genlogger.debug("Test_dataset resolved")
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)
genlogger.debug("test_loader resolved")

model.eval()

num_samples = len(test_loader)

with torch.no_grad():
    total_mae = 0
    total_mse = 0
    
    counter = 0
    
    print("\n")
    
    for img, count in test_loader:
        img, count = img.to(device), count.to(device)
        
        outputs = model(img)
        
        abs_error = torch.abs(outputs - count)
        sq_error = (outputs - count) ** 2
        
        total_mae += abs_error.sum().item()
        total_mse += sq_error.sum().item()
        
        counter += 1
        
        print(f"Testing the model => {(counter/num_samples)*100:.1f}%", end="\r")
        
    mae = total_mae / num_samples
    rmse = (total_mse / num_samples) ** 0.5
    
print(f"\nTest MAE: {mae:.2f}, RMSE: {rmse:.2f}\n")
    
    
    


