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
import json
import matplotlib.pyplot as plt
from model_blueprint import build_model
# ---------------

# ---Different-Variations---

with open("permutation_values.json", "r") as json_file:
    
    f = json.load(json_file)

epochs = f[0]["epochs"]
batch_size = f[0]["batch_size"]
num_of_conv_layers = f[0]["num_of_conv_layers"]
num_of_neural_layers = f[0]["num_of_neural_layers"]
# hyperparameters

conv_start_channel_variations = f[0]["conv_start_channel_variations"]
# start channel will be coded here, but the rest will be x2 every layer
kernel_size = f[0]["kernel_size"]
pool_size = f[0]["pool_size"]
    
for epoch in epochs:
    # while this is a for loop, at the end of this loop, 
    #   the epoch list will be updated by removing the value from the list, 
    #   and the new epoch list will be updated in the json,
    #   and the program will stop with exit(0).
    #   This way, the models will be process by epochs, since i am scared
    #   that my computer will explode from all the models.
    #   When i restart the program, it wont start from 0, but rather from the next epoch value
    
    for b in batch_size:
    
        # ---INPUT-PrePROCESSING-----

        train_img_dir = "training_datasets/ShanghaiTech/part_B/train_data/images"
        train_gt_dir  = "training_datasets/ShanghaiTech/part_B/train_data/ground_truth"

        train_dataset = TrainingDataset(train_img_dir, train_gt_dir)
        genlogger.debug("Train_dataset resolved")
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=b, shuffle=False)
        genlogger.debug("train_loader resolved")

        device = torch.device("mps" if torch.mps.is_available() else "cpu")
        print(f"Device configured to : {device}")
        
        # counting = 0
        # for images, counts in train_loader:
        #     print(f"\n[{counting}] Batch images shape: {images.shape}")
        #     print(f"[{counting}] Batch counts shape: {counts.shape}")
        #     print(f"[{counting}] Counts values:{counts}\n")
        #     counting += 1
        #     if counting == 1:
        #         break

        # ---MODEL-CREATION-----

        """
        This section will follow the model creation listed in the history_of_models.txt file.
        Once the model is created is created and trained, a new model is created, so the previous one will be overwritten in the code.
        This means that this section is constantly being overwritten, but a copy is kept in the txt file above.
        """
        counting = 0
        for conv_layer in num_of_conv_layers:
            for lin_layer in num_of_neural_layers:
                for conv_start in conv_start_channel_variations:
                    
                    conv_in_channels = []
                    conv_out_channels = []
                    
                    for i in range(conv_layer):
                        
                        if i == 0:
                            
                            conv_in_channels.append(3)
                            conv_out_channels.append(conv_start)
                            continue
                        
                        conv_in_channels.append(conv_out_channels[i-1])
                        conv_out_channels.append(conv_in_channels[i]*2)

                    for k_size in kernel_size:
                        for p_size in pool_size:
                                
                            model = build_model(num_of_conv_layer=conv_layer,
                                                num_of_neural_layer=lin_layer,
                                                conv_in_channels=conv_in_channels,
                                                conv_out_channels=conv_out_channels,
                                                kernel_size=k_size,
                                                pool_size=p_size,
                                                )


                            print(f"Here is the architecture of the model created: \n{model}")

                            # ---TRAINING-THE-MODEL---

                            model  = model.to(device=device)
                            genlogger.debug(f"Model set to run on device ({device})")

                            criterion = nn.MSELoss()
                            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
                            
                            print(f"Numner of epochs defined: {epoch}")

                            size_of_train = len(train_loader)

                            for e in range(epoch):
                                running_loss = 0.0
                                counter = 0
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

                            # ---TESTING-THE-MODEL----
                            genlogger.debug("Starting Testing the model...")

                            test_img_dir = "training_datasets/ShanghaiTech/part_B/test_data/images"
                            test_gt_dir  = "training_datasets/ShanghaiTech/part_B/test_data/ground_truth"

                            test_dataset = TrainingDataset(test_img_dir, test_gt_dir)
                            genlogger.debug("Test_dataset resolved")
                            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=b, shuffle=False)
                            genlogger.debug("test_loader resolved")

                            model.eval()

                            num_samples = len(test_loader)

                            with torch.no_grad():
                                total_mae = 0
                                total_mse = 0
                                
                                counter = 0
                                
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
                            
                            # ---ADDING-THE-STATS---

                            """
                            Structure of components in list of stats (json file)
                            model = {
                                "model_name":"",
                                "num_of_epochs":"",
                                "mae":"",
                                "rmse":""
                                "num_of_layers":"",
                                "layers": {
                                    layer_1:"",
                                    layer_2:"",
                                    # etc etc etc
                                }
                            }
                            """
                            num_of_layers = len(model)
                            layers = {}
                            for i in range(len(model)):
                                layers[f"layer_{i+1}"] = f"{model[i]}"
                                
                            model_dict = {}

                            with open("stats.json", "r") as f:
                                stats = json.load(f)

                            num_reached = len(stats)

                            model_dict["model_name"] = f"model_{num_reached}"
                            model_dict["num_of_epochs"] = f"{epochs}"
                            model_dict["mae"] = f"{round(mae)}"
                            model_dict["rmse"] = f"{round(rmse)}"
                            model_dict["num_of_layers"] = f"{num_of_layers}"
                            model_dict["layers"] = layers

                            stats.append(model_dict)

                            with open("stats.json", "w") as f:
                                json.dump(stats, f, indent=4)
                                
                            print("New stats saved in the stats.json file!")

                            del model
                            torch.cuda.empty_cache()
                            
    epochs.remove(epoch)
    
    with open("permutation_values.json", "w") as json_file:
        f[0]["epochs"] = epochs
        json.dump(f, json_file, indent=2)
        
    exit()

# ---CREATING-DIAGRAM----
"""
Plotting the data gathered as of now
I will plot different graphs to see how changing a factor changes the outcome
list of different plotting:
    - mae vs all models
    - mae vs epochs in groups of the same identical model
    - mae vs number of conv layers grouped in same epochs and layer densities
    - mae vs number of neural layers grouped in same conv layers and epochs
    - mae vs density of conv layer grouped in identical structure and epochs
total of 5 graphs
"""

# data_1 = []
# model_name = []
# mae = []

# for m in stats:
    
#     this_mae = m["mae"]
#     this_model_name = m["model_name"]
    
#     data_1.append((this_model_name, this_mae))

# data_1 = sorted(data_1, key=lambda t: t[1])

# for e in data_1:
#     model_name.append(e[0])
#     mae.append(e[1])
    
# plt.scatter(model_name, mae)

# plt.title("MAE value for ALL")
# plt.xlabel("Model Names")
# plt.ylabel("MAE")

# plt.savefig("mae_all.pdf")

# plt.show()
# 
# 
# ---SAVING-THE-MODEL----
# genlogger.debug("Saving the model...")

# torch.save(model.state_dict(), "models/first_model.pth")
# print("\nModel Saved!")
# genlogger.debug(f"Model saved!")