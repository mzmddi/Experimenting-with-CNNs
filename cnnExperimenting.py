# ---NOTES---
"""
Entry point of the entire program
FOOTNOTES: 
    (1) [batch_size, num_conv_layers, conv_start, kernel_size, pool_size, num_neural_layers, neural_start]
"""

# ---IMPORTS---
import sys
import os
from components.permutations import permutate, get_first_permutation_list, update_json_file_list
from components.create_dataset import get_dataset
from components.model_blueprint import build_model
from components.training import train_this_model
from components.visualize import visualize_mae_vs_all_models, visualize_mae_v_epochs
from components.model_architecture import get_model_architecture
import torch
import torch.nn as nn
# ---CODE---

if __name__=="__main__":
    
    os.system("clear")
    
    if len(sys.argv) == 1 or "-train" in sys.argv:
        # default execution
        
        batch_size = 64
        
        train_loader, val_loader, test_loader = get_dataset(batch_size=batch_size)
        # getting all 3 datasets (Dataloader Class)
        
        print(f"length of train_loader: {len(train_loader)*batch_size}")
        print(f"length of val_loader: {len(val_loader)*batch_size}")
        print(f"length of test loader: {len(test_loader)*batch_size}")
        
        model_architecture = get_model_architecture()
        # getting the model's architecture values
        
        device = torch.device("mps" if torch.mps.is_available() else "cpu")
        # setting the device to run on mps (i have a macbook air m2, cuda isnt in my computer)
        print(f"Device configured to: {device}")
        
        model = build_model(num_conv_layers=int(model_architecture["num_of_conv_layer"]),
                            conv_start=int(model_architecture["starting_conv_channel"]),
                            kernel_size=int(model_architecture["conv_kernel_size"]),
                            pool_size=2,
                            num_neural_layers=int(model_architecture["num_of_neural_layer"]),
                            neural_start=int(model_architecture["starting_neural_channel"]))
        print("Model created!\n")
        print(model)
        
        model = model.to(device=device)
        print(f"Model set to use the {next(model.parameters()).device} device!")
        
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        # setting up the criterion and optimizer for the training
        # set the optimizer at 1e-4 learning rate
        
        # NEED TO APPLY torch.sigmoid(outputs) WHENEVER THE MODEL MAKES A PREDICTION
        # outputs = model(images)      # raw logits
        # preds = torch.sigmoid(outputs) > 0.5
        
        num_epochs = 10
        
        train_data_size = len(train_loader.dataset)
        val_data_size = len(val_loader.dataset)
        
        print("Starting Training!\n")
        
        for epoch in range(num_epochs):
            
            # print(f"Epoch {epoch+1}/{num_epochs} training in progress ... (not constantly printing to avoid I/O using resources)", end="\r")
            model.train()
            # set the model state to train just to be sure
            
            running_loss = 0.0
            correct = 0
            total = 0
            
            counter = 0
            
            for images, labels in train_loader:
                
                images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * images.size(0)
                
                preds = torch.sigmoid(outputs) > 0.5
                correct += (preds.float() == labels).sum().item()
                total += labels.size(0)
                
                # counter += 1
                
                # if (total/train_data_size)*100 % 10 == 0:
                    
                #     print(f"Epoch {epoch+1} training in progress: {(total/train_data_size)*100:.1f}%", end="\r")
            
            epoch_loss = running_loss / total
            epoch_acc = correct / total
            # accuracy
            print(f"Epoch {epoch+1}/{num_epochs} DONE, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc*100:.3f} %")
            
            # print(f"Epoch {epoch+1}/{num_epochs} validation in progress ... (not constantly printing to avoid I/O using resources)", end="\r")
            
            model.eval()
            
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                
                for images, labels in val_loader:
                    
                    images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
                    
                    outputs = model(images)
                    
                    preds = torch.sigmoid(outputs) > 0.5
                    
                    val_correct += (preds.float() == labels).sum().item()
                    
                    val_total += labels.size(0)
                    
                    # if (val_total/val_data_size)*100 % 10 == 0:
                        
                    #     print(f"Epoch {epoch+1} validation in progress: {(total/val_data_size)*100:.1f}%", end="\r")
                    
            
            val_acc = val_correct / val_total
            
            print(f"Validation Accuracy: {val_acc*100:.3f} %\n")
            
            
                
                
                
            
            
        exit()
        
        
        
        
        
        
        
        
        # below is the old version, with the other ShanghaiTech dataset. Not using that anymore
        
        # while True:
            
            # perm_list = get_first_permutation_list()
            # print(f"Permutation used for this iteration: {perm_list}")
            
            # batch_size = perm_list[0]
            # # see footnote (1) above for index definitions
            # print(f"Batch_size = {batch_size}")
            
            # train_dataset = get_dataset(batch_size=batch_size, mode="train")
            # print("Training dataset retrieved!")
            
            # device = torch.device("mps" if torch.mps.is_available() else "cpu")
            # print(f"Device configured to : {device}")
            
            # model = build_model(num_conv_layers=perm_list[1],
            #                     conv_start=perm_list[2],
            #                     kernel_size=perm_list[3],
            #                     pool_size=perm_list[4],
            #                     num_neural_layers=perm_list[5],
            #                     neural_start=perm_list[6])
            # print("Model created!\n")
            
            # model = model.to(device=device)
            # print("Model set to use the mps device!")
            
            # train_this_model(model, train_dataset, device, batch_size=batch_size, num_conv_layers=perm_list[1], num_neural_layers=perm_list[5])
            
            # update_json_file_list()
            
            # del model
            # torch.cuda.empty_cache()
            
            # print("Row loop completed! Model deleted from memory and torch cache emptied.\nRestarting loop with new index[0] list!")
            
            # exit()
            
    elif "-permutate" in sys.argv:
        permutate()
        pass
    
    elif "-visualize" in sys.argv:
        
        if len(sys.argv) == 2:
            # default -visualize flag
            # python3 main.py -visualize
            visualize_mae_vs_all_models()
            
        elif "-all" in sys.argv:
            pass
        
        elif "-epochs" in sys.argv:
            visualize_mae_v_epochs()
        
        elif "-convLayers" in sys.argv:
            pass
        
        elif "-neuLayers" in sys.argv:
            pass
        
        elif "-batchSize" in sys.argv:
            pass