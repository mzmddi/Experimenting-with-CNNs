# ---NOTES---
# ---IMPORTS---
import torch.nn as nn
import torch
from components.create_dataset import get_dataset
from pathlib import Path
import json
# ---CODE---

def train_this_model(model, train_dataset, device, batch_size=2):
    
    filepath = Path("./supporting_files/stats.json")
    if not filepath.exists():
        filepath.touch()
        with open(filepath, "w") as json_file:
            dump = []
            json.dump(dump, json_file, indent=4)
    
    total_epochs = 250
                          
    for epoch in range(total_epochs):
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
                
        if epoch % 20 == 0 and epoch != 0:
            # every 25 epochs, we pause training, switch to .eval() mode, test the model with test data, 
            #       record metrics, and return back to .train() mode for next epoch
            
            test_dataset = get_dataset(batch_size=batch_size, mode="test")
            
            model.eval()
            
            num_samples = len(test_dataset)
            
            with torch.no_grad():
                total_mae = 0
                total_mse = 0

                counter = 0

                for img, count in test_dataset:
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

            print(f"\nTest MAE: {mae:.2f}, RMSE: {rmse:.2f}")
            
            num_of_layers = len(model)
            layers = {}
            for i in range(len(model)):
                layers[f"layer_{i+1}"] = f"{model[i]}"
                
            model_dict = {}
            
            with open(filepath, "r") as f:
                stats = json.load(f)
                
            num_reached = len(stats)
            
            model_dict["model_name"] = f"model_{num_reached}"
            model_dict["num_of_epochs"] = f"{epoch}"
            model_dict["mae"] = f"{round(mae)}"
            model_dict["rmse"] = f"{round(rmse)}"
            model_dict["num_of_layers"] = f"{num_of_layers}"
            model_dict["layers"] = layers
            
            stats.append(model_dict)
            
            with open(filepath, "w") as f:
                json.dump(stats, f, indent=4)
                
            print("New stats saved in the stats.json file!")
            
            model.train()
        
        running_loss = 0.0
        counter = 0
        
        for img, count in train_dataset:
            
            img, count = img.to(device), count.to(device)
            optimizer.zero_grad()
            outputs = model(img)
            loss = criterion(outputs, count)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            counter += 1
            
            print(f"Epoch {epoch+1} in progress: {(counter/(len(train_dataset)))*100:.1f}%", end="\r")
            
        print(f"\nEpoch {epoch+1} out of {total_epochs} DONE, Loss: {running_loss/len(train_dataset):.4f}")