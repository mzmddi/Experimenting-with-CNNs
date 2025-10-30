# ---NOTES---
"""
./cnnExperimenting.py
entry-way to perform the experimentations

// Setting up the virtual environment
    ** at the root directory **
    (1) python3 -m venv .venv
    (2) source .venv/bin/activate
    ** if (some or all) requirements are not installed yet **
    (3) pip3 install -r requirements.txt
"""

# ---IMPORTS---
import sys
import os
from components.create_dataset import get_dataset
from components.model_blueprint import build_model
from components.model_architecture import get_model_architecture
from components.record_data import Recorder
import torch
import torch.nn as nn
# ---CODE---

if __name__=="__main__":
    
    os.system("clear")
    
    r = Recorder()
    
    if len(sys.argv) == 1 or "-train" in sys.argv:
        # default execution
        
        batch_size = 64
        
        train_loader, val_loader, test_loader = get_dataset(batch_size=batch_size)
        # getting all 3 datasets (Dataloader Class)
        
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
        print("Model created!")
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
        
        num_epochs = 3
        
        train_data_size = len(train_loader.dataset)
        val_data_size = len(val_loader.dataset)
        
        print("--- Starting Training! ---")
        
        layers = {}
        num_of_layers = len(model)
        for i in range(len(model)):
            layers[f"layer_{i}"] = f"{model[i]}"
        
        r.record_metadata({
            "batch_size": str(batch_size),
            "total_epoch_number": str(num_epochs),
            "total_num_layers": str(num_of_layers),
            "total_conv_layer": str(model_architecture["num_of_conv_layer"]),
            "total_neural_layer": str(model_architecture["num_of_neural_layer"]),
            "layers": layers
        })
        
        for epoch in range(num_epochs):
            
            r.record_timing("training_start")
            
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
                
            epoch_loss = running_loss / total
            epoch_acc = correct / total
            # accuracy
            print(f"Epoch {epoch+1}/{num_epochs} DONE, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc*100:.2f}")
            
            model.eval()
            
            val_correct = 0
            val_total = 0
            
            dog_class = 1
            dog_true_pos = 0
            dog_false_pos = 0
            dog_false_neg = 0
            # variable used for precision, recall, and f1 centered around the dog class
            
            cat_class = 0
            cat_true_pos = 0
            cat_false_pos = 0
            cat_false_neg = 0
            # variable used for precision, recall, and f1 centered around the cat class
            
            with torch.no_grad():
                
                for images, labels in val_loader:
                    
                    images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
                    
                    outputs = model(images)
                    
                    preds = torch.sigmoid(outputs) > 0.5
                    
                    val_correct += (preds.float() == labels).sum().item()
                    
                    val_total += labels.size(0)
                
                    dog_true_pos += ((preds == dog_class) & (labels == dog_class)).sum().item()
                    dog_false_pos += ((preds == dog_class) & (labels != dog_class)).sum().item()
                    dog_false_neg += ((preds != dog_class) & (labels == dog_class)).sum().item()
                    
                    cat_true_pos += ((preds == cat_class) & (labels == cat_class)).sum().item()
                    cat_false_pos += ((preds == cat_class) & (labels != cat_class)).sum().item()
                    cat_false_neg += ((preds != cat_class) & (labels == cat_class)).sum().item()
            
            val_acc = val_correct / val_total
            
            precision_dog = dog_true_pos / (dog_true_pos + dog_false_pos) if (dog_true_pos + dog_false_pos) > 0 else 0
            recall_dog = dog_true_pos / (dog_true_pos + dog_false_neg) if (dog_true_pos + dog_false_neg) > 0 else 0
            f1_dog = 2 * (precision_dog * recall_dog) / (precision_dog + recall_dog) if (precision_dog + recall_dog) > 0 else 0
            
            precision_cat = cat_true_pos / (cat_true_pos + cat_false_pos) if (cat_true_pos + cat_false_pos) > 0 else 0
            recall_cat = cat_true_pos / (cat_true_pos + cat_false_neg) if (cat_true_pos + cat_false_neg) > 0 else 0
            f1_cat = 2 * (precision_cat * recall_cat) / (precision_cat + recall_cat) if (precision_cat + recall_cat) > 0 else 0
            
            print(f"Validation Accuracy: {val_acc*100:.2f}")
            print(f"Dog -> Precision: {precision_dog*100:.2f}, Recall: {recall_dog*100:.2f}, F1: {f1_dog*100:.2f}")
            print(f"Cat -> Precision: {precision_cat*100:.2f}, Recall: {recall_cat*100:.2f}, F1: {f1_cat*100:.2f}")
            print(15*"-")
            
            val_metrics = {
                "accuracy": f"{val_acc*100:.2f}",
                "dog_precision": f"{precision_dog*100:.2f}",
                "dog_recall": f"{recall_dog*100:.2f}",
                "dog_f1": f"{f1_dog*100:.2f}",
                "cat_precision": f"{precision_cat*100:.2f}",
                "cat_recall": f"{recall_cat*100:.2f}",
                "cat_f1": f"{f1_cat*100:.2f}"
            }
            
            r.record_epoch(epoch, val_metrics)
                
        r.record_timing("training_end")
        print('--- Starting Testing! ---')
        model.eval()
        with torch.no_grad():
            
            test_dog_class = 1
            test_dog_true_pos = 0
            test_dog_false_pos = 0
            test_dog_false_neg = 0
            
            test_cat_class = 0
            test_cat_true_pos = 0
            test_cat_false_pos = 0
            test_cat_false_neg = 0
            
            test_correct = 0
            test_total = 0
            
            for test_images, test_labels, in test_loader:
                
                test_images, test_labels = test_images.to(device), test_labels.to(device).float().unsqueeze(1)
                test_outputs = model(test_images)
                test_preds = torch.sigmoid(test_outputs) > 0.5
                
                test_correct += (test_preds.float() == test_labels).sum().item()
                test_total += test_labels.size(0)
                
                test_dog_true_pos += ((test_preds == test_dog_class) & (test_labels == test_dog_class)).sum().item()
                test_dog_false_pos += ((test_preds == test_dog_class) & (test_labels != test_dog_class)).sum().item()
                test_dog_false_neg += ((test_preds != test_dog_class) & (test_labels == test_dog_class)).sum().item()
                
                test_cat_true_pos += ((test_preds == test_cat_class) & (test_labels == test_cat_class)).sum().item()
                test_cat_false_pos += ((test_preds == test_cat_class) & (test_labels != test_cat_class)).sum().item()
                test_cat_false_neg += ((test_preds != test_cat_class) & (test_labels == test_cat_class)).sum().item()
                
                
        test_accuracy = test_correct / test_total

        test_precision_dog = test_dog_true_pos / (test_dog_true_pos + test_dog_false_pos) if (test_dog_true_pos + test_dog_false_pos) > 0 else 0
        test_recall_dog = test_dog_true_pos / (test_dog_true_pos + test_dog_false_neg) if (test_dog_true_pos + test_dog_false_neg) > 0 else 0
        test_f1_dog = 2 * (test_precision_dog * test_recall_dog) / (test_precision_dog + test_recall_dog) if (test_precision_dog + test_recall_dog) > 0 else 0
        
        test_precision_cat = test_cat_true_pos / (test_cat_true_pos + test_cat_false_pos) if (test_cat_true_pos + test_cat_false_pos) > 0 else 0
        test_recall_cat = test_cat_true_pos / (test_cat_true_pos + test_cat_false_neg) if (test_cat_true_pos + test_cat_false_neg) > 0 else 0
        test_f1_cat = 2 * (test_precision_cat * test_recall_cat) / (test_precision_cat + test_recall_cat) if (test_precision_cat + test_recall_cat) > 0 else 0
        
        print(f"Test Accuracy: {test_accuracy*100:.2f}")
        print(f"Dog - Test Precision: {test_precision_dog*100:.2f}, Test Recall: {test_recall_dog*100:.2f}, Test F1: {test_f1_dog*100:.2f}")
        print(f"Cat - Test Precision: {test_precision_cat*100:.2f}, Test Recall: {test_recall_cat*100:.2f}, Test F1: {test_f1_cat*100:.2f}")
        print(15*"-", end="\n\n")
        
        testing_metrics = {
            "test_accuracy": f"{test_accuracy*100:.2f}",
            "test_dog_precision": f"{test_precision_dog*100:.2f}",
            "test_dog_recall": f"{test_recall_dog*100:.2f}",
            "test_dog_f1": f"{test_f1_dog*100:.2f}",
            "test_cat_precision": f"{test_precision_cat*100:.2f}",
            "test_cat_recall": f"{test_recall_cat*100:.2f}",
            "test_cat_f1": f"{test_f1_cat*100:.2f}"
        }
        
        r.record_testing(testing_metrics)
        r.save_model_metrics_to_json()
        del model
        torch.cuda.empty_cache()