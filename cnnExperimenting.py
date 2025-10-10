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
from components.visualize import visualize_mae_vs_all_models
import torch
# ---CODE---

if __name__=="__main__":
    
    os.system("clear")
    
    if len(sys.argv) == 1 or "-train" in sys.argv:
        # default execution
        
        while True:
            
            perm_list = get_first_permutation_list()
            print(f"Permutation used for this iteration: {perm_list}")
            
            batch_size = perm_list[0]
            # see footnote (1) above for index definitions
            print(f"Batch_size = {batch_size}")
            
            train_dataset = get_dataset(batch_size=batch_size, mode="train")
            print("Training dataset retrieved!")
            
            device = torch.device("mps" if torch.mps.is_available() else "cpu")
            print(f"Device configured to : {device}")
            
            model = build_model(num_conv_layers=perm_list[1],
                                conv_start=perm_list[2],
                                kernel_size=perm_list[3],
                                pool_size=perm_list[4],
                                num_neural_layers=perm_list[5],
                                neural_start=perm_list[6])
            print("Model created!\n")
            
            model = model.to(device=device)
            print("Model set to use the mps device!")
            
            train_this_model(model, train_dataset, device, batch_size=batch_size)
            
            update_json_file_list()
            
            del model
            torch.cuda.empty_cache()
            
            print("Row loop completed! Model deleted from memory and torch cache emptied.\nRestarting loop with new index[0] list!")
        
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
            pass
        
        elif "-convLayers" in sys.argv:
            pass
        
        elif "-neuLayers" in sys.argv:
            pass
        
        elif "-batchSize" in sys.argv:
            pass