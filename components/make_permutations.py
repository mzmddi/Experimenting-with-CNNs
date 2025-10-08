
# ---NOTES---
"""
- Independentely run program for permutating the different values of the architecture of the model
    and saving it in a json. The data of the json is meant to be read by main.py
- METADATA OF JSON LIST:
    [
        [batch_size, num_conv_layers, conv_start, kernel_size, pool_size, num_neural_layers, neural_start]
    ]
"""
# ---IMPORTS---
import json
from pathlib import Path
# ---CODE---
def permutate():
    
    file_path = Path("./supporting_files/permutation_values.json")
    if not file_path.exists():
        file_path.touch()
        with open(file_path, "w") as json_file:
            dump = []
            json.dump(dump, json_file, indent=4)
            
    
    count = 0
    print("\n")
        
    for batch_size in [2, 4, 8, 16]:
        for num_conv_layers in [1, 2, 3, 4]:
            for conv_start in [32, 64]:
                for kernel_size in [3, 5]:
                    for pool_size in [2, 3]:
                        for num_neural_layers in [2, 3]:
                            for neural_start in [64, 128]:
                            
                                with open(file_path, "r") as json_file:
                                    permutations = json.load(json_file)
                            
                                v = []
                                
                                v.append(batch_size)
                                v.append(num_conv_layers)
                                v.append(conv_start)
                                v.append(kernel_size)
                                v.append(pool_size)
                                v.append(num_neural_layers)
                                v.append(neural_start)
                                
                                permutations.append(v)
                                
                                with open(file_path, "w") as json_file:
                                    json.dump(permutations, json_file, indent=4)
                                    
                                count += 1
                                
                                print(f"Permutations added: {count}", end="\r")
                            
    with open(file_path, "r") as json_file:
        permutations = json.load(json_file)
        
        print(f"Number of permutations: {len(permutations)}\n")
        print("Permutation finished. Find the list of lists at ./supporting_files/permutation_values.json!\n")