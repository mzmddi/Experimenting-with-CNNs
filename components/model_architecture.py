# ---NOGTES---
"""
Proper structure of the architecture json file:
{
    num_of_conv_layer: int
    starting_conv_channel: int
    conv_kernel_size: int
    num_of_neural_layer: int
    starting_neural_channel: int
}
"""
# ---IMPORTS---
import json
from pathlib import Path
# ---CODE---

def get_model_architecture():
    
    file_path = Path("./supporting_files/model_architecture.json")
    if not file_path.exists():
        file_path.touch()
        print("File did not exist. File has been created by this program.")
        print("Please define (ONLY ONE) architecture for the model in order to proceed.")
        exit()
    
    with open(file_path, "r") as json_file:
        try:
            loaded_architecture = json.load(json_file)
        except Exception as e:
            print(f"Architecture failed to be loaded in program. Please review {file_path}.\nFull Error => {e}")
            exit()
    
    arch = {
            "num_of_conv_layer": None,
            "starting_conv_channel": None,
            "conv_kernel_size": None,
            "num_of_neural_layer": None,
            "starting_neural_channel": None
            }
    
    for key in arch.keys():
        
        if key not in loaded_architecture.keys():
            print(f"Entry {key} not found in {file_path}. Please add it to continue.")
            exit()
        
        arch[key] = loaded_architecture[key]
    
    print("Architecture loaded from json file:")
    for k in arch.keys():
        print(f"\t{k}: {arch[k]}")
        
    return arch
        