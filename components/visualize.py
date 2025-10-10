# ---NOTES---
"""
Plotting the data gathered as of now
I will plot different graphs to see how changing a factor changes the outcome
list of different plotting:
    - (1) mae vs all models
    - (2) mae vs epochs in groups of the same identical model
    - (3) mae vs number of conv layers grouped in identical models except epochs and layer densities
    - (4) mae vs number of neural layers grouped in identical models except conv layers and epochs
    - (5) mae vs batch size grouped in identical models except batch_size
    - (6) mae vs kernel size grouped in identical models except kernel size
total of 6 graphs
"""

# ---IMPORTS---
import json
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
# ---CODE---

def _get_stats_list():
    
    file_path = Path("./supporting_files/stats.json")
    
    if not file_path.exists():
        print("Cannot visualize data since stats.json does not exist yet!\nPlease train models to gather some data!")
        exit()
        
    with open(file_path, "r") as f:
        stats = json.load(f)
        
    return stats
    
def visualize_mae_vs_all_models():
    
    stats = _get_stats_list()
    
    data_1 = []
    model_number = []
    mae = []

    for m in stats:
        
        this_mae = m["mae"]
        this_model_name = m["model_name"]
        
        data_1.append((this_model_name, this_mae))

    for e in data_1:
        model_number.append(int(e[0].split("_")[1]))
        mae.append(e[1])
        
    plt.scatter(model_number, mae)

    plt.title("MAE value for ALL")
    plt.xlabel("Model Number")
    plt.ylabel("MAE")

    plt.savefig("./supporting_files/model_charts/mae_vs_all_models.pdf")

    plt.show()

 