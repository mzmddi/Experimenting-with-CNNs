
# ---IMPORTS---
import torch.nn as nn
# -------------



def build_model(num_of_conv_layer, num_of_neural_layer, conv_in_channels, conv_out_channels, kernel_size, linear_in_channels, linear_out_channels, pool_size):
    
    layers = []
    
    for i in range(num_of_conv_layer - 1):
        
        layers.append(nn.Conv2d(in_channels=conv_in_channels[i], out_channels=conv_out_channels[i], kernel_size=kernel_size, stride=1, padding=1))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(kernel_size=pool_size, stride=pool_size))
    
    layers.append(nn.Flatten())
    
    for j in range(num_of_neural_layer - 1):
        
        if j == (num_of_neural_layer - 1):
            layers.append(nn.Linear(in_features=linear_in_channels[j], out_features=linear_out_channels[j]))
            pass
            
        layers.append(nn.Linear(in_features=linear_in_channels[j], out_features=linear_out_channels[j]))
        layers.append(nn.ReLU())
    
        
    model = nn.Sequential(*layers)
    
    return model

"""
PARAMS:
    num_of_conv_layer: int
    num_of_neural_layer: int
    conv_in_channels: List[int]
    conv_out_channels: List[int]
    kernel_size: int
    linear_in_channels: List[int]
    linear_out_channels: List[int]
    pool_size: int
"""


"""
 nn.Conv2d(in_channels=3, out_channels=32, kernel_size=2, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    
    # convolution block #1
    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    
    ## neural network section
    nn.Flatten(),
    
    nn.Linear(64*64*64,128),
    nn.ReLU(),
    nn.Linear(128, 1)
"""

"""
layers = []
layers.append(layer)
model = nn.Sequential(*layers)
"""

"""
model = nn.Sequential(
    # convolution component
    # convolution block #1
    nn.Conv2d(in_channels=3, out_channels=32, kernel_size=2, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    
    # convolution block #1
    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    
    ## neural network section
    nn.Flatten(),
    
    nn.Linear(64*64*64,128),
    nn.ReLU(),
    nn.Linear(128, 1)
)
"""