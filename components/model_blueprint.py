# ---NOTES---
"""
    - This method constructs the model layer by layer based on the params given
    - Parameters are passed from the permutation list
    - RECALL PERMUTATION LIST STRUCTURE:
        [batch_size, num_conv_layers, conv_start, kernel_size, pool_size, num_neural_layers, neural_start]
"""
# ---IMPORTS---
import torch.nn as nn
# -------------

def build_model(num_conv_layers, conv_start, kernel_size, pool_size, num_neural_layers, neural_start):
    """
    PARAMS:
        num_conv_layers: int
        conv_start: int
        kernel_size: int
        pool_size: int
        num_neural_layers: int
        neural_start: int
    """
    layers = []
    conv_in_channels = []
    conv_out_channels = []

    for i in range(num_conv_layers):

        if i == 0:
            
            conv_in_channels.append(3)
            conv_out_channels.append(conv_start)
            continue
        
        conv_in_channels.append(conv_out_channels[i-1])
        conv_out_channels.append(conv_in_channels[i]*2)

    for i in range(num_conv_layers):
        
        layers.append(nn.Conv2d(in_channels=conv_in_channels[i], out_channels=conv_out_channels[i], kernel_size=kernel_size, stride=1, padding=1))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(kernel_size=pool_size, stride=pool_size))
    
    layers.append(nn.Flatten())
    
    for i in range(num_neural_layers):
        
        if i+1 == num_neural_layers:
            
            layers.append(nn.LazyLinear(1))
            continue
        
        layers.append(nn.LazyLinear(neural_start*(i+1)))
        layers.append(nn.ReLU())
    
    model = nn.Sequential(*layers)
    
    return model