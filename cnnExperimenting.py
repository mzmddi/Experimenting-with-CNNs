# ---NOTES---
"""
Entry point of the entire program
FOOTNOTES: 
    (1) [batch_size, num_conv_layers, conv_start, kernel_size, pool_size, num_neural_layers, neural_start]
"""

# ---IMPORTS---
import sys
import os
from components.make_permutations import permutate
# ---CODE---

if __name__=="__main__":
    
    os.system("clear")
    
    if len(sys.argv) == 1:
        # default execution
        pass
    
    elif "-permutate" in sys.argv:
        permutate()
        pass
    
    elif "-train" in sys.argv:
        pass
    
    elif "-visualize" in sys.argv:
        
        if len(sys.argv) == 3:
            # default -visualize flag
            # python3 main.py -visualize
            pass
            
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