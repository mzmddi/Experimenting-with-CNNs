# ---NOTES---
"""
Entry point of the entire program
"""

# ---IMPORTS---
import sys
# ---CODE---

if __name__=="__main__":
    
    if len(sys.argv) == 2:
        # default execution
        pass
    
    elif "-permutate" in sys.argv:
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
    
    pass