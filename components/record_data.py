# ---NOTES---
# ---IMPORTS---
import json
from pathlib import Path
from datetime import datetime
# ---CODE---

class Recorder:
    
    def __init__(self):
        
        self.file_path = Path("./supporting_files/model_metrics.json")
        
        self.recorded_model_name = False
        
        self.recorded_meta_data = False
        
        self.recorded_arch_struct = False
        
        self.model_name = self._get_model_name()
        
        self.file_exists = self._file_exist()
        
        self.model_data = self.record_model_name()
        
    def _get_right_now(self):
        
        now = datetime.now()

        year = now.year
        
        month = now.month
        
        day = now.day
        
        hour = now.hour
        
        minute = now.minute
        
        second = now.second
        
        return {
            "year": year,
            "month": month,
            "day":day,
            "hour": hour,
            "minute": minute,
            "second": second
        }
    
    def _file_exist(self):
        
        if not self.file_path.exists():
            self.file_path.touch()
            with open(self.file_path, "w") as json_file:
                dump = []
                json.dump(dump, json_file, indent=4)
            
        return True
            
    def _get_model_name(self):
        
        return f"model_{self._get_right_now()["year"]}_{self._get_right_now()["month"]}_{self._get_right_now()["day"]}_{self._get_right_now()["hour"]}_{self._get_right_now()["minute"]}_{self._get_right_now()["second"]}"
    
    def save_model_metrics_to_json(self):
        # to be called once, at the end of the training of the model
        # this sends all the recorded data to the json file.
        # the mechanism works
        
        with open(self.file_path, "r") as json_file:
            
            stats = json.load(json_file)
        
        stats.append(self.model_data)
        
        with open(self.file_path, "w") as json_file:
            
            json.dump(stats, json_file, indent=4)
        
    def record_model_name(self):
        
        return {"model_name" : self.model_name}
    
    def record_timing(self, event_name):
        
        if "time_stamps" not in self.model_data["metadata"]:
            self.model_data["metadata"]["time_stamps"] = {}
        
        # self.model_data["time_stamps"][str(event_name)] = f"{self._get_right_now()["day"]:02d}/{self._get_right_now()["month"]:02d}/{self._get_right_now()["year"]:02d} - {self._get_right_now()["hour"]:02d}:{self._get_right_now()["minute"]:02d}:{self._get_right_now()["second"]:02d}"

        self.model_data["metadata"]["time_stamps"][str(event_name)] = f"{datetime.now().strftime("%d/%m/%Y - %H:%M:%S")}"
        
        match event_name:
            
            case "training_start": 
                self.start_training_time = datetime.now()
                
            case "training_end":
                self.end_training_time = datetime.now()
                
                duration = str((self.end_training_time - self.start_training_time)).split(".")[0]
                
                self.model_data["metadata"]["time_stamps"]["duration"] = f"{duration}"
                
    def record_metadata(self, metadata):
        """
        PARAMS
        metadata: Dictionnary = {
            batch_size: int
            total_epoch_number: int
            total_conv_layer: int
            total_neural_layer: int
            layers: Dict(layer: nn.Layers) (layer# -> layer)
        }
        """
        # The entire metadata dictionnary passed to this method is recorded as a dictionnary
        # the key for the metadata dictionnary within the overall dictionnary is "metadata"
        # self.model_data["metadata"] = metadata
        
        self.model_data["metadata"] = metadata
            
    def record_epoch(self, epoch_num, val_metrics):
        """
        PARAMS
        epoch_num: int
        epoch_val: Dict {
                "accuracy": str,
                "dog_precision": str,
                "dog_recall": str,
                "dog_f1": str,
                "cat_precision": str,
                "cat_recall": str, 
                "cat_f1": str
        }
        """
        
        if "validation_metrics" not in self.model_data:
            self.model_data["validation_metrics"] = {}
        
        right_now = str(datetime.now()).split(".")[0]
        
        val_metrics["time_stamp"] = right_now
        
        self.model_data["validation_metrics"][f"epoch_{epoch_num}"] = val_metrics
        
    
    def record_testing(self, testing_metrics):
        """
        PARAMS:
        testing_metrics: Dict {
            "test_accuracy": str,
            "test_dog_precision": str,
            "test_dog_recall": str,
            "test_dog_f1": str,
            "test_cat_precision": str,
            "test_cat_recall": str, 
            "test_cat_f1": str
        }
        }
        """
        
        if "testing_metrics" not in self.model_data:
            self.model_data["testing_metrics"] = {}
        
        right_now = str(datetime.now()).split(".")[0]
        
        testing_metrics["time_stamp"] = right_now
        
        self.model_data["testing_metrics"] = testing_metrics
        
# lets say i have this dictionnary structure in python: 
# dict = {
# "model_name": "model_0"
# }
# my program needs to add values to this dictionnary at different points of the program. One of the data i need to put is another dictionnart inside. After a couple entries of data, the dictionnary would look like this: 
# dict = {
# "model_name": "model_0",
# "validation_metrics":
# }