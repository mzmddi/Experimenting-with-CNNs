"""
./logging/logger.py
defining the logger used for logging
"""

# ---IMPORTS---
import logging
import os
from datetime import datetime, timezone, timedelta
# -------------


LOG_DIRECTORY = os.path.join(os.path.dirname(__file__), "logging")
os.makedirs(LOG_DIRECTORY, exist_ok=True)

log_level = 20

class EasternTimeFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        
        eastern_offset = timedelta(hours=-4)
        
        dt = datetime.fromtimestamp(record.created, timezone.utc) + eastern_offset
        return dt.strftime(datefmt) if datefmt else dt.isoformat()

formatter = EasternTimeFormatter("%(asctime)s [%(name)s] %(levelname)s @ %(filename)s:%(lineno)d - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

file_handler = logging.FileHandler(os.path.join(LOG_DIRECTORY, "general.log"))
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)
stream_handler.setFormatter(formatter)

genlogger = logging.getLogger("genlogger")
genlogger.setLevel(log_level)
genlogger.addHandler(file_handler)
genlogger.addHandler(stream_handler)
