# data_logger.py
import csv
import os
from datetime import datetime

class DataLogger:
    """
    Logs driver drowsiness events to a daily CSV file with timestamps.
    Creates a new log file each day under the 'logs' directory.
    """

    def __init__(self, log_dir="logs"):
        """
        Initialize the logger and ensure the log directory exists.
        """
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.current_date = None
        self.file_path = None
        self._update_log_file()

    def _update_log_file(self):
        """
        Create or switch to a new log file when the date changes.
        """
        today = datetime.now().strftime("%Y-%m-%d")
        if today != self.current_date:
            self.current_date = today
            self.file_path = os.path.join(self.log_dir, f"{today}_events.csv")

            # Initialize CSV file with headers if it's new
            if not os.path.exists(self.file_path):
                with open(self.file_path, mode="w", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow(["Timestamp", "Event Type", "Details"])

    def log_event(self, event_type, details=""):
        """
        Log an event with the current timestamp.
        """
        self._update_log_file()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.file_path, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, event_type, details])

        print(f"[LOG] {timestamp} - {event_type}: {details}")
