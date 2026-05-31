# modules/excel_logger.py
import pandas as pd
from datetime import datetime
import os

class TechnicalViolationLogger:
    def __init__(self, output_filepath="traffic_violation_report.xlsx"):
        self.filepath = output_filepath
        self.logged_ids = set()  # Keeps track of already logged vehicle IDs
        self.records = []

    def log_violation(self, vehicle_id, vehicle_type, zone_label, duration_sec):
        """
        Logs an illegal parking event to Excel if it hasn't been logged yet.
        """
        if vehicle_id in self.logged_ids:
            return

        self.logged_ids.add(vehicle_id)
        
        new_record = {
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Vehicle ID": int(vehicle_id),
            "Vehicle Type": vehicle_type,
            "Location/Zone": zone_label,
            "Stationary Duration (s)": float(duration_sec),
            "Status": "ILLEGAL"
        }
        self.records.append(new_record)

        # Convert to Pandas DataFrame and save to Excel
        df = pd.DataFrame(self.records)
        df.to_excel(self.filepath, index=False)
        print(f"[EXCEL LOG] Logged violation for Vehicle #{vehicle_id} in {zone_label}")
