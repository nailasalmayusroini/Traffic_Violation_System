# modules/excel_logger.py
import pandas as pd
from datetime import datetime
import os

class TechnicalViolationLogger:
    def __init__(self, output_filepath="traffic_violation_report.xlsx"):
        output_dir = os.path.dirname(output_filepath) if os.path.dirname(output_filepath) else "final_output"
        base_name = os.path.basename(output_filepath)
        
        # Create directory if it doesn't exist yet
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"[INFO] Created directory: '{output_dir}'")
            
        self.filepath = os.path.join(output_dir, base_name)
        self.logged_ids = set()  # Track already logged vehicle IDs
        self.records = []

    def log_violation(self, vehicle_id, vehicle_type, zone_label, duration_sec):
        """Appends an illegal event record into memory."""
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
        print(f"[STAGE LOG] Cached violation for Vehicle ID {vehicle_id} in memory.")

    def save(self):
        """Physically compiles and writes all recorded rows to the Excel sheet once."""
        if not self.records:
            print("[WARNING] No violations recorded during this run. Creating placeholder report.")
            df = pd.DataFrame(columns=["Timestamp", "Vehicle ID", "Vehicle Type", "Location/Zone", "Stationary Duration (s)", "Status"])
        else:
            df = pd.DataFrame(self.records)
            
        df.to_excel(self.filepath, index=False)
        print(f"[EXCEL SUCCESS] Final report permanently written to: {self.filepath}")
