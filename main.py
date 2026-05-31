# main.py
import cv2
import os
# Import your teammates' code from the modules package
from modules.vehicle_tracker import VehicleTrackerPipeline
from modules.zone_logic import load_zones, update_all_vehicles
from modules.excel_logger import TechnicalViolationLogger

def main():
    video_path = "input.mp4" # Put any stock traffic video here for testing
    
    if not os.path.exists(video_path):
        print(f"[ERROR] Please place a sample video file named '{video_path}' in this folder.")
        return

    # 1. Initialize variables and modules
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30.0 # Fallback default
    
    frame_id = 0
    timer_state = {} # Stores tracking history and timers for Person 3
    
    # Instantiate the modules
    tracker_pipeline = VehicleTrackerPipeline()
    logger = TechnicalViolationLogger()
    
    # Load the zones mapped out by zone_maker.py
    try:
        zones = load_zones("zones.json")
    except FileNotFoundError:
        print("[WARNING] 'zones.json' not found. Please run zone_maker.py first to configure zones!")
        return

    print("\n[INFO] Starting Master Intelligent Traffic Monitoring Pipeline...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_id += 1
        
        # --- PHASE 1: Person 2's AI Tracking ---
        # Returns: [frame_id, track_id, [x1, y1, x2, y2], class_name]
        tracking_payloads = tracker_pipeline.process_frame(frame, frame_id)
        
        # --- PHASE 2: Data Transformation (Your Role) ---
        # Convert Person 2's output into the dictionary structure Person 3's logic expects
        prepared_vehicles = []
        vehicle_type_map = {} # Keep track of types for your Excel logger
        
        for payload in tracking_payloads:
            fid, track_id, box, class_name = payload
            x1, y1, x2, y2 = box
            
            # Calculate the explicit bottom-center centroid of the vehicle bounding box
            cx = int((x1 + x2) / 2)
            cy = int(y2) # Using bottom edge coordinates to verify road-contact region
            
            prepared_vehicles.append({
                "id": track_id,
                "centroid": (cx, cy)
            })
            vehicle_type_map[track_id] = class_name

        # --- PHASE 3: Person 3's Spatial-Temporal Rule Checking ---
        # Returns a list of dictionaries: {"id": tracking_id, "status": "ILLEGAL/LEGAL/MONITORING", ...}
        evaluated_statuses = update_all_vehicles(prepared_vehicles, zones, timer_state, fps)

        # --- PHASE 4: Person 4's Excel Event Logging ---
        for vehicle_report in evaluated_statuses:
            v_id = vehicle_report["id"]
            status = vehicle_report["status"]
            zone_label = vehicle_report["zone_label"]
            duration = vehicle_report["seconds"]
            
            if status == "ILLEGAL":
                v_type = vehicle_type_map.get(v_id, "unknown")
                # Trigger your logger module to append to Excel seamlessly
                logger.log_violation(v_id, v_type, zone_label, duration)

        # --- PHASE 5: Basic Pipeline Verification Display ---
        # Draw basic visual bounding boxes to verify everything works before passing to Person 1
        for payload in tracking_payloads:
            _, track_id, box, class_name = payload
            
            # Find matching status evaluation
            status = "MONITORING"
            for report in evaluated_statuses:
                if report["id"] == track_id:
                    status = report["status"]
                    break
            
            # Set bounding box border color based on legal criteria
            color = (0, 255, 0) # Green for legal
            if status == "ILLEGAL":
                color = (0, 0, 255) # Red for violations
                
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
            cv2.putText(frame, f"ID: {track_id} | {status}", (box[0], box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow("Master Integration Pipeline Sandbox", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n[INFO] Processing complete. Final spreadsheet saved.")

if __name__ == "__main__":
    main()
