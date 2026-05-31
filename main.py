# main.py
import cv2
import os

from modules.vehicle_tracker import VehicleTrackerPipeline
from modules.zone_logic import load_zones, update_all_vehicles
from modules.excel_logger import TechnicalViolationLogger

def main():
    print("== RUNNING TRAFFIC VIOLATION DETECTION SYSTEM ==")
    loc_num = input("Enter the location number you want to analyze (1, 2, 3, 4, or 5): ").strip()
    
    video_path = f"../videos/location{loc_num}.mp4"
    zones_json_path = f"../configuration_data/zones_location{loc_num}.json"
    output_excel_path = f"../final_outputs/traffic_violation_report_location{loc_num}.xlsx"
    
    if not os.path.exists(video_path) or not os.path.exists(zones_json_path):
        print(f"Error: Missing {video_path} or {zones_json_path} in this directory.")
        exit()

    # 1. Initialize variables and modules
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30.0 
    
    frame_id = 0
    timer_state = {} # Stores tracking history and timers
    
    # Instantiate the modules
    tracker_pipeline = VehicleTrackerPipeline()
    logger = TechnicalViolationLogger()
    
    # Load the zones mapped out by zone_maker.py
    try:
        zones = load_zones(zones_json_path)
    except FileNotFoundError:
        print("[WARNING] '{zones_json_path}' not found. Please run zone_maker.py first to configure zones!")
        return

    print("\n[INFO] Starting Master Intelligent Traffic Monitoring Pipeline...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_id += 1
        
        # --- PHASE 1:AI Tracking ---
        # Returns: [frame_id, track_id, [x1, y1, x2, y2], class_name]
        tracking_payloads = tracker_pipeline.process_frame(frame, frame_id)
        
        # --- PHASE 2: Data Transformation ---
        prepared_vehicles = []
        vehicle_type_map = {} 
        
        for payload in tracking_payloads:
            fid, track_id, box, class_name = payload
            x1, y1, x2, y2 = box
            
            cx = int((x1 + x2) / 2)
            cy = int(y2)
            
            prepared_vehicles.append({
                "id": track_id,
                "centroid": (cx, cy)
            })
            vehicle_type_map[track_id] = class_name

        # --- PHASE 3: Spatial-Temporal Rule Checking ---
        # Returns a list of dictionaries: {"id": tracking_id, "status": "ILLEGAL/LEGAL/MONITORING", ...}
        evaluated_statuses = update_all_vehicles(prepared_vehicles, zones, timer_state, fps)

        # --- PHASE 4: Excel Event Logging ---
        for vehicle_report in evaluated_statuses:
            v_id = vehicle_report["id"]
            status = vehicle_report["status"]
            zone_label = vehicle_report["zone_label"]
            duration = vehicle_report["seconds"]
            
            if status == "ILLEGAL":
                v_type = vehicle_type_map.get(v_id, "unknown")
                logger.log_violation(v_id, v_type, zone_label, duration)

        # --- PHASE 5: Basic Pipeline Verification Display ---
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
    logger.save_report(output_excel_path)
    print("\n[INFO] Processing complete. Final spreadsheet saved.")

if __name__ == "__main__":
    main()
