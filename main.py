import cv2
import os
import json      
import numpy as np

from modules.vehicle_tracker import VehicleTrackerPipeline
from modules.zone_logic import load_zones, update_all_vehicles
from modules.excel_logger import TechnicalViolationLogger

def main():
    print("== RUNNING TRAFFIC VIOLATION DETECTION SYSTEM ==")
    loc_num = input("Enter the location number you want to analyze (1, 2, 3): ").strip()
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    video_path = os.path.join(BASE_DIR, "videos", f"location{loc_num}.mp4")
    zones_json_path = os.path.join(BASE_DIR, "configuration_data", f"zones_location{loc_num}.json")
    
    video_output_dir = os.path.join(BASE_DIR, "final_output")
    os.makedirs(video_output_dir, exist_ok=True) # Guarantee directory exists first!
    
    output_excel_path = os.path.join(video_output_dir, f"traffic_violation_report_location{loc_num}.xlsx")
    
    if not os.path.exists(video_path) or not os.path.exists(zones_json_path):
        print(f"Error: Missing {video_path} or {zones_json_path} in this directory.")
        exit()

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30.0 
    
    frame_id = 0
    timer_state = {} # Stores tracking history and timers
    
    # Instantiate the modules
    tracker_pipeline = VehicleTrackerPipeline()
    logger = TechnicalViolationLogger(output_filepath=output_excel_path)
    
    # Load the zones mapped out by zone_maker.py
    try:
        zones = load_zones(zones_json_path)
    except FileNotFoundError:
        print(f"[WARNING] '{zones_json_path}' not found. Please run zone_maker.py first to configure zones!")
        return

    print("\n[INFO] Starting Master Intelligent Traffic Monitoring Pipeline...")
    
    cv2.namedWindow("Traffic Monitoring System", cv2.WINDOW_NORMAL)
    
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if orig_h > orig_w:
        cv2.resizeWindow("Traffic Monitoring System", 450, 800)
    else:
        cv2.resizeWindow("Traffic Monitoring System", 1280, 720)
    
    # --- VIDEO RECORDER INITIALIZATION ---
    video_output_dir = os.path.join(BASE_DIR, "final_output")
    os.makedirs(video_output_dir, exist_ok=True)
    video_out_path = os.path.join(video_output_dir, f"tracked_violation_location{loc_num}.avi")
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(video_out_path, fourcc, fps, (orig_w, orig_h))
    print(f"[INFO] Video recording initialized. Saving to: {video_out_path}")


    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_id += 1
        h, w, _ = frame.shape  
        
        x_scale = w / 640.0
        y_scale = h / 640.0

        scaled_zones = []
        for zone in zones:
            scaled_poly = [[int(pt[0] * x_scale), int(pt[1] * y_scale)] for pt in zone["polygon"]]
            scaled_zone = zone.copy()
            scaled_zone["poly_array"] = np.array(scaled_poly, dtype=np.int32)
            scaled_zones.append(scaled_zone)
        
        # AI Tracking 
        tracking_payloads = tracker_pipeline.process_frame(frame, frame_id)
        
        # Data Transformation
        prepared_vehicles = []
        
        for payload in tracking_payloads:
            fid, track_id, box, class_name = payload
            x1, y1, x2, y2 = box
            
            cx = int((x1 + x2) / 2)
            cy = int(y2)
            
            prepared_vehicles.append({
                "id": track_id,
                "centroid": (cx, cy)
            })

        # Spatial-Temporal Rule Checking (Using scaled zones)
        evaluated_statuses = update_all_vehicles(prepared_vehicles, scaled_zones, timer_state, fps)

        for report in evaluated_statuses:
            tid = report["id"]
            if tid in timer_state:
                timer_state[tid]["zone_label"] = report["zone_label"]
                for payload in tracking_payloads:
                    if payload[1] == tid:
                        timer_state[tid]["vehicle_type"] = payload[3]
                        break

        # Excel Event Logging 
        current_frame_ids = [veh["id"] for veh in prepared_vehicles]
        for track_id in list(timer_state.keys()):
            vehicle_data = timer_state[track_id]

            if vehicle_data.get("status") == "ILLEGAL":
                if track_id not in current_frame_ids:
                    v_type = vehicle_data.get("vehicle_type", "unknown")
                    final_duration = vehicle_data.get("stationary_sec", 0.0)
                    zone_label = vehicle_data.get("zone_label", "Unknown Zone")

                    logger.log_violation(track_id, v_type, zone_label, round(final_duration, 1))
            
                    del timer_state[track_id]

        # Draw UI Layout Elements
        for payload in tracking_payloads:
            _, track_id, box, class_name = payload
            
            status = "MONITORING"
            for report in evaluated_statuses:
                if report["id"] == track_id:
                    status = report["status"]
                    break
            
            color = (0, 255, 0) # Green
            if status == "ILLEGAL":
               color = (0, 0, 255) # Red
                
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
            cv2.putText(frame, f"ID: {track_id} | {status}", (box[0], box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Render Frame To Scaled Window 
        video_writer.write(frame)
        cv2.imshow("Traffic Monitoring System", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # CLEANUP
    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()
    
    try:
        if hasattr(logger, 'save'):
            logger.save()
        elif hasattr(logger, 'close'):
            logger.close()
        print("[INFO] Excel data pipeline flushed and saved successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to write Excel sheet to disk: {e}")

    print("\n[INFO] Processing complete. Final spreadsheet saved.")

if __name__ == "__main__":
    main()
