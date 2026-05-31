import cv2
import numpy as np
from ultralytics import YOLO
# Note: Ensure you have 'deep-sort-realtime' installed via pip
from deep_sort_realtime.deepsort_tracker import DeepSort

class VehicleTrackerPipeline:
    def __init__(self, model_path="yolov8n.pt"):
        """
        Initializes the AI Detection and Tracking Pipeline.
        COCO Classes for vehicles: 2 (car), 3 (motorcycle), 5 (bus), 7 (truck)
        """
        print("[INFO] Initializing YOLOv8 model...")
        self.model = YOLO(model_path)
        
        print("[INFO] Initializing DeepSORT Tracker...")
        self.tracker = DeepSort(
            max_age=30,           # Retain ID for 30 frames if object is temporarily occluded
            n_init=3,             # Confirm track after 3 consecutive frames
            nms_max_overlap=1.0,  # Non-maxima suppression threshold
            max_cosine_distance=0.2
        )
        self.allowed_classes = [2, 3, 5, 7] 

    def process_frame(self, frame, frame_id):
        """
        Processes a single video frame to detect and track vehicles.
        Returns a structured list: [Frame_ID, Track_ID, [x1, y1, x2, y2], Class_Name]
        """
        # 1. Run YOLOv8 Detection
        results = self.model(frame, verbose=False)[0]
        detections = []

        # 2. Parse Detections for DeepSORT input format: [ [left, top, w, h], confidence, detection_class ]
        for box in results.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            
            if class_id in self.allowed_classes and confidence > 0.4:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1
                class_name = self.model.names[class_id]
                
                detections.append(([x1, y1, w, h], confidence, class_name))

        # 3. Update DeepSORT Tracking
        tracks = self.tracker.update_tracks(detections, frame=frame)
        frame_payload = []

        # 4. Extract Active Tracks
        for track in tracks:
            if not track.is_confirmed():
                continue
                
            track_id = track.track_id
            ltrb = track.to_ltrb() # Get bounding box in [left, top, right, bottom] format
            x1, y1, x2, y2 = map(int, ltrb)
            class_name = track.get_class()
            
            # Construct the clean packet for Person 3
            frame_payload.append([frame_id, track_id, [x1, y1, x2, y2], class_name])
            
        return frame_payload

# --- Sandbox Mock Execution for Pair B Parallel Testing ---
if __name__ == "__main__":
    # Replace with the person 1 video path
    cap = cv2.VideoCapture("person_traffic.mp4")
    pipeline = VehicleTrackerPipeline()
    frame_counter = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_counter += 1
        
        # Extract the structured payload
        tracking_data = pipeline.process_frame(frame, frame_counter)
        
        # This printed output is exactly what Person 3 needs to ingest
        if len(tracking_data) > 0:
            print(f"Frame {frame_counter} Payload Example: {tracking_data[0]}")
            
        # Optional: Render basic visualization for validation before passing to Person 1/4
        for data in tracking_data:
            fid, tid, box, cls_name = data
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
            cv2.putText(frame, f"ID: {tid} {cls_name}", (box[0], box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        
        cv2.imshow("Pair B Development Sandbox", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
