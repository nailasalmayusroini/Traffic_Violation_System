import os
import json
import cv2
import numpy as np

ZONES_FILE = "zones.json"
VALID_ZONE_TYPES = {"no_parking", "limited_parking"}

def load_zones(filepath=ZONES_FILE):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"'{filepath}' not found. Run zone_maker.py first.")
    with open(filepath, "r") as f:
        zones = json.load(f)
    for zone in zones:
        if zone["zone_type"] not in VALID_ZONE_TYPES:
            raise ValueError(f"Invalid zone_type '{zone['zone_type']}' in zone '{zone['id']}'.")
    print(f"Loaded {len(zones)} zone(s) from '{filepath}'")
    return zones

def get_zone_for_vehicle(cent, zones):
    cx, cy = float(cent[0]), float(cent[1])
    for zone in zones:
        res = cv2.pointPolygonTest(zone["poly_array"], (cx, cy), measureDist=False)
        if res >= 0:
            return zone
    return None

def make_empty_state():
    """Initializes a persistent tracking state for a newly detected vehicle."""
    return {
        "zone_id": None,
        "stationary_frames": 0,
        "stationary_sec": 0.0,
        "status": "MONITORING",
    }

def classify_vehicle(vehicle, zones, timer_state, fps):
    vid = vehicle["id"]
    cent = vehicle["centroid"]

    if vid not in timer_state:
        timer_state[vid] = make_empty_state()

    state = timer_state[vid]
    zone = get_zone_for_vehicle(cent, zones)

    # Vehicle is Outside the Polygon Zone Boundaries 
    if zone is None:
        state["zone_id"] = None
        state["stationary_frames"] = 0
        state["stationary_sec"] = 0.0
        state["status"] = "MONITORING"
        return {"id": vid, "status": "MONITORING", "zone_label": None, "seconds": 0.0}

    zone_id = zone["id"]
    zone_type = zone["zone_type"]

    # Vehicle has Transformed or Entered a New Zone 
    if state["zone_id"] != zone_id:
        state["zone_id"] = zone_id
        state["stationary_frames"] = 0
        state["stationary_sec"] = 0.0
        state["status"] = "LEGAL"

    # Active Frame Temporal Monitoring Loop 
    state["stationary_frames"] += 1
    state["stationary_sec"] = state["stationary_frames"] / fps

    if zone_type == "no_parking":
        state["status"] = "ILLEGAL"
    elif zone_type == "limited_parking":
        state["status"] = "ILLEGAL"

    return {
        "id": vid,
        "status": state["status"],
        "zone_label": zone["label"],
        "seconds": round(state["stationary_sec"], 1),
    }

def update_all_vehicles(vehicles, zones, timer_state, fps):
    """Evaluates tracking frames against spatial zones."""
    return [classify_vehicle(v, zones, timer_state, fps) for v in vehicles]
