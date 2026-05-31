import os
import json
import cv2
import numpy as np
from collections import deque

ZONES_FILE = "zones.json"
STATIONARY_THRESHOLD_PX = 8
TIME_LIMIT_SECONDS = 30.0
STATIONARY_WINDOW = 5

VALID_ZONE_TYPES = {"no_parking", "limited_parking"}


def load_zones(filepath=ZONES_FILE):
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"'{filepath}' not found. Run zone_maker.py first."
        )
    with open(filepath, "r") as f:
        zones = json.load(f)
    for zone in zones:
        if zone["zone_type"] not in VALID_ZONE_TYPES:
            raise ValueError(
                f"Invalid zone_type '{zone['zone_type']}' in zone '{zone['id']}'. "
                f"Must be one of: {VALID_ZONE_TYPES}"
            )
        zone["poly_array"] = np.array(zone["polygon"], dtype=np.int32)
    print(f"Loaded {len(zones)} zone(s) from '{filepath}'")
    for z in zones:
        print(f"  {z['id']} - {z['label']} ({z['zone_type']})")
    return zones


def get_zone_for_vehicle(cent, zones):
    cx, cy = float(cent[0]), float(cent[1])
    for zone in zones:
        res = cv2.pointPolygonTest(zone["poly_array"], (cx, cy), measureDist=False)
        if res >= 0:
            return zone
    return None


def is_stationary(cent_history):
    if len(cent_history) < 2:
        return False
    dx = cent_history[-1][0] - cent_history[0][0]
    dy = cent_history[-1][1] - cent_history[0][1]
    dist = (dx**2 + dy**2) ** 0.5
    return dist < STATIONARY_THRESHOLD_PX


def make_empty_state():
    return {
        "cent_history": deque(maxlen=STATIONARY_WINDOW),
        "zone_id": None,
        "stationary_frames": 0,
        "stationary_sec": 0.0,
        "status": "MONITORING",
        # TODO: handle overlapping zones
    }


def classify_vehicle(vehicle, zones, timer_state, fps):
    vid = vehicle["id"]
    cent = vehicle["centroid"]

    if vid not in timer_state:
        timer_state[vid] = make_empty_state()

    state = timer_state[vid]
    state["cent_history"].append(cent)
    stationary = is_stationary(state["cent_history"])

    zone = get_zone_for_vehicle(cent, zones)

    if zone is None:
        state["zone_id"] = None
        state["stationary_frames"] = 0
        state["stationary_sec"] = 0.0
        state["status"] = "MONITORING"
        return {"id": vid, "status": "MONITORING", "zone_label": None, "seconds": 0.0}

    zone_id = zone["id"]
    zone_type = zone["zone_type"]

    if state["zone_id"] != zone_id:
        state["zone_id"] = zone_id
        state["stationary_frames"] = 0
        state["stationary_sec"] = 0.0
        state["status"] = "LEGAL"

    if zone_type == "no_parking":
        state["status"] = "ILLEGAL"

    elif zone_type == "limited_parking":
        if stationary:
            state["stationary_frames"] += 1
            state["stationary_sec"] = state["stationary_frames"] / fps
            state["status"] = "ILLEGAL" if state["stationary_sec"] >= TIME_LIMIT_SECONDS else "LEGAL"
        else:
            state["stationary_frames"] = 0
            state["stationary_sec"] = 0.0
            state["status"] = "LEGAL"

    return {
        "id": vid,
        "status": state["status"],
        "zone_label": zone["label"],
        "seconds": round(state["stationary_sec"], 1),
    }


def update_all_vehicles(vehicles, zones, timer_state, fps):
    active_ids = {v["id"] for v in vehicles}
    for gid in set(timer_state.keys()) - active_ids:
        del timer_state[gid]
    return [classify_vehicle(v, zones, timer_state, fps) for v in vehicles]
