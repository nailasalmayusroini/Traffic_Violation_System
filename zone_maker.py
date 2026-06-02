import cv2
import json
import numpy as np
import os

print("=== TRAFFIC VIOLATION SYSTEM: ZONE CONFIGURATOR ===")
loc_num = input("Enter location number to map (1, 2, 3): ").strip()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

VIDEO_PATH = os.path.join(BASE_DIR, "videos", f"location{loc_num}.mp4")  # Looks for location1.mov, location2.mov, etc.
OUTPUT_FILE = os.path.join(BASE_DIR, "configuration_data", f"zones_location{loc_num}.json")  # Saves unique zones_location1.json, etc.
FRAME_SIZE = (640, 640)

ZONE_TYPES = {
    "1": "no_parking",
    "2": "limited_parking",
}

points = []
zones = []


def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((int(x), int(y))) # Explicit int conversion for clean JSON saving
        print(f"  Corner added: ({x}, {y})  [{len(points)} total]")


if not os.path.exists(VIDEO_PATH):
    print(f"\nERROR: Could not find the file '{VIDEO_PATH}' in this folder.")
    print("Please make sure your files are named exactly 'location1.mp4', 'location2.mp4', etc.")
    exit()

cap = cv2.VideoCapture(VIDEO_PATH)
ok, frame = cap.read()
cap.release()

if not ok:
    print(f"\nERROR: OpenCV could not decode the codec of '{VIDEO_PATH}' natively.")
    exit()

frame = cv2.resize(frame, FRAME_SIZE)

print(f"\nSuccessfully loaded: {VIDEO_PATH}")
print("Left click = add corner | Enter = finish zone | ESC = save and quit")
print("-" * 50)

window_title = f"Zone Maker - Location {loc_num}"
cv2.namedWindow(window_title)
cv2.setMouseCallback(window_title, click_event)

while True:
    display = frame.copy()

    for zone in zones:
        pts = np.array(zone["polygon"], dtype=np.int32)
        color = zone["color"]
        overlay = display.copy()
        cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, 0.3, display, 0.7, 0, display)
        cv2.polylines(display, [pts], isClosed=True, color=color, thickness=2)
        lx, ly = zone["polygon"][0]
        cv2.putText(display, zone["label"], (lx + 5, ly - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    for p in points:
        cv2.circle(display, p, 5, (0, 255, 255), -1)
    if len(points) > 1:
        for i in range(len(points) - 1):
            cv2.line(display, points[i], points[i + 1], (0, 255, 255), 1)
        cv2.line(display, points[-1], points[0], (0, 255, 255), 1)

    cv2.putText(display, f"Points: {len(points)}  Zones: {len(zones)}",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(display, "ENTER = finish zone    ESC = save & quit",
                (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

    cv2.imshow(window_title, display)
    key = cv2.waitKey(20) & 0xFF

    if key == 13: # ENTER KEY
        if len(points) < 3:
            print("  Need at least 3 points. Keep clicking.")
            continue

        print("  Zone name: ", end="", flush=True)
        name = input().strip()

        print("  Zone type:")
        print("    1 = no_parking      (immediate violation)")
        print("    2 = limited_parking (30 second timer)")
        print("  Enter 1 or 2: ", end="", flush=True)
        choice = input().strip()
        ztype = ZONE_TYPES.get(choice, "limited_parking")
        color = (0, 0, 220) if ztype == "no_parking" else (0, 140, 255)

        zones.append({
            "id": f"zone_{len(zones) + 1}",
            "label": name,
            "zone_type": ztype,
            "polygon": points.copy(),
            "color": color,
        })
        print(f"  Saved '{name}' as {ztype} ({len(points)} corners).")
        print(f"  Total zones so far: {len(zones)}\n")
        points.clear()

    elif key == 27: # ESC KEY
        break

cv2.destroyAllWindows()

if len(zones) == 0:
    print("\nNo zones were drawn. File was not saved.")
    exit()

save_data = [
    {"id": z["id"], "label": z["label"],
     "zone_type": z["zone_type"], "polygon": z["polygon"]}
    for z in zones
]

with open(OUTPUT_FILE, "w") as f:
    json.dump(save_data, f, indent=2)

print(f"\nSaved {len(zones)} zone(s) to '{OUTPUT_FILE}'")
print(json.dumps(save_data, indent=2))
