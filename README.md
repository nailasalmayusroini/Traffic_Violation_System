# Traffic Violation Detection System

An automated Computer Vision pipeline designed to detect illegal parking and traffic violations in restricted zones. The system monitors user-defined "Illegal Zones" via video streams, tracks vehicles across frames, and logs confirmed parking infractions directly into structured data sheets.

## Features
* **Object Detection:** Utilizes **YOLOv8** to localize and classify target vehicles (`car`, `truck`, `motorcycle`).
* **Multi-Object Tracking:** Uses **DeepSORT** to assign unique tracking IDs across sequential frames.
* **Geometric ROI Masking:** Evaluates zone compliance by checking if a vehicle's bounding box centroid crosses into the drawn coordinate boundaries via OpenCV.
* **Temporal Threshold Filtering:** Applies a **1.2-second threshold rule** to separate passing or transient traffic from actual stationary violations.
* **Automated Logging:** Automatically exports confirmed violation records (Vehicle Class, Unique ID, and precise Stop Duration) directly to Microsoft Excel spreadsheets.

## Tools and Technologies
* **Programming Language:** Python
* **Computer Vision Library:** OpenCV
* **Deep Learning Framework:** PyTorch (YOLOv8 + DeepSORT)
* **Data Management:** Pandas & Microsoft Excel
