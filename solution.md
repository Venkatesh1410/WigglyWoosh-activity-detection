# WigglyWoosh Technical Challenge Solution

## Solution Overview

The proposed solution categorizes dog behavior as either Active or Static using two complementary data sources:
- Video frames
- IMU sensor data attached to the collar

The resulting output is the timeline of activities tagged with confidence values recorded at 2 Hz (500 ms intervals) and stored in timeline.json file.


## Methodology

### Video-Based Activity Estimation

The YOLOv8n is a pretrained model that is applied for dog localization in the video feed. The localization is not done based on full-motion detection but only in the detected dog area.
To balance accuracy and computational efficiency, YOLO inference is executed every 10 frames.


### Persistent ROI Tracking

Object detection algorithms can sometimes fail at detections because of pose variations, blurring effects, or partial occlusions. In order to maintain object continuity, the latest bounding box is used for 30 frames(1 second).
This allows activity estimation to remain focused on the dog even during temporary detection failures.


### Motion Analysis

Frame-to-frame motion is calculated using grayscale difference analysis.

Two operating modes are used:
- ROI Motion: Motion computed within the dog bounding box.
- Full-Frame Fallback: Used only when no valid ROI is available.

This hybrid approach improves robustness while ensuring uninterrupted activity estimation.


### Confidence Normalization

The ROI-based motion and the full-frame motion have naturally different value ranges. In order to ensure that neither becomes dominant over the other, the two are normalized independently prior to converting them into video confidence values.


### IMU Activity Estimation

The IMU branch combines both:
- Accelerometer magnitude
- Gyroscope magnitude

The energy of a rolling window in the merged signal is determined and normalized to obtain the IMU activity confidence score.


### Sensor Fusion

The final activity confidence is obtained through weighted fusion:
- Final Confidence = 0.7 × Video Confidence + 0.3 × IMU Confidence

This is because the two sources complement each other as follows:
- Video gives evidence of visible motion.
- The IMU detects physical motion even when visual motion is not so evident.


## Temporal Alignment

Video stream and IMU stream have a differing time length. This is because IMU is available for approximately the first half of the recording, but the video stream goes on after the IMU recording interval.

To address this, sensor fusion occurs when both streams are present. If the timestamp goes beyond the IMU time length, the software automatically uses video- based confidence while retaining the same output format and sampling frequency.

This is to ensure continuity in the timeline generation through the entire video stream without wasting any useful data.


## Design Evolution

The initial version of the pipeline depended only on motion detection. Though computationally efficient, it was prone to motion becoming lost within the rest of the frame due to local motion being spread out across the entire scene.

In order to fix this, the YOLO dog localization was added. By limiting the scope of motion detection to the dog’s detected location, it becomes much easier to focus on what is important and still maintain robustness from always having the whole image/frame  to fall back on.

This detector-assisted design forms the basis of the final submitted solution


## Output

The pipeline generates a timeline.json file sampled at 2 Hz, where each entry contains:
- timestamp_ms
- activity
- confidence

The complete implementation is provided in a single executable file:
- run_pipeline.py
