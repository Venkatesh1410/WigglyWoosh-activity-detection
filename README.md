# WigglyWoosh-activity-detection

Activity classification system pipeline using both computer vision and IMU sensor inputs for the classification of dog activity as either "Active" or "Static."

## Files

- [run_pipeline.py](./run_pipeline.py) — Main inference pipeline
- [solution.md](./solution.md) — Technical explanation of the approach
- [timeline.json](./timeline.json) — Generated activity timeline sampled at 2 Hz

## Approach

The solution combines:

1. YOLOv8n-based dog localization
2. Persistent bounding box tracking
3. ROI-based motion estimation
4. Full-frame fallback motion estimation
5. IMU energy analysis (accelerometer + gyroscope)
6. Confidence-based sensor fusion

Activity confidence is determined through the process of weighted fusion of visual and inertial signals and then exported as a structured JSON timeline.

## Output Format

Each timeline entry contains:

{
  "timestamp_ms": 500,
  "activity": "Active",
  "confidence": 0.82
}

## Execution

{python run_pipeline.py Dog_Video.mp4 collar_imu.csv}

This generates:
- timeline.json
