import cv2
import pandas as pd
import numpy as np
import sys
from ultralytics import YOLO
import json


YOLO_INTERVAL = 10
BOX_TIMEOUT = 30
WINDOW_MS = 500


# --------------------------------------------------
# STEP 1
# Video Loading
# --------------------------------------------------

def load_video(video_path):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(
            f"Could not open video: {video_path}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_count = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    duration_ms = int(
        (frame_count / fps) * 1000
    )

    return cap, fps, duration_ms


# --------------------------------------------------
# STEP 2
# IMU Loading
# --------------------------------------------------

def load_imu(imu_path):

    return pd.read_csv(imu_path)


# --------------------------------------------------
# STEP 3
# YOLO Loading
# --------------------------------------------------

def load_yolo():

    return YOLO("yolov8n.pt")


# --------------------------------------------------
# STEP 4 + 5 + 6
# YOLO Localization
# Persistent Boxes
# ROI Motion
# Full Frame Fallback
# --------------------------------------------------

def get_video_motion(video_path, model):

    cap = cv2.VideoCapture(video_path)

    ret, prev_frame = cap.read()

    if not ret:
        raise ValueError(
            "Could not read video"
        )

    prev_gray = cv2.cvtColor(
        prev_frame,
        cv2.COLOR_BGR2GRAY
    )

    frame_idx = 1

    last_box = None
    box_age = BOX_TIMEOUT + 1

    motion_values = []
    motion_sources = []

    roi_count = 0
    full_count = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        if frame_idx % YOLO_INTERVAL == 0:

            results = model(
                frame,
                verbose=False
            )

            dog_found = False

            for r in results:

                for box in r.boxes:

                    cls = int(box.cls[0])

                    if cls == 16:

                        x1, y1, x2, y2 = map(
                            int,
                            box.xyxy[0]
                        )

                        last_box = (
                            x1,
                            y1,
                            x2,
                            y2
                        )

                        box_age = 0

                        dog_found = True
                        break

                if dog_found:
                    break

        use_roi = False

        if (
            last_box is not None
            and box_age <= BOX_TIMEOUT
        ):

            x1, y1, x2, y2 = last_box

            roi_prev = prev_gray[
                y1:y2,
                x1:x2
            ]

            roi_curr = gray[
                y1:y2,
                x1:x2
            ]

            if (
                roi_prev.size > 0
                and roi_prev.shape == roi_curr.shape
            ):

                diff = cv2.absdiff(
                    roi_prev,
                    roi_curr
                )

                motion = float(
                    np.mean(diff)
                )

                use_roi = True
                roi_count += 1

        if not use_roi:

            diff = cv2.absdiff(
                prev_gray,
                gray
            )

            motion = float(
                np.mean(diff)
            )

            full_count += 1

        motion_values.append(
            motion
        )

        motion_sources.append(
            "ROI" if use_roi else "FULL"
        )

        prev_gray = gray.copy()

        frame_idx += 1
        box_age += 1

    cap.release()

    total = roi_count + full_count

    print(f"\nROI Frames: {roi_count}")
    print(f"FULL Frames: {full_count}")
    print(
        f"ROI Coverage: "
        f"{100 * roi_count / total:.2f}%"
    )

    return motion_values, motion_sources


# --------------------------------------------------
# STEP 7
# Separate Normalization
# --------------------------------------------------

def get_video_confidences(
    motion_values,
    motion_sources,
    fps
):

    roi_values = [
        m
        for m, s in zip(
            motion_values,
            motion_sources
        )
        if s == "ROI"
    ]

    full_values = [
        m
        for m, s in zip(
            motion_values,
            motion_sources
        )
        if s == "FULL"
    ]

    roi_min = min(roi_values)
    roi_max = max(roi_values)

    full_min = min(full_values)
    full_max = max(full_values)

    frame_confidences = []

    for motion, source in zip(
        motion_values,
        motion_sources
    ):

        if source == "ROI":

            conf = (
                motion - roi_min
            ) / (
                roi_max - roi_min + 1e-8
            )

        else:

            conf = (
                motion - full_min
            ) / (
                full_max - full_min + 1e-8
            )

        frame_confidences.append(
            float(conf)
        )

    window_frames = int(
        fps * 0.5
    )

    video_confidences = []

    for start in range(
        0,
        len(frame_confidences),
        window_frames
    ):

        chunk = frame_confidences[
            start:start + window_frames
        ]

        if len(chunk) == 0:
            continue

        avg_conf = np.mean(chunk)

        video_conf = (
            0.3 +
            0.7 * avg_conf
        )

        video_confidences.append(
        float(video_conf)
    )
        

    return video_confidences


# --------------------------------------------------
# STEP 8
# IMU Confidence
# --------------------------------------------------

def get_imu_confidence(imu):

    accel_mag = np.sqrt(
        imu["accel_x"]**2 +
        imu["accel_y"]**2 +
        imu["accel_z"]**2
    )

    gyro_mag = np.sqrt(
        imu["gyro_x"]**2 +
        imu["gyro_y"]**2 +
        imu["gyro_z"]**2
    )

    combined_signal = (
        0.6 * accel_mag +
        0.4 * gyro_mag
    )

    imu["energy"] = pd.Series(
        combined_signal
    ).rolling(
        window=50,
        min_periods=1
    ).std()

    imu["energy"] = (
        imu["energy"]
        .fillna(0)
    )

    energy_min = imu["energy"].min()
    energy_max = imu["energy"].max()

    imu["imu_confidence"] = (
        imu["energy"] - energy_min
    ) / (
        energy_max - energy_min + 1e-8
    )

    return imu

# --------------------------------------------------
# STEP 9
# Sensor Fusion
# --------------------------------------------------

def build_timeline(
    duration_ms,
    video_confidences,
    imu
):

    timeline = []

    timestamps = np.arange(
        0,
        duration_ms + 1,
        WINDOW_MS
    )

    imu_end_time = (
        imu["timestamp_ms"].max()
    )

    for i, ts in enumerate(
        timestamps
    ):

        if i >= len(
            video_confidences
        ):
            break

        video_conf = (
            video_confidences[i]
        )

        if ts <= imu_end_time:

            nearest_idx = (
                imu["timestamp_ms"] - ts
            ).abs().idxmin()

            imu_conf = float(
                imu.loc[
                    nearest_idx,
                    "imu_confidence"
                ]
            )

            final_conf = (
                0.7 * video_conf +
                0.3 * imu_conf
            )

        else:

            final_conf = (
                video_conf
            )

        activity = (
            "Active"
            if final_conf >= 0.25
            else "Static"
        )

        timeline.append({

            "timestamp_ms":
            int(ts),

            "activity":
            activity,

            "confidence":
            round(
                float(final_conf),
                3
            )

        })

    return timeline
# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    if len(sys.argv) != 3:

        print(
            "Usage:\n"
            "python run_pipeline_clean.py "
            "<video_path> <imu_csv>"
        )

        return

    video_path = sys.argv[1]
    imu_path = sys.argv[2]

    cap, fps, duration_ms = load_video(
        video_path
    )

    imu = load_imu(
        imu_path
    )

    model = load_yolo()

    motion_values, motion_sources = (
        get_video_motion(
            video_path,
            model
        )
    )

    video_confidences = (
        get_video_confidences(
            motion_values,
            motion_sources,
            fps
        )
    )

    imu = get_imu_confidence(
        imu
    )

    timeline = build_timeline(
    duration_ms,
    video_confidences,
    imu
 )

    with open(
    "timeline.json",
    "w"
)   as f:

        json.dump(
        timeline,
        f,
        indent=2
    )

    print(
    f"\nMotion Samples: "
    f"{len(motion_values)}"
)

    print(
    f"Video Windows: "
    f"{len(video_confidences)}"
)

    print(
    f"Timeline Entries: "
    f"{len(timeline)}"
)

    print(
    "\nSTEP 1-10 COMPLETE"
)

    print(
    "timeline.json generated"
)

    cap.release()


if __name__ == "__main__":
    main()