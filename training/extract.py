import cv2
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAINING_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.utils.config import load_json, time_to_seconds

race_data = load_json(os.path.join(TRAINING_DIR, "races.json"))

VIDEO_PATH = os.path.join(PROJECT_ROOT, race_data["video_name"])
RAW_FRAMES_DIR = os.path.join(TRAINING_DIR, "data", "1_raw_frames")

os.makedirs(RAW_FRAMES_DIR, exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)

for event in race_data["events"]:
    base_sec = time_to_seconds(event["start_time"])
    for i in range(5):
        target_sec = base_sec + (i * 2)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(target_sec * fps))
        success, frame = cap.read()
        if success:
            cv2.imwrite(os.path.join(RAW_FRAMES_DIR, f"time_{target_sec}.jpg"), frame)

cap.release()
