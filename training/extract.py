import cv2
import json
import os

with open('races.json', 'r') as f:
    race_data = json.load(f)

VIDEO_PATH = race_data["video_name"]
RAW_FRAMES_DIR = "data/1_raw_frames"

def time_to_sec(t_str):
    h, m, s = map(int, t_str.split(':'))
    return h * 3600 + m * 60 + s

if not os.path.exists(RAW_FRAMES_DIR): 
    os.makedirs(RAW_FRAMES_DIR)

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)

for event in race_data["events"]:
    base_sec = time_to_sec(event["start_time"])
    for i in range(5):
        target_sec = base_sec + (i * 2)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(target_sec * fps))
        success, frame = cap.read()
        if success:
            cv2.imwrite(f"{RAW_FRAMES_DIR}/time_{target_sec}.jpg", frame)

cap.release()
