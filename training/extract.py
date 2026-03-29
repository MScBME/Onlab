import cv2
import json
import os
from collections import defaultdict

JSON_FILE_PATH = "races.json"
RAW_FRAMES_DIR = "data/1_raw_frames"
OUTPUT_METADATA_FILE = "frames_metadata.jsonl"
FRAMES_TO_EXTRACT = 5
SECONDS_BETWEEN_FRAMES = 2

def time_to_sec(t_str):
    h, m, s = map(int, t_str.split(':'))
    return h * 3600 + m * 60 + s

os.makedirs(RAW_FRAMES_DIR, exist_ok=True)

with open(JSON_FILE_PATH, 'r') as f:
    race_data = json.load(f)

events_by_video = defaultdict(list)
for event in race_data["events"]:
    events_by_video[event["video_id"]].append(event)

for video_id in events_by_video:
    events_by_video[video_id].sort(key=lambda x: time_to_sec(x["start_time"]))

with open(OUTPUT_METADATA_FILE, 'a') as meta_file:
    
    for video_id, events in events_by_video.items():
        video_path = race_data["videos"][video_id]
        print(f"Opening video: {video_path} ({video_id})")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Warning: Could not open {video_path}. Skipping...")
            continue
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames_to_skip = int(SECONDS_BETWEEN_FRAMES * fps) - 1

        for event in events:
            base_sec = time_to_sec(event["start_time"])
            start_frame_number = int(base_sec * fps)
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_number)
            
            for i in range(FRAMES_TO_EXTRACT):
                target_sec = base_sec + (i * SECONDS_BETWEEN_FRAMES)
                
                success, frame = cap.read()
                
                if success:
                    base_filename = f"{video_id}_{target_sec}"
                    image_path = f"{RAW_FRAMES_DIR}/{base_filename}.jpg"
                    cv2.imwrite(image_path, frame)
                    
                    frame_meta = {k: v for k, v in event.items() if k in ("video_id", "gender", "stroke")}
                    frame_meta["filename"] = f"{base_filename}.jpg"
                    meta_file.write(json.dumps(frame_meta) + "\n")
                
                if i < FRAMES_TO_EXTRACT - 1:
                    for _ in range(frames_to_skip):
                        cap.grab()
                        
        cap.release()