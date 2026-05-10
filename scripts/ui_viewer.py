import sys
import tkinter as tk
from tkinter import ttk
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageTk
from collections import deque

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.detection.detector import SwimmerDetector
from src.video.loader import VideoLoader
from src.utils.paths import CLIPS_JSON, MODELS_DIR, RAW_VIDEO_DIR
from src.utils.config import load_json, time_to_seconds
from src.training.lane_processor import DST_PTS, LANE_H, LANE_W
from src.training.metadata import load_video_catalog

POOL_LENGTH_M = 25.0

class SwimmerDetectionUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Swimmer Detector Pro")
        self.root.geometry("1400x900")
        self.root.configure(bg="#1e1e1e")

        self.raw_pos_history = deque(maxlen=10)
        self.speed_history = deque(maxlen=30)
        self.pos_history = deque(maxlen=2)
        
        self.is_playing = False
        self.current_loader = None
        self._gen = None
        
        self.video_catalog = load_video_catalog()
        self.clips_data = load_json(CLIPS_JSON).get("clips", [])
        self.available_models = list(MODELS_DIR.glob("*.pt"))

        self.setup_ui()

    def setup_ui(self):
        controls = ttk.Frame(self.root, padding="10")
        controls.pack(side=tk.TOP, fill=tk.X)

        self.clip_combo = ttk.Combobox(controls, values=[c['id'] for c in self.clips_data], width=25)
        self.clip_combo.pack(side=tk.LEFT, padx=5)
        if self.clips_data: self.clip_combo.current(0)

        self.model_combo = ttk.Combobox(controls, values=[m.name for m in self.available_models], width=20)
        self.model_combo.pack(side=tk.LEFT, padx=5)
        if self.available_models: self.model_combo.current(0)

        self.conf_slider = ttk.Scale(controls, from_=0.05, to=1.0, value=0.5, orient=tk.HORIZONTAL)
        self.conf_slider.pack(side=tk.LEFT, padx=5)

        tk.Button(controls, text="▶ START", bg="#28a745", fg="white", command=self.start_inference).pack(side=tk.LEFT, padx=10)
        tk.Button(controls, text="■ STOP", bg="#dc3545", fg="white", command=self.stop_inference).pack(side=tk.LEFT, padx=5)

        dash = ttk.Frame(self.root, padding="15")
        dash.pack(side=tk.BOTTOM, fill=tk.X)

        ttk.Label(dash, text="LIVE SPEED", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=10)
        self.speed_bar = ttk.Progressbar(dash, orient=tk.HORIZONTAL, mode='determinate')
        self.speed_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        self.speed_text = ttk.Label(dash, text="0.00 m/s", font=("Courier", 18, "bold"), foreground="#00ff00")
        self.speed_text.pack(side=tk.RIGHT, padx=20)

        self.paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.paned.pack(side=tk.TOP, expand=True, fill=tk.BOTH, padx=10, pady=5)

        self.video_container = ttk.Frame(self.paned)
        self.warped_container = ttk.Frame(self.paned)
        self.paned.add(self.video_container, weight=4)
        self.paned.add(self.warped_container, weight=1)

        self.video_label = ttk.Label(self.video_container)
        self.video_label.pack(expand=True, fill=tk.BOTH)
        self.warped_label = ttk.Label(self.warped_container)
        self.warped_label.pack(expand=True, fill=tk.BOTH)

    def calculate_speed(self, current_cy, timestamp):
        self.raw_pos_history.append(current_cy)
        smoothed_cy = sum(self.raw_pos_history) / len(self.raw_pos_history)
        
        current_pos_m = smoothed_cy * (POOL_LENGTH_M / LANE_H)
        self.pos_history.append((timestamp, current_pos_m))
        
        if len(self.pos_history) < 2:
            return 0.0
            
        t_old, p_old = self.pos_history[0]
        t_new, p_new = self.pos_history[-1]
        dt = t_new - t_old
        
        if dt <= 0: return 0.0
        instant_speed = abs(p_new - p_old) / dt
        
        self.speed_history.append(instant_speed)
        smoothed_speed = sum(self.speed_history) / len(self.speed_history)
        
        return min(smoothed_speed, 2.5)

    def start_inference(self):
        self.stop_inference()
        clip_id, model_name = self.clip_combo.get(), self.model_combo.get()
        clip = next(c for c in self.clips_data if c['id'] == clip_id)
        
        self.current_detector = SwimmerDetector(str(MODELS_DIR / model_name), confidence=float(self.conf_slider.get()))
        video = self.video_catalog.get_video(clip["video_id"])
        self.current_loader = VideoLoader(str(RAW_VIDEO_DIR / video["path"]), 
                                         start_sec=time_to_seconds(clip.get("start_time", "00:00:00")), 
                                         end_sec=time_to_seconds(clip.get("end_time")))
        
        lane_id = clip.get("lane")
        self.lane_coords = np.array(self.video_catalog.get_lane_coordinates(clip["video_id"], lane_id), dtype=np.float32)
        self.matrix = cv2.getPerspectiveTransform(self.lane_coords, DST_PTS)
        self.matrix_inv = cv2.getPerspectiveTransform(DST_PTS, self.lane_coords)

        self._gen = self.current_loader.read_frames()
        self.is_playing = True
        self.process_frame()

    def stop_inference(self):
        self.is_playing = False
        self.speed_bar['value'] = 0
        self.speed_text.configure(text="0.00 m/s")
        self.raw_pos_history.clear()
        self.speed_history.clear()

    def process_frame(self):
        if not self.is_playing or self._gen is None: return

        try:
            _, timestamp, frame = next(self._gen)
            warped = cv2.warpPerspective(frame, self.matrix, (LANE_W, LANE_H))
            detection = self.current_detector.detect(warped)
            
            speed = 0.0
            if detection:
                (cx, cy), (x1, y1, x2, y2), _ = detection
                speed = self.calculate_speed(cy, timestamp)
                
                cv2.rectangle(warped, (x1, y1), (x2, y2), (0, 255, 0), 2)
                pt_orig = cv2.perspectiveTransform(np.array([[[cx, cy]]], dtype=np.float32), self.matrix_inv)[0][0]
                cv2.circle(frame, (int(pt_orig[0]), int(pt_orig[1])), 8, (0, 255, 0), -1)

            self.speed_bar['value'] = (speed / 2.5) * 100
            self.speed_text.configure(text=f"{speed:.2f} m/s")
            
            pts = self.lane_coords.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], isClosed=True, color=(255, 0, 0), thickness=2)

            self.update_canvas(frame, self.video_label)
            self.update_canvas(warped, self.warped_label)
            self.root.after(1, self.process_frame)
            
        except StopIteration:
            self.stop_inference()

    def update_canvas(self, frame, label):
        self.root.update_idletasks()
        win_w, win_h = label.winfo_width(), label.winfo_height()
        
        if win_w < 10 or win_h < 10: return 

        h, w = frame.shape[:2]
        ratio = min(win_w/w, win_h/h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        
        frame_resized = cv2.resize(frame, (new_w, new_h))
        img = Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        label.imgtk = imgtk
        label.configure(image=imgtk)

if __name__ == "__main__":
    root = tk.Tk()
    app = SwimmerDetectionUI(root)
    root.mainloop()