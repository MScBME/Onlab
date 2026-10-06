import queue
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.detection.detector import SwimmerDetector
from src.tracking.kalman_filter import SwimmerKalmanFilter1D
from src.training.lane_processor import DST_PTS, LANE_H, LANE_W
from src.training.metadata import load_video_catalog
from src.utils.config import load_json, time_to_seconds
from src.utils.paths import CLIPS_JSON, MODELS_DIR, RAW_VIDEO_DIR
from src.video.loader import VideoLoader

POOL_LENGTH_M = 12.5


class VideoWorker(threading.Thread):
    def __init__(self, clip, model_path, conf, lane_coords, out_queue):
        super().__init__(daemon=True)
        self.clip = clip
        self.lane_coords = lane_coords
        self.queue = out_queue
        self.running = True

        self.active_model_path = model_path
        self.pending_model_path = None
        self.conf = conf

    def set_model(self, model_path):
        """Menet közbeni modellváltás kérése (Hot-swap)."""
        self.pending_model_path = model_path

    def set_conf(self, conf):
        self.conf = conf

    def stop(self):
        self.running = False

    def run(self):
        matrix = cv2.getPerspectiveTransform(self.lane_coords, DST_PTS)
        matrix_inv = cv2.getPerspectiveTransform(DST_PTS, self.lane_coords)
        pts_lane = self.lane_coords.astype(np.int32).reshape((-1, 1, 2))

        detector = SwimmerDetector(str(self.active_model_path), confidence=self.conf)
        video_cat = load_video_catalog()
        vid_path = RAW_VIDEO_DIR / video_cat.get_video(self.clip["video_id"])["path"]
        start_s = time_to_seconds(self.clip.get("start_time", "00:00:00"))
        end_s = time_to_seconds(self.clip.get("end_time"))
        loader = VideoLoader(str(vid_path), start_sec=start_s, end_sec=end_s)

        kf = None
        t_prev = time.time()
        fps = 30.0

        for _, ts, frame in loader.read_frames():
            if not self.running:
                break
            loop_start = time.time()

            # Modellváltás menet közben
            if self.pending_model_path:
                try:
                    detector = SwimmerDetector(str(self.pending_model_path), confidence=self.conf)
                    self.active_model_path = self.pending_model_path
                except Exception as e:
                    print(f"Modellváltási hiba: {e}")
                self.pending_model_path = None

            detector.confidence = self.conf
            warped = cv2.warpPerspective(frame, matrix, (LANE_W, LANE_H))
            det = detector.detect(warped)

            speed, pos_m = 0.0, 0.0
            status = "KERESÉS"

            if det:
                (cx, cy), (x1, y1, x2, y2), conf_val = det
                pos_m = cy * (POOL_LENGTH_M / LANE_H)
                if kf is None:
                    kf = SwimmerKalmanFilter1D(pos_m, ts)
                else:
                    kf.predict(ts)
                    kf.update(pos_m)

                status = f"DETEKTÁLVA ({conf_val:.2f})"
                cv2.rectangle(warped, (x1, y1), (x2, y2), (0, 255, 0), 2)
                pt_orig = cv2.perspectiveTransform(np.array([[[cx, cy]]], dtype=np.float32), matrix_inv)[0][0]
                cv2.circle(frame, (int(pt_orig[0]), int(pt_orig[1])), 7, (0, 255, 0), -1)

            elif kf is not None:
                pos_m, _ = kf.predict(ts)
                status = "VÍZ ALATT (Kalman)"
                pred_cy = np.clip(pos_m * (LANE_H / POOL_LENGTH_M), 0, LANE_H)
                pt_pred = cv2.perspectiveTransform(np.array([[[LANE_W / 2, pred_cy]]], dtype=np.float32), matrix_inv)[0][0]
                cv2.circle(frame, (int(pt_pred[0]), int(pt_pred[1])), 6, (0, 255, 255), -1)

            if kf:
                speed = kf.get_speed()

            cv2.polylines(frame, [pts_lane], isClosed=True, color=(255, 120, 0), thickness=2)

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-4))
            t_prev = now

            data = (frame, warped, speed, pos_m, fps, status)
            if self.queue.full():
                try: self.queue.get_nowait()
                except queue.Empty: pass
            self.queue.put(data)

            # Időzítés a valós videó-FPS-hez
            time.sleep(max(0.001, (1.0 / (loader.fps or 30.0)) - (time.time() - loop_start)))

        self.queue.put(None)  # Vége jelzés


class SwimmerUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Swimmer Detector")
        self.root.geometry("1360x780")
        self.root.configure(bg="#1a1a1a")

        self.clips = load_json(CLIPS_JSON).get("clips", [])
        self.models = list(MODELS_DIR.glob("*.pt"))
        self.catalog = load_video_catalog()
        self.queue = queue.Queue(maxsize=1)
        self.worker = None

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_ui(self):
        # 1. Felső vezérlősáv
        top = tk.Frame(self.root, bg="#262626", pady=6)
        top.pack(fill=tk.X)

        # Klip választó automatikusan méretezett szélességgel
        clip_ids = [c["id"] for c in self.clips]
        clip_w = max([len(cid) for cid in clip_ids] + [22]) + 2

        tk.Label(top, text="Klip:", fg="#ddd", bg="#262626").pack(side=tk.LEFT, padx=(10, 2))
        self.clip_cb = ttk.Combobox(top, values=clip_ids, state="readonly", width=clip_w)
        self.clip_cb.pack(side=tk.LEFT, padx=5)
        if self.clips: self.clip_cb.current(0)

        # Modell választó automatikusan méretezett szélességgel
        model_names = [m.name for m in self.models]
        model_w = max([len(mname) for mname in model_names] + [24]) + 2

        tk.Label(top, text="Modell:", fg="#ddd", bg="#262626").pack(side=tk.LEFT, padx=(10, 2))
        self.model_cb = ttk.Combobox(top, values=model_names, state="readonly", width=model_w)
        self.model_cb.pack(side=tk.LEFT, padx=5)
        if self.models: self.model_cb.current(0)
        self.model_cb.bind("<<ComboboxSelected>>", self._on_model_change)

        # Konfidencia
        tk.Label(top, text="Conf:", fg="#ddd", bg="#262626").pack(side=tk.LEFT, padx=(10, 2))
        self.conf_scale = tk.Scale(top, from_=0.1, to=0.9, resolution=0.05, orient=tk.HORIZONTAL,
                                   bg="#262626", fg="white", highlightthickness=0, command=self._on_conf_change)
        self.conf_scale.set(0.4)
        self.conf_scale.pack(side=tk.LEFT, padx=5)

        self.btn_start = tk.Button(top, text="▶ Start", bg="#2e7d32", fg="white", font=("Segoe UI", 9, "bold"), padx=10, command=self.start)
        self.btn_start.pack(side=tk.LEFT, padx=8)
        tk.Button(top, text="■ Stop", bg="#c62828", fg="white", font=("Segoe UI", 9, "bold"), padx=10, command=self.stop).pack(side=tk.LEFT, padx=4)

        # 2. Alsó telemetria Dashboard (fixen rögzítve, nem tűnik el)
        bottom = tk.Frame(self.root, bg="#202020", pady=8)
        bottom.pack(side=tk.BOTTOM, fill=tk.X)

        self.lbl_speed = tk.Label(bottom, text="0.00 m/s", font=("Consolas", 24, "bold"), fg="#00e676", bg="#202020")
        self.lbl_speed.pack(side=tk.LEFT, padx=20)

        self.lbl_status = tk.Label(bottom, text="KÉSZENLÉT", font=("Segoe UI", 11, "bold"), fg="#aaa", bg="#202020")
        self.lbl_status.pack(side=tk.LEFT, padx=15)

        self.lbl_info = tk.Label(bottom, text="Pozíció: 0.0 m | FPS: 0.0", font=("Segoe UI", 11), fg="#888", bg="#202020")
        self.lbl_info.pack(side=tk.RIGHT, padx=20)

        # 3. Középső megjelenítő felület (Canvas alapú, nincs átméretezési fagyás)
        center = tk.Frame(self.root, bg="#1a1a1a")
        center.pack(expand=True, fill=tk.BOTH, padx=8, pady=4)

        self.can_vid = tk.Canvas(center, bg="#0d0d0d", highlightthickness=0)
        self.can_vid.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=(0, 4))
        self.can_warp = tk.Canvas(center, bg="#0d0d0d", highlightthickness=0, width=240)
        self.can_warp.pack(side=tk.RIGHT, fill=tk.Y)

    def _on_model_change(self, _=None):
        if self.worker and self.worker.is_alive():
            self.worker.set_model(MODELS_DIR / self.model_cb.get())

    def _on_conf_change(self, val):
        if self.worker and self.worker.is_alive():
            self.worker.set_conf(float(val))

    def start(self):
        self.stop()
        clip = next((c for c in self.clips if c["id"] == self.clip_cb.get()), None)
        if not clip: return

        lane_coords = np.array(self.catalog.get_lane_coordinates(clip["video_id"], clip.get("lane")), dtype=np.float32)
        model_path = MODELS_DIR / self.model_cb.get()

        self.worker = VideoWorker(clip, model_path, float(self.conf_scale.get()), lane_coords, self.queue)
        self.worker.start()
        self.btn_start.configure(state=tk.DISABLED)
        self.root.after(20, self._update)

    def stop(self):
        if self.worker and self.worker.is_alive():
            self.worker.stop()
            self.worker = None
        self.btn_start.configure(state=tk.NORMAL)
        self.lbl_status.configure(text="LEÁLLÍTVA", fg="#888")

    def _update(self):
        if not self.worker: return

        try:
            item = self.queue.get_nowait()
        except queue.Empty:
            self.root.after(20, self._update)
            return

        if item is None:
            self.stop()
            self.lbl_status.configure(text="VÉGE", fg="#00e676")
            return

        frame, warped, spd, pos, fps, status = item

        # Képek renderelése a Canvas-re
        self._draw_canvas(self.can_vid, frame)
        self._draw_canvas(self.can_warp, warped)

        # Telemetria frissítése
        self.lbl_speed.configure(text=f"{spd:.2f} m/s")
        self.lbl_status.configure(text=status, fg="#00e676" if "DETEKT" in status else "#ffd600" if "VÍZ" in status else "#ff5252")
        self.lbl_info.configure(text=f"Pozíció: {pos:.2f} m | FPS: {fps:.1f}")

        self.root.after(20, self._update)

    def _draw_canvas(self, canvas, cv_img):
        cw, ch = canvas.winfo_width(), canvas.winfo_height()
        if cw < 10 or ch < 10: return

        h, w = cv_img.shape[:2]
        scale = min(cw / w, ch / h)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))

        rgb = cv2.cvtColor(cv2.resize(cv_img, (nw, nh)), cv2.COLOR_BGR2RGB)
        img_tk = ImageTk.PhotoImage(image=Image.fromarray(rgb))
        canvas.img_tk = img_tk
        canvas.delete("all")
        canvas.create_image(cw // 2, ch // 2, image=img_tk, anchor=tk.CENTER)

    def on_close(self):
        self.stop()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = SwimmerUI(root)
    root.mainloop()