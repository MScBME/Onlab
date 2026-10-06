import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis.speed import filter_positions_kalman, smooth_and_cap_speed
from src.detection.detector import SwimmerDetector
from src.training.metadata import load_video_catalog
from src.utils.config import load_json, time_to_seconds
from src.utils.paths import CLIPS_JSON, MODELS_DIR, RAW_VIDEO_DIR, RUN_CONFIG_JSON
from src.video.lane_warp import DST_PTS, LANE_H, LANE_W
from src.video.loader import VideoLoader


def main():
    parser = argparse.ArgumentParser(description="Jelfeldolgozási szintek összehasonlítása: Nyers vs. Mozgóátlag vs. Kalman.")
    parser.add_argument("--clip", help="Klip ID")
    parser.add_argument("--model", help="Modell fájlneve")
    args = parser.parse_args()

    settings = load_json(RUN_CONFIG_JSON)
    clip_id = args.clip or settings.get("active_clip_id")
    model_name = args.model or settings.get("model_path", "models/swimmer_freestyle.pt")

    clips_data = load_json(CLIPS_JSON)
    clip = next((c for c in clips_data.get("clips", []) if c["id"] == clip_id), None)

    if not clip:
        print(f"Hiba: A klip '{clip_id}' nem található.")
        sys.exit(1)

    lane_id = clip.get("lane")
    video_catalog = load_video_catalog()
    video = video_catalog.get_video(clip["video_id"])
    video_path = RAW_VIDEO_DIR / video["path"]
    start_sec = time_to_seconds(clip.get("start_time", "00:00:00"))
    end_sec = time_to_seconds(clip.get("end_time"))

    lane_coords = np.array(video_catalog.get_lane_coordinates(clip["video_id"], lane_id), dtype=np.float32)
    lane_length_m = video_catalog.get_lane_length(clip["video_id"], lane_id)
    matrix = cv2.getPerspectiveTransform(lane_coords, DST_PTS)

    model_path = Path(model_name) if Path(model_name).is_absolute() else MODELS_DIR / Path(model_name).name

    print(f"Elemzés futtatása: {model_path.name} | Klip: {clip_id}...")
    detector = SwimmerDetector(str(model_path), confidence=0.4)
    loader = VideoLoader(str(video_path), start_sec=start_sec, end_sec=end_sec)

    raw_positions = []
    timestamps = []

    for _, ts, frame in loader.read_frames():
        warped = cv2.warpPerspective(frame, matrix, (LANE_W, LANE_H))
        detection = detector.detect(warped)
        if detection:
            (cx, cy), _, _ = detection
            raw_positions.append(cy * (lane_length_m / LANE_H))
            timestamps.append(ts)

    if len(raw_positions) < 15:
        print("Nincs elég detektálás az elemzéshez.")
        return

    # 1. Nyers sebesség (abs(dy/dt)) - Zajos pontok
    raw_pos = np.array(raw_positions)
    raw_ts = np.array(timestamps)
    dt = np.diff(raw_ts)
    dt[dt == 0] = 1e-6
    raw_v = np.abs(np.diff(raw_pos) / dt)
    raw_v_ts = raw_ts[1:]

    # 2. Hagyományos mozgóátlag (egyszerű 7 elemű konvolúció a nyers sebességen)
    moving_avg_v = np.convolve(raw_v, np.ones(7) / 7, mode="same")

    # 3. Új Kalman-szűrt sebesség
    kalman_ts, _, kalman_v = filter_positions_kalman(raw_positions, timestamps)
    final_ts, final_v = smooth_and_cap_speed(raw_positions, timestamps, speed_window_size=7)

    # --- ÁBRÁZOLÁS ---
    plt.figure(figsize=(15, 8))

    t0 = raw_ts[0]
    plt.plot(raw_v_ts - t0, raw_v, label="1. Nyers derivált (abs(dp/dt) - zajos)", color="red", alpha=0.3, linewidth=0.8)
    plt.plot(raw_v_ts - t0, moving_avg_v, label="2. Egyszerű mozgóátlag (fáziskésés, torzítás)", color="orange", linestyle="--", linewidth=1.8)
    plt.plot(kalman_ts - t0, kalman_v, label="3. Kalman-szűrt sebesség", color="blue", linewidth=2.0)
    plt.plot(final_ts - t0, final_v, label="4. Végleges simított profil (Karcsapások profilja)", color="green", linewidth=3.0)

    plt.title(f"Sebességbecslési eljárások összehasonlítása\nModell: {model_path.name} | Klip: {clip_id}", fontsize=13)
    plt.xlabel("Idő [s]", fontsize=11)
    plt.ylabel("Sebesség [m/s]", fontsize=11)
    plt.ylim(0, 3.5)

    plt.legend(loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.5)

    output_path = f"pipeline_analysis_{clip_id}.png"
    plt.savefig(output_path, dpi=300)
    print(f"✅ Ábra elmentve: {output_path}")
    plt.show()


if __name__ == "__main__":
    main()