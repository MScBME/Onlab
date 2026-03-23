import cv2
import numpy as np
import json
import os

with open('../data/lane_coordinates.json', 'r') as f:
    lane_config = json.load(f)

INPUT_DIR = "1_raw_frames"
OUTPUT_DIR = "2_processed_dataset"
LANE_W, LANE_H = 256, 1024
DST_PTS = np.array([[0, 0], [LANE_W, 0], [LANE_W, LANE_H], [0, LANE_H]], dtype="float32")

if not os.path.exists(OUTPUT_DIR): 
    os.makedirs(OUTPUT_DIR)

for img_file in [f for f in os.listdir(INPUT_DIR) if f.endswith(".jpg")]:
    img = cv2.imread(os.path.join(INPUT_DIR, img_file))
    h_orig, w_orig = img.shape[:2]
    label_path = os.path.join(INPUT_DIR, img_file.replace(".jpg", ".txt"))

    if not os.path.exists(label_path): 
        continue

    with open(label_path, 'r') as f:
        labels = [line.strip().split() for line in f.readlines()]

    for lane in lane_config["lanes"]:
        if lane["id"] > 6:
            continue
        src_pts = np.array(lane["coordinates"], dtype="float32")
        M = cv2.getPerspectiveTransform(src_pts, DST_PTS)
        warped_lane = cv2.warpPerspective(img, M, (LANE_W, LANE_H))
        
        lane_labels = []
        for label in labels:
            cls, x_c, y_c, w_b, h_b = map(float, label)
            px_x, px_y = x_c * w_orig, y_c * h_orig
            
            if cv2.pointPolygonTest(src_pts, (px_x, px_y), False) >= 0:
                pt_center = np.array([[[px_x, px_y]]], dtype="float32")
                new_center = cv2.perspectiveTransform(pt_center, M)[0][0]
                
                px_w, px_h = w_b * w_orig, h_b * h_orig
                pt_edge = np.array([[[px_x + px_w/2, px_y + px_h/2]]], dtype="float32")
                new_edge = cv2.perspectiveTransform(pt_edge, M)[0][0]

                new_x = new_center[0] / LANE_W
                new_y = new_center[1] / LANE_H
                
                new_wb = (abs(new_edge[0] - new_center[0]) * 2) / LANE_W
                new_hb = (abs(new_edge[1] - new_center[1]) * 2) / LANE_H
                
                new_wb = min(new_wb, 1.0)
                new_hb = min(new_hb, 1.0)
                
                lane_labels.append(f"0 {new_x:.6f} {new_y:.6f} {new_wb:.6f} {new_hb:.6f}")

        if lane_labels:
            base_name = os.path.splitext(img_file)[0]
            out_name = f"{base_name}_L{lane['id']}"
            cv2.imwrite(f"{OUTPUT_DIR}/{out_name}.jpg", warped_lane)
            with open(f"{OUTPUT_DIR}/{out_name}.txt", "w") as f_out:
                f_out.write("\n".join(lane_labels))
