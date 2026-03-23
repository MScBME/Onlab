import cv2
import os

IMG_DIR = "data/2_processed_dataset"
VERIFY_DIR = "data/3_visualization"
if not os.path.exists(VERIFY_DIR): os.makedirs(VERIFY_DIR)

for file in os.listdir(IMG_DIR):
    if file.endswith(".jpg"):
        img = cv2.imread(os.path.join(IMG_DIR, file))
        h, w, _ = img.shape
        label_path = os.path.join(IMG_DIR, file.replace(".jpg", ".txt"))
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    _, x_c, y_c, w_b, h_b = map(float, line.split())
                    
                    x1 = int((x_c - w_b/2) * w)
                    y1 = int((y_c - h_b/2) * h)
                    x2 = int((x_c + w_b/2) * w)
                    y2 = int((y_c + h_b/2) * h)
                    
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            cv2.imwrite(os.path.join(VERIFY_DIR, file), img)

print("Verification images saved in '3_verification' folder.")
