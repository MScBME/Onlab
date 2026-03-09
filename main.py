import cv2
from ultralytics import YOLO

# 1. Load a pre-trained YOLO model
# YOLOv8n is the 'nano' version: fast and great for testing
model = YOLO('yolov8n.pt') 

# 2. Open the video file
video_path = "res/csik_d2du.mp4"
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # 3. The Magic: Detect AND Track in one line
    # 'persist=True' tells the model to remember IDs across frames
    # 'tracker="bytetrack.yaml"' uses the ByteTrack algorithm (which includes a Kalman filter)
    # 'classes=[0]' tells YOLO to ONLY look for the "person" class
    results = model.track(frame, persist=True, tracker="bytetrack.yaml", classes=[0])

    # 4. Draw the bounding boxes and tracking IDs on the frame
    # results[0].plot() automatically draws the boxes and IDs for you
    annotated_frame = results[0].plot()

    # 5. Display the output
    cv2.imshow("Tracking by Detection", annotated_frame)

    # Press 'q' to quit early
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()