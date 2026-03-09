import cv2


class SwimmerTracker:
    def __init__(self):
        self.tracker = cv2.TrackerCSRT_create()
        self.initialized = False

    def initialize(self, frame):
        bbox = cv2.selectROI("Select swimmer", frame, False)
        self.tracker.init(frame, bbox)
        self.initialized = True

    def update(self, frame):
        success, bbox = self.tracker.update(frame)

        if not success:
            return None

        x, y, w, h = bbox
        cx = x + w / 2
        cy = y + h / 2

        return cx, cy