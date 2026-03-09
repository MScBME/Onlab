import cv2
import numpy as np

class SwimmerTracker:
    def __init__(self, lane_points):
        self.tracker = cv2.TrackerCSRT_create()
        self.initialized = False
        
        self.lane_points = np.array(lane_points, np.int32).reshape((-1, 1, 2))
        self.lane_x, self.lane_y, self.lane_w, self.lane_h = cv2.boundingRect(self.lane_points)
        self.offset_points = self.lane_points - [self.lane_x, self.lane_y]
        
        self.mask = np.zeros((self.lane_h, self.lane_w, 3), dtype=np.uint8)
        cv2.fillPoly(self.mask, [self.offset_points], (255, 255, 255))

    def _get_masked_crop(self, frame):
        cropped = frame[self.lane_y : self.lane_y + self.lane_h, self.lane_x : self.lane_x + self.lane_w]
        return cv2.bitwise_and(cropped, self.mask)

    def initialize(self, frame):
        masked_frame = self._get_masked_crop(frame)
        bbox = cv2.selectROI("Select swimmer", masked_frame, False)
        self.tracker.init(masked_frame, bbox)
        self.initialized = True

    def update(self, frame):
        masked_frame = self._get_masked_crop(frame)
        success, bbox = self.tracker.update(masked_frame)

        if not success:
            return None

        x, y, w, h = bbox
        
        real_x = x + self.lane_x
        real_y = y + self.lane_y
        
        cx = real_x + w / 2
        cy = real_y + h / 2

        return cx, cy