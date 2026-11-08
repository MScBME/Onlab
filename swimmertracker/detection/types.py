"""What a detector returns, and how one detection is picked from several.

A detector has a ``confidence`` attribute, the minimum score, and a ``detect(image) -> list[Detection]`` method
that returns boxes in the pixel coordinates of the image it was given.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float

    @property
    def center(self) -> tuple:
        return (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2

    @property
    def bbox(self) -> tuple:
        return self.x1, self.y1, self.x2, self.y2


def highest_confidence(detections: list) -> Optional[Detection]:
    return max(detections, key=lambda d: d.confidence, default=None)
