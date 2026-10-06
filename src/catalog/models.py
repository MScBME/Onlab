from dataclasses import dataclass, field
from typing import Optional

from src.utils.timefmt import parse_time

DEFAULT_LANE_LENGTH_M = 25.0


@dataclass
class Lane:
    id: int
    coordinates: list  # 4 x [x, y] in the project's corner convention (see src/video/lane_warp.py)
    length_m: float = DEFAULT_LANE_LENGTH_M

    @classmethod
    def from_dict(cls, data: dict) -> "Lane":
        return cls(
            id=int(data["id"]),
            coordinates=[list(p) for p in data["coordinates"]],
            length_m=float(data.get("length_m", DEFAULT_LANE_LENGTH_M)),
        )

    def to_dict(self) -> dict:
        return {"id": self.id, "coordinates": self.coordinates, "length_m": self.length_m}


@dataclass
class Video:
    id: str
    path: str  # relative to data/raw
    lanes: list
    extra: dict = field(default_factory=dict)  # unknown keys, preserved on save

    @classmethod
    def from_dict(cls, data: dict) -> "Video":
        extra = {k: v for k, v in data.items() if k not in ("id", "path", "lanes")}
        return cls(
            id=data["id"],
            path=data["path"],
            lanes=[Lane.from_dict(lane) for lane in data.get("lanes", [])],
            extra=extra,
        )

    def to_dict(self) -> dict:
        return {"id": self.id, "path": self.path, "lanes": [lane.to_dict() for lane in self.lanes], **self.extra}

    def lane(self, lane_id: int) -> Lane:
        for lane in self.lanes:
            if lane.id == lane_id:
                return lane
        raise KeyError(f"Lane {lane_id} not found for video '{self.id}'")


@dataclass
class Clip:
    id: str
    video_id: str
    lane: int
    start_time: str
    end_time: str
    description: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict) -> "Clip":
        return cls(
            id=data["id"],
            video_id=data["video_id"],
            lane=int(data["lane"]),
            start_time=data.get("start_time", "00:00:00"),
            end_time=data["end_time"],
            description=data.get("description"),
        )

    def to_dict(self) -> dict:
        data = {
            "id": self.id,
            "video_id": self.video_id,
            "lane": self.lane,
            "start_time": self.start_time,
            "end_time": self.end_time,
        }
        if self.description:
            data["description"] = self.description
        return data

    @property
    def start_sec(self) -> float:
        return parse_time(self.start_time)

    @property
    def end_sec(self) -> float:
        return parse_time(self.end_time)
