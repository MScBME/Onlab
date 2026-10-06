import random

import pytest

from src.training import lane_processor
from src.utils.config import read_json
from src.utils.paths import VIDEOS_JSON
from src.video import lane_warp
from src.video.lane_warp import normalize_corners, validate_corners

EXISTING_LANES = [
    (video["id"], lane["id"], lane["coordinates"])
    for video in read_json(VIDEOS_JSON)["videos"]
    for lane in video["lanes"]
]


@pytest.mark.parametrize("video_id, lane_id, coords", EXISTING_LANES)
def test_existing_lanes_are_already_normalized_and_valid(video_id, lane_id, coords):
    rng = random.Random(f"{video_id}-{lane_id}")
    for _ in range(12):
        shuffled = coords[:]
        rng.shuffle(shuffled)
        assert normalize_corners(shuffled) == coords
    assert validate_corners(coords) == []


def test_normalized_first_edge_is_the_left_lane_end():
    coords = normalize_corners([[1548, 761], [0, 774], [1487, 713], [90, 724]])
    assert coords == [[0, 774], [90, 724], [1487, 713], [1548, 761]]


def test_validate_rejects_concave_and_out_of_frame():
    concave = [[0, 0], [100, 40], [200, 0], [100, 100]]
    assert any("convex" in e for e in validate_corners(concave))
    lane = [[0, 774], [90, 724], [1487, 713], [1548, 761]]
    assert any("inside the video frame" in e for e in validate_corners(lane, frame_size=(1280, 720)))


def test_wrong_point_count():
    with pytest.raises(ValueError):
        normalize_corners([[0, 0], [1, 1], [2, 2]])
    assert validate_corners([[0, 0], [1, 1]]) == ["A lane needs exactly 4 corners."]


def test_training_pipeline_uses_the_same_warp_constants():
    assert lane_processor.LANE_W == lane_warp.LANE_W == 256
    assert lane_processor.LANE_H == lane_warp.LANE_H == 1024
    assert (lane_processor.DST_PTS == lane_warp.DST_PTS).all()
