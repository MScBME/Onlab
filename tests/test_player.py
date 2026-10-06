import threading
import time

import cv2
import numpy as np
import pytest

from src.analysis.session import AnalysisParams
from src.playback.player import Player, PlayerListener, PlayerState

FPS = 30
W, H = 320, 180
LANE = ((0, 120), (10, 100), (300, 95), (315, 115))


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    path = tmp_path_factory.mktemp("video") / "synthetic.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for i in range(FPS * 2):
        frame = np.full((H, W, 3), 40, np.uint8)
        cv2.circle(frame, (20 + i * 4, 108), 5, (255, 255, 255), -1)
        writer.write(frame)
    writer.release()
    return path


class FakeDetector:
    def __init__(self, _model_path):
        self.confidence = None
        self.calls = 0

    def detect(self, _warped):
        self.calls += 1
        cy = 100.0 + 8 * self.calls
        return (128.0, cy), (100, int(cy) - 10, 156, int(cy) + 10), 0.8


class Recorder(PlayerListener):
    def __init__(self):
        self.states, self.frames, self.summaries, self.infos = [], [], [], []
        self.event = threading.Event()

    def on_opened(self, info):
        self.infos.append(info)

    def on_frame(self, result):
        self.frames.append(result)
        self.event.set()

    def on_state(self, state, message):
        self.states.append(state)
        self.event.set()

    def on_summary(self, summary):
        self.summaries.append(summary)
        self.event.set()


def wait_for(predicate, timeout=10.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.01)
    return False


@pytest.fixture
def player():
    recorder = Recorder()
    p = Player(recorder, detector_factory=FakeDetector)
    yield p, recorder
    p.close()


def params(video, run_id=1, start=0.2, end=0.7):
    return AnalysisParams(video_path=video, lane_id=1, lane_coords=LANE, lane_length_m=25.0, start_sec=start,
                          end_sec=end, model_path=video.with_suffix(".pt"), confidence=0.4, run_id=run_id)


def test_open_and_seek_emit_preview_frames(player, video):
    p, rec = player
    p.open(video)
    p.set_preview_lane(LANE)
    assert wait_for(lambda: rec.infos and rec.frames)
    assert rec.infos[0].width == W and rec.infos[0].fps == pytest.approx(FPS)
    p.seek(1.0)
    assert wait_for(lambda: any(abs(f.timestamp - 1.0) < 0.02 for f in rec.frames))
    last = rec.frames[-1]
    assert last.run_id is None and last.status is None and last.warped.shape == (1024, 256, 3)


def test_analysis_runs_to_the_end_of_the_range(player, video):
    p, rec = player
    p.open(video)
    p.start_analysis(params(video, run_id=7))
    assert wait_for(lambda: rec.summaries)
    summary = rec.summaries[0]
    assert summary.completed and summary.run_id == 7
    assert summary.frames_processed == pytest.approx(0.5 * FPS, abs=2)
    assert PlayerState.LOADING_MODEL in rec.states and rec.states[-1] == PlayerState.FINISHED
    analysed = [f for f in rec.frames if f.run_id == 7]
    assert analysed and all(0.2 - 1e-6 <= f.timestamp <= 0.7 + 1e-6 for f in analysed)


def test_pause_resume_and_stop(player, video):
    p, rec = player
    p.open(video)
    p.start_analysis(params(video, start=0.0, end=1.9))
    assert wait_for(lambda: PlayerState.ANALYZING in rec.states)
    p.pause()
    assert wait_for(lambda: rec.states[-1] == PlayerState.PAUSED)
    count = len(rec.frames)
    time.sleep(0.2)
    assert len(rec.frames) == count  # nothing is decoded while paused
    p.resume()
    assert wait_for(lambda: len(rec.frames) > count)
    p.stop()
    assert wait_for(lambda: rec.summaries)
    assert rec.summaries[0].completed is False
    assert wait_for(lambda: rec.states[-1] == PlayerState.IDLE)


def test_restart_discards_the_previous_run(player, video):
    p, rec = player
    p.open(video)
    p.start_analysis(params(video, run_id=1, start=0.0, end=1.9))
    assert wait_for(lambda: any(f.run_id == 1 for f in rec.frames))
    p.start_analysis(params(video, run_id=2, start=0.2, end=0.5))
    assert wait_for(lambda: rec.summaries)
    assert [s.run_id for s in rec.summaries] == [2]


def test_release_and_errors_keep_the_thread_alive(player, video, tmp_path):
    p, rec = player
    p.open(tmp_path / "missing.mp4")
    assert wait_for(lambda: rec.states and rec.states[-1] == PlayerState.ERROR)
    p.open(video)
    assert wait_for(lambda: rec.infos)
    p.release()
    assert wait_for(lambda: rec.states[-1] == PlayerState.IDLE)
