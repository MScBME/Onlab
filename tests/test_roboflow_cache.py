"""The annotation cache is reused only for the Roboflow project and version configured in .env."""
import json

import pytest

roboflow_client = pytest.importorskip("research.training.roboflow_client")

SOURCE = {"workspace": "ws", "project": "swimmer_tracker_eventid_v2", "version": 3}


class FakeRoboflow:
    downloads = []

    def __init__(self, api_key):
        pass

    def workspace(self, _name):
        return self

    def project(self, _name):
        return self

    def version(self, number):
        self.number = number
        return self

    def download(self, _fmt, location):
        FakeRoboflow.downloads.append(self.number)
        images = __import__("pathlib").Path(location) / "train" / "images"
        images.mkdir(parents=True)
        (images / "d2du_00_00_01_000_jpg.rf.abc123.jpg").write_bytes(b"new")
        return type("Dataset", (), {"location": location})()


@pytest.fixture
def setup(tmp_path, monkeypatch):
    FakeRoboflow.downloads = []
    monkeypatch.setattr(roboflow_client, "Roboflow", FakeRoboflow)
    monkeypatch.setenv("ROBOFLOW_API_KEY", "key")
    target = tmp_path / "annotated"
    return target, monkeypatch


def configure(monkeypatch, **changes):
    monkeypatch.setattr(roboflow_client, "configured_source", lambda: {**SOURCE, **changes})


def test_downloads_and_records_the_source(setup):
    target, monkeypatch = setup
    configure(monkeypatch)
    assert roboflow_client.download_dataset(target) == SOURCE
    assert (target / "d2du_00_00_01_000.jpg").exists()
    assert json.loads((target / ".source.json").read_text()) == SOURCE


def test_reuses_the_cache_for_the_same_source(setup):
    target, monkeypatch = setup
    configure(monkeypatch)
    roboflow_client.download_dataset(target)
    roboflow_client.download_dataset(target)
    assert FakeRoboflow.downloads == [3]


def test_downloads_again_when_the_version_changes(setup):
    target, monkeypatch = setup
    configure(monkeypatch)
    roboflow_client.download_dataset(target)
    configure(monkeypatch, version=4)
    roboflow_client.download_dataset(target)
    assert FakeRoboflow.downloads == [3, 4]
    assert roboflow_client.cached_source(target)["version"] == 4


def test_a_cache_without_a_source_record_is_not_trusted(setup):
    target, monkeypatch = setup
    configure(monkeypatch)
    target.mkdir()
    (target / "old.jpg").write_bytes(b"old")
    roboflow_client.download_dataset(target)
    assert FakeRoboflow.downloads == [3]
    assert not (target / "old.jpg").exists()
