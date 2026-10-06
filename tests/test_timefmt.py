import pytest

from src.utils.timefmt import format_time, parse_time


@pytest.mark.parametrize("text, seconds", [
    ("00:00:00", 0.0),
    ("00:07:54", 474.0),
    ("01:12:41", 4361.0),
    ("00:00:41.5", 41.5),
    ("02:31:56.937", 9116.937),
])
def test_parse_time(text, seconds):
    assert parse_time(text) == pytest.approx(seconds)


@pytest.mark.parametrize("text", ["", "7:54", "aa:bb:cc", "00:60:00", "00:00:60", "-1:00:00"])
def test_parse_time_rejects_malformed(text):
    with pytest.raises(ValueError):
        parse_time(text)


def test_format_time_omits_zero_milliseconds():
    assert format_time(474.0) == "00:07:54"
    assert format_time(474.0, always_ms=True) == "00:07:54.000"
    assert format_time(474.0167) == "00:07:54.017"


@pytest.mark.parametrize("seconds", [0.0, 1.5, 474.017, 3599.999, 9116.937])
def test_round_trip(seconds):
    assert parse_time(format_time(seconds)) == pytest.approx(seconds, abs=5e-4)
