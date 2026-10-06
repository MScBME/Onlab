"""Time string helpers shared by the CLI scripts and the desktop app.

The JSON metadata stores times as ``HH:MM:SS``, optionally with milliseconds (``HH:MM:SS.mmm``).
"""


def parse_time(text: str) -> float:
    """Parse ``HH:MM:SS[.fff]`` into seconds. Raises ValueError on malformed input."""
    parts = str(text).strip().split(":")
    if len(parts) != 3:
        raise ValueError(f"Time must be HH:MM:SS, got '{text}'")
    try:
        hours, minutes, seconds = int(parts[0]), int(parts[1]), float(parts[2])
    except ValueError:
        raise ValueError(f"Time must be HH:MM:SS, got '{text}'") from None
    if hours < 0 or not 0 <= minutes < 60 or not 0 <= seconds < 60:
        raise ValueError(f"Time out of range: '{text}'")
    return hours * 3600 + minutes * 60 + seconds


def format_time(seconds: float, always_ms: bool = False) -> str:
    """Format seconds as ``HH:MM:SS``; milliseconds are appended when non-zero or ``always_ms``."""
    total_ms = int(round(max(0.0, seconds) * 1000))
    hh, rem = divmod(total_ms, 3_600_000)
    mm, rem = divmod(rem, 60_000)
    ss, ms = divmod(rem, 1000)
    base = f"{hh:02d}:{mm:02d}:{ss:02d}"
    if ms or always_ms:
        return f"{base}.{ms:03d}"
    return base
