from yt_dlp import YoutubeDL
from yt_dlp.postprocessor import FFmpegPostProcessor

# 1. MANUALLY SET THE FFMPEG PATH GLOBALLY
# This bypasses the check that causes your "Aborting" error
FFmpegPostProcessor._ffmpeg_location.set("C:/_tools/ffmpeg/ffmpeg-2026-03-01-git-862338fe31-full_build/bin/ffmpeg.exe")

url = "https://www.youtube.com/watch?v=pXxlpmlvcsc"

ydl_opts = {
    "format": "bestvideo+bestaudio/best",
    "outtmpl": "res/clip.%(ext)s",
    "overwrites": True,
    
    # Use download_ranges for efficient seeking (seconds)
    "download_ranges": lambda info_dict, ydl: [{
        'start_time': 10530, # 02:55:30
        'end_time': 10585,   # 02:56:25
    }],
    
    "force_keyframes_at_cuts": True,
}

with YoutubeDL(ydl_opts) as ydl:
    ydl.download([url])