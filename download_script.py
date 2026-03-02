from yt_dlp import YoutubeDL
from yt_dlp.postprocessor import FFmpegPostProcessor

# 1. MANUALLY SET THE FFMPEG PATH GLOBALLY
# This bypasses the check that causes your "Aborting" error
FFmpegPostProcessor._ffmpeg_location.set("C:/ffmpeg/ffmpeg.exe")

url = "https://www.youtube.com/watch?v=pXxlpmlvcsc"

ydl_opts = {
    "format": "bestvideo+bestaudio/best",
    "outtmpl": "clip.%(ext)s",
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