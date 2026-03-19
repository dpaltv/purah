import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

DEFAULT_WATCH_FOLDER = PROJECT_ROOT / "watch"
DEFAULT_OUTPUT_FOLDER = PROJECT_ROOT / "output"
SHORTS_OUTPUT_FOLDER = DEFAULT_OUTPUT_FOLDER / "shorts"

BUFFER_SECONDS = 180

LM_STUDIO_BASE_URL = os.environ.get("LM_STUDIO_URL", "http://localhost:1234/v1")
LM_STUDIO_API_KEY = os.environ.get("LM_STUDIO_API_KEY", "lm-studio")
LM_STUDIO_MODEL = os.environ.get("LM_STUDIO_MODEL", "Phi-3-mini-128k-instruct-imatrix-smashed")

WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "base")

WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "auto")

SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm"}

SEGMENT_CATEGORIES = [
    "coding_idea",
    "funny_moment",
    "technical_detail",
    "demo",
    "tip",
]

TRANSCRIPT_WORD_WINDOW = 50

TRANSCRIPT_CHUNK_MINUTES = 600  # Disable chunking - 10 hours exceeds 5-hour video

TRANSCRIPT_CHUNK_OVERLAP_MINUTES = 5
