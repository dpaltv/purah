# Purah

Twitch stream shorts extractor that uses whisper.cpp (via faster-whisper) for transcription and local LLMs (via LM Studio) for analyzing content to find short-form video segments.

## Installation

```bash
pip install -e .
```

## Requirements

- Python 3.9+
- ffmpeg (system dependency)
- LM Studio running locally with a model loaded
- BUNGEE font installed (for ASS subtitle word highlighting)

## Usage

```bash
purah --help
purah run <video_file>              # Full pipeline + subtitles + burned videos
purah watch                        # Watch folder + full pipeline + subtitles
purah transcribe <video_file>     # Transcribe only
purah analyze <video_file>        # Analyze only
purah extract <video_file>        # Extract shorts
purah subtitles <video_file>       # Generate subtitle files (SRT, VTT, ASS)
purah chapters <video_file>       # Extract YouTube chapters
purah status
```

## Options

Each command supports:
- `-o, --output-dir` - Output directory
- `-v, --verbose` - Enable verbose logging

The `subtitles` command additionally supports:
- `-f, --format` - Subtitle format (srt, vtt, ass)
- `--burn` - Burn subtitles into video

## Output

```
output/
├── shorts/                              # Extracted short segments
│   └── shorts_HHMMSS_category_title.mp4
├── video.transcript.json                 # Whisper transcription
├── video.segments.json                # LLM analysis
├── video_with_subtitles.mp4             # Burned with ASS subs
├── video.sbv                         # YouTube upload ready
├── video.srt                         # Basic subtitles
├── video.vtt                         # Web subtitles
└── video.ass                         # Word highlighting
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LM_STUDIO_URL` | `http://localhost:1234/v1` | LLM API base URL |
| `LM_STUDIO_API_KEY` | `lm-studio` | LLM API key |
| `LM_STUDIO_MODEL` | `google/gemma-4-e4b` | Model name |
| `WHISPER_MODEL` | `small` | Whisper model size |
| `SUBTITLE_FONT_NAME` | `BUNGEE` | Subtitle font |