# Purah

Snap! Purah will take Snapshots/Clips of longer streams

Purah is a local, AI-powered automation script designed to turn long programming streams into "YouTube Short" style clips. Optimized for Apple Silicon (M1/M2/M3) using `whisper.cpp` for hardware-accelerated transcription and a local LLM (via LM Studio) for content analysis.

## What does it do?

This script acts as a fully automated clipping suite that runs on your Mac Mini. It follows a 5-step process to take your vidoes and extract the goods!

1.  **Watcher:** Monitors a specific input folder for new video files (e.g., from OBS or `rsync`).
2.  **The Ear (Transcription):** Uses `whisper-cli` (Core ML optimized) to generate a timestamped transcript of the video.
3.  **The Brain (Analysis):** Sends the transcript to a local LLM (running in LM Studio, e.g., Qwen3-Coder) to identify "Knowledge Nuggets"—distinct technical concepts, debugging moments, or architectural explanations, or just fun moments from the stream.
4.  **The Cutter:** Uses `ffmpeg` to losslessly cut these segments into standalone video files and generates a `summary.md` with titles and descriptions.
5.  **Offload:** Automatically moves the original input and output files to a new location (ex: Remote storage), then wipes the input folder to wait for a new video.

---

## How to use

The script can run in **Watcher Mode** (for automation) or **Manual Mode** (for specific files).

### 1. Watcher Mode (Recommended)
Run this in the background. It will wait for files to arrive in your input folder.

```bash
# Basic usage (uses defaults defined in script)
python3 purah.py --watch

# Custom input folder and remote offload target
python3 purah.py --watch -i ~/Desktop/DropZone -o /Volumes/MyNAS/Stream_Archives

```

### 2. Manual Mode

Process a single existing video file immediately.

```bash
# Process a single file
python3 purah.py --file ~/Movies/stream_recording.mp4

# Run ONLY the transcription step (useful for testing)
python3 purah.py --file ~/Movies/stream.mp4 --step transcribe

# Run ONLY the analysis step (if transcript already exists)
python3 purah.py --file ~/Movies/stream.mp4 --step analyze

```

### Configuration Flags

| Flag | Description | Default |
| --- | --- | --- |
| `-i`, `--input` | Directory to watch for new files | `/input` |
| `-o`, `--output` | Remote directory to offload finished files | `/output` |
| `--lm-url` | URL of your local LLM server | `http://localhost:1234/v1/chat/completions` |
| `--lm-model` | Model ID string to pass to API | `qwen3-coder-30b` |
| `--step` | Specific step to run (Manual Mode only) | `all` (Options: `transcribe`, `analyze`, `cut`) |

---

## Installation

### 1. System Dependencies

You need **FFmpeg** for video processing and **Whisper.cpp** for transcription.

```bash
# Install FFmpeg and Whisper CLI via Homebrew
brew install ffmpeg whisper-cpp

```

### 2. Download the Whisper Model

We use the `large-v3` model for maximum technical accuracy.

```bash
mkdir -p ~/models/whisper
cd ~/models/whisper
curl -L -o ggml-large-v3.bin [https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin](https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin)

```

### 3. Python Requirements

Install the necessary libraries for file watching and API requests.

```bash
pip install watchdog requests

```

### 4. LM Studio Setup

1. Download and install [LM Studio](https://lmstudio.ai/).
2. Load a coding-capable model (Recommended: **Qwen2.5-Coder-32B** or **DeepSeek-Coder-V2**).
3. Go to the **Local Server** tab (the `<->` icon).
4. **Start Server** on port `1234`.

### 5. Final Configuration

Open `purah.py` and ensure the `WHISPER_PATH` matches your install location.
You can find your path by running `which whisper-cli`.

```python
# purah.py

# Update this line if your path is different
WHISPER_PATH = "/opt/homebrew/bin/whisper-cli"
```
```
