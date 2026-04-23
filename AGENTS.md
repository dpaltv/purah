# AGENTS.md - Purah Development Guide

## Project Overview

Purah is a Twitch stream shorts extractor that uses whisper.cpp (via faster-whisper) for transcription and local LLMs (via LM Studio) for analyzing content to find short-form video segments.

## Project Structure

```
purah/
├── bin/purah          # Entry point script
├── purah/             # Main package
│   ├── __init__.py    # Public API exports
│   ├── cli.py         # CLI commands (click-based)
│   ├── config.py      # Configuration constants
│   ├── pipeline.py    # Main pipeline orchestration
│   ├── transcriber.py # Video transcription (faster-whisper)
│   ├── analyzer.py    # LLM-based segment analysis
│   ├── extractor.py   # Video segment extraction (ffmpeg)
│   ├── subtitler.py   # Subtitle generation (SRT, VTT, ASS with word highlighting)
│   └── watcher.py     # File system watcher
├── watch/             # Default folder for watching videos
└── output/            # Default output folder
```

## Build/Install Commands

```bash
# Install package in development mode
pip install -e .

# Install all dependencies
pip install -r requirements.txt

# Run the CLI
purah --help
purah run <video_file>              # Full pipeline + subtitles + burned videos
purah watch                        # Watch folder + full pipeline + subtitles
purah transcribe <video_file>     # Transcribe only
purah analyze <video_file>        # Analyze only
purah extract <video_file>        # Extract shorts + subtitles + burned videos
purah subtitles <video_file>       # Generate subtitle files (SRT, VTT, ASS)
purah chapters <video_file>       # Extract YouTube chapters
purah status
```

## Testing

This project uses **pytest** for testing.

```bash
# Run all tests
pytest

# Run tests with verbose output
pytest -v

# Run a single test file
pytest tests/test_analyzer.py

# Run a single test function
pytest tests/test_analyzer.py::test_parse_timestamp

# Run tests matching a pattern
pytest -k "test_parse"

# Run with coverage
pytest --cov=purah

# Run tests in parallel (requires pytest-xdist)
pytest -n auto
```

## Code Style Guidelines

### General

- Python 3.9+ required
- Use type hints for all function parameters and return values
- Use `Optional[T]` instead of `T | None` for broader compatibility
- Use absolute imports within the package: `from . import config`
- Use relative imports for sibling modules: `from .transcriber import Transcriber`

### Naming Conventions

| Element | Convention | Example |
|---------|------------|---------|
| Modules | snake_case | `my_module.py` |
| Classes | PascalCase | `class PipelineManager` |
| Functions/methods | snake_case | `def process_video()` |
| Variables | snake_case | `video_path` |
| Constants | UPPER_SNAKE_CASE | `DEFAULT_WATCH_FOLDER` |
| Type variables | PascalCase | `T = TypeVar('T')` |

### Import Organization

Standard library imports first, then third-party, then local:

```python
import logging       # stdlib
import sys           # stdlib
from pathlib import Path  # stdlib

import click         # third-party
from rich.console import Console  # third-party

from . import config          # local - package relative
from .transcriber import Transcriber  # local - sibling module
```

### Formatting

- 4 spaces for indentation (no tabs)
- Maximum line length: 88 characters (Black default)
- Use f-strings for string formatting
- Use single quotes for strings: `'hello'` not `"hello"`
- Add trailing commas in multi-line collections
- One import per line

```python
# Good
segments = [
    seg.get("text", "").strip()
    for seg in segments
    if seg.get("start", 0) >= start_seconds
]

# Bad - double quotes
segments = [
    seg.get("text", "").strip()
    for seg in segments
    if seg.get("start", 0) >= start_seconds
]
```

### Docstrings

- Use docstrings for public classes and functions
- Keep docstrings concise; the code should be self-explanatory
- Use Google-style docstrings:

```python
def transcribe(self, video_path: Path, output_path: Optional[Path] = None) -> dict:
    """Transcribe a video file using faster-whisper.

    Args:
        video_path: Path to the input video file.
        output_path: Optional path for the output JSON. Defaults to video.transcript.json.

    Returns:
        Dictionary containing the transcription data.

    Raises:
        FileNotFoundError: If the video file doesn't exist.
    """
```

### Type Hints

Always use type hints for function signatures:

```python
# Good
def process_video(self, video_path: Path) -> dict:
    pass

def analyze(
    self,
    transcript_data: dict,
    video_path: Path,
) -> dict:
    pass

# Bad
def process_video(self, video_path):
    pass
```

### Error Handling

- Raise descriptive exceptions with context
- Log errors before raising when appropriate
- Catch specific exceptions, not bare `except:`

```python
# Good
if not video_path.exists():
    raise FileNotFoundError(f"Video file not found: {video_path}")

try:
    result = pipeline.process_video(video_path)
except Exception as e:
    logger.error(f"Failed to process {video_path}: {e}")
    raise

# Bad
if not video_path.exists():
    raise Exception()
```

### Logging

- Use `logging.getLogger(__name__)` for module loggers
- Use `logger` as the variable name
- Use appropriate log levels: DEBUG for verbose info, INFO for progress, WARNING/ERROR for issues

```python
logger = logging.getLogger(__name__)

logger.info(f"Transcribing {video_path}...")
logger.debug(f"Loaded {len(segments)} segments")
logger.warning(f"Retrying ({attempt + 1}/{max_retries})")
logger.error(f"Failed to connect: {e}")
```

### Path Handling

- Use `pathlib.Path` for all file paths
- Always resolve paths: `Path(video_path).resolve()`
- Create directories with `mkdir(parents=True, exist_ok=True)`

```python
video_path = Path(video_path).resolve()
self.output_dir.mkdir(parents=True, exist_ok=True)
```

### CLI Commands (cli.py)

- Use `click` decorators for CLI
- Always call `setup_logging()` at the start of commands
- Use `console.print()` for rich-formatted output
- Use `@click.option` for flags and options

```python
@click.command()
@click.argument("file", type=click.Path(exists=True))
@click.option("--verbose", "-v", is_flag=True)
def run(file, verbose):
    """Run the full pipeline."""
    setup_logging(verbose)
    console.print(f"[bold cyan]Processing:[/bold cyan] {file}")
```

## Configuration (config.py)

All configuration is in `config.py`:
- Use `os.environ.get()` for environment variables with defaults
- Use `Path` for folder paths
- Use sets for collections like video extensions

## External Dependencies

| Package | Purpose |
|---------|---------|
| click | CLI framework |
| rich | Terminal output formatting |
| faster-whisper | Audio transcription |
| requests | HTTP client for LLM API |
| watchdog | File system monitoring |
| ffmpeg | Video manipulation (system dependency) |

## Adding New Features

1. Add CLI command in `cli.py`
2. Add business logic in appropriate module
3. Export public functions/classes in `__init__.py`
4. Add tests in `tests/` directory
5. Update this AGENTS.md if conventions change

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LM_STUDIO_URL` | `http://localhost:1234/v1` | LLM API base URL |
| `LM_STUDIO_API_KEY` | `lm-studio` | LLM API key |
| `LM_STUDIO_MODEL` | `google/gemma-4-e4b` | Model name |
| `WHISPER_MODEL` | `small` | Whisper model size (tiny/base/small/medium/large) |
| `WHISPER_DEVICE` | `auto` | Device for whisper (cpu/cuda) |
| `SUBTITLE_FONT_NAME` | `BUNGEE` | Font for subtitle rendering |
| `SUBTITLE_FONT_SIZE` | `20` | Font size for subtitles |

## Subtitle Features

The project supports generating subtitles in multiple formats:
- **SBV**: YouTube's native format - use this for uploading to YouTube
- **SRT**: Basic subtitles, maximum compatibility
- **VTT**: Web-native format with basic styling
- **ASS**: Advanced format with word highlighting and custom fonts (BUNGEE)

### Word Highlighting (ASS format)

When using the ASS format, the current word being spoken is highlighted using karaoke-style effects. This requires:
1. `word_timestamps=True` in the whisper transcription (already enabled)
2. BUNGEE font installed on the system

### Burning Subtitles into Video

Use ffmpeg to embed subtitles:
- SRT/VTT: `ffmpeg -i video.mp4 -vf "subtitles=subs.srt" output.mp4`
- ASS: `ffmpeg -i video.mp4 -vf "ass=subs.ass" output.mp4`

### Output Files

```
video.mp4                           # Original video
video.sbv                           # YouTube upload ready
video.srt                           # Basic subtitles
video.vtt                           # Web-compatible subtitles
video.ass                           # Word highlighting + BUNGEE font (local only)
video_with_subtitles.mp4            # Burned with ASS

output/shorts/
├── shorts_HHMMSS_category_title.mp4          # Original short
├── shorts_HHMMSS_category_title.sbv           # YouTube upload ready
├── shorts_HHMMSS_category_title.srt          # Basic subtitles
├── shorts_HHMMSS_category_title.vtt          # Web subtitles
├── shorts_HHMMSS_category_title.ass          # Word highlighting (local only)
└── shorts_HHMMSS_category_title_hq.mp4      # Burned with ASS
```
