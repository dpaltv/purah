import json
import subprocess
import logging
from pathlib import Path
from typing import Optional, List

from . import config

logger = logging.getLogger(__name__)


def sanitize_filename(name: str) -> str:
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        name = name.replace(char, "_")
    name = name[:100]
    return name.strip("_")


class Extractor:
    def __init__(
        self,
        output_dir: Optional[Path] = None,
        buffer_seconds: Optional[int] = None,
    ):
        self.output_dir = output_dir or config.SHORTS_OUTPUT_FOLDER
        self.buffer_seconds = buffer_seconds or config.BUFFER_SECONDS
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_duration(self, video_path: Path) -> float:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        
        return float(result.stdout.strip())

    def extract_segment(
        self,
        video_path: Path,
        start_seconds: float,
        end_seconds: float,
        output_name: str,
        category: str = "short",
    ) -> Path:
        video_path = Path(video_path)
        
        buffer_start = max(0, start_seconds - self.buffer_seconds)
        video_duration = self.get_duration(video_path)
        buffer_end = min(video_duration, end_seconds + self.buffer_seconds)
        
        duration = buffer_end - buffer_start
        
        safe_category = sanitize_filename(category)
        safe_name = sanitize_filename(output_name)
        
        timestamp_str = f"{int(start_seconds // 3600):02d}{int((start_seconds % 3600) // 60):02d}{int(start_seconds % 60):02d}"
        
        output_filename = f"shorts_{timestamp_str}_{safe_category}_{safe_name}.mp4"
        output_path = self.output_dir / output_filename

        cmd = [
            "ffmpeg",
            "-y",
            "-ss", str(buffer_start),
            "-i", str(video_path),
            "-t", str(duration),
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-c:a", "aac",
            "-b:a", "128k",
            str(output_path),
        ]

        logger.info(
            f"Extracting segment: {output_name} "
            f"(buffer: {buffer_start:.0f}s - {buffer_end:.0f}s, "
            f"duration: {duration:.0f}s)"
        )

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg failed: {result.stderr}")

        logger.info(f"Extracted: {output_path}")
        return output_path

    def extract_all_segments(
        self,
        video_path: Path,
        segments_data: dict,
    ) -> List[Path]:
        video_path = Path(video_path)
        extracted_files = []

        for segment in segments_data.get("segments", []):
            try:
                output_path = self.extract_segment(
                    video_path,
                    segment["start_seconds"],
                    segment["end_seconds"],
                    segment.get("title", f"segment_{segment['id']}"),
                    segment.get("category", "short"),
                )
                extracted_files.append(output_path)
            except Exception as e:
                logger.error(f"Failed to extract segment {segment.get('id')}: {e}")
                continue

        metadata_path = self.output_dir / f"{video_path.stem}_segments.json"
        with open(metadata_path, "w") as f:
            json.dump(segments_data, f, indent=2)
        
        logger.info(f"Saved segment metadata to {metadata_path}")
        
        return extracted_files
