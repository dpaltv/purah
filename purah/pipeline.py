import json
import logging
from pathlib import Path
from typing import Optional

from .transcriber import Transcriber
from .analyzer import Analyzer
from .extractor import Extractor
from . import config

logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(
        self,
        watch_folder: Optional[Path] = None,
        output_folder: Optional[Path] = None,
    ):
        self.watch_folder = watch_folder or config.DEFAULT_WATCH_FOLDER
        self.output_folder = output_folder or config.DEFAULT_OUTPUT_FOLDER
        
        self.transcriber = Transcriber()
        self.analyzer = Analyzer()
        self.extractor = Extractor(output_dir=self.output_folder / "shorts")

    def process_video(self, video_path: Path) -> dict:
        video_path = Path(video_path).resolve()
        
        logger.info(f"Starting pipeline for: {video_path}")
        
        transcript_path = video_path.with_suffix(".transcript.json")
        
        # Check if transcript already exists
        if transcript_path.exists():
            logger.info(f"Loading existing transcript: {transcript_path}")
            with open(transcript_path, "r") as f:
                transcript_data = json.load(f)
        else:
            logger.info("Step 1: Transcribing video...")
            transcript_data = self.transcriber.transcribe(video_path, transcript_path)
        
        # Check if segments already exist
        segments_path = video_path.with_suffix(".segments.json")
        if segments_path.exists():
            logger.info(f"Loading existing segments: {segments_path}")
            with open(segments_path, "r") as f:
                segments_data = json.load(f)
        else:
            logger.info("Step 2: Analyzing transcript for shorts segments...")
            segments_data = self.analyzer.analyze(transcript_data, video_path)
            
            with open(segments_path, "w") as f:
                json.dump(segments_data, f, indent=2)
            logger.info(f"Saved segment analysis to: {segments_path}")
        
        logger.info("Step 3: Extracting video segments...")
        extracted = self.extractor.extract_all_segments(video_path, segments_data)
        
        result = {
            "video": str(video_path),
            "transcript": str(transcript_path),
            "segments": str(segments_path),
            "extracted": [str(f) for f in extracted],
            "count": len(extracted),
        }
        
        logger.info(f"Pipeline complete! Extracted {len(extracted)} segments")
        return result

    def transcribe_only(self, video_path: Path) -> Path:
        video_path = Path(video_path)
        transcript_path = video_path.with_suffix(".transcript.json")
        self.transcriber.transcribe(video_path, transcript_path)
        return transcript_path

    def analyze_only(self, video_path: Path) -> Path:
        video_path = Path(video_path)
        transcript_path = video_path.with_suffix(".transcript.json")
        
        if not transcript_path.exists():
            raise FileNotFoundError(
                f"Transcript not found: {transcript_path}. "
                "Run 'purah transcribe' first."
            )
        
        with open(transcript_path, "r") as f:
            transcript_data = json.load(f)
        
        segments_data = self.analyzer.analyze(transcript_data, video_path)
        
        segments_path = video_path.with_suffix(".segments.json")
        with open(segments_path, "w") as f:
            json.dump(segments_data, f, indent=2)
        
        logger.info(f"Saved segment analysis to: {segments_path}")
        return segments_path

    def extract_only(self, video_path: Path) -> list:
        video_path = Path(video_path)
        segments_path = video_path.with_suffix(".segments.json")
        
        if not segments_path.exists():
            raise FileNotFoundError(
                f"Segments file not found: {segments_path}. "
                "Run 'purah analyze' first."
            )
        
        with open(segments_path, "r") as f:
            segments_data = json.load(f)
        
        extracted = self.extractor.extract_all_segments(video_path, segments_data)
        return extracted


def create_pipeline(
    watch_folder: Optional[Path] = None,
    output_folder: Optional[Path] = None,
) -> Pipeline:
    return Pipeline(watch_folder=watch_folder, output_folder=output_folder)
