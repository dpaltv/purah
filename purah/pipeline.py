import json
import logging
from pathlib import Path
from typing import Optional

from .transcriber import Transcriber
from .analyzer import Analyzer, ChapterExtractor
from .extractor import Extractor
from .subtitler import Subtitler
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
        self.chapter_extractor = ChapterExtractor()
        self.extractor = Extractor(output_dir=self.output_folder / "shorts")
        self.subtitler = Subtitler(output_dir=self.output_folder)

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

    def generate_subtitles(
        self,
        video_path: Path,
        burn: bool = False,
        burn_format: str = "ass",
    ) -> dict:
        video_path = Path(video_path)
        transcript_path = video_path.with_suffix(".transcript.json")
        
        if not transcript_path.exists():
            raise FileNotFoundError(
                f"Transcript not found: {transcript_path}. "
                "Run 'purah transcribe' first."
            )
        
        with open(transcript_path, "r") as f:
            transcript_data = json.load(f)
        
        logger.info("Generating subtitle files...")
        subtitle_files = self.subtitler.generate_subtitle_files(
            transcript_data, video_path
        )
        
        result = {
            "video": str(video_path),
            "subtitle_files": {k: str(v) for k, v in subtitle_files.items()},
        }
        
        if burn and "ass" in subtitle_files:
            burn_path = self.subtitler.burn_subtitles(
                video_path,
                subtitle_files["ass"],
                subtitle_format="ass",
            )
            result["burned_video"] = str(burn_path)
        elif burn and "srt" in subtitle_files:
            burn_path = self.subtitler.burn_subtitles(
                video_path,
                subtitle_files["srt"],
                subtitle_format="srt",
            )
            result["burned_video"] = str(burn_path)
        
        logger.info(f"Generated {len(subtitle_files)} subtitle files")
        return result

    def extract_chapters(self, video_path: Path) -> Path:
        video_path = Path(video_path)
        transcript_path = video_path.with_suffix(".transcript.json")
        
        if not transcript_path.exists():
            raise FileNotFoundError(
                f"Transcript not found: {transcript_path}. "
                "Run 'purah transcribe' first."
            )
        
        chapters_path = video_path.with_suffix(".chapters.json")
        if chapters_path.exists():
            logger.info(f"Loading existing chapters: {chapters_path}")
            return chapters_path
        
        with open(transcript_path, "r") as f:
            transcript_data = json.load(f)
        
        logger.info("Extracting chapters...")
        chapters_data = self.chapter_extractor.extract_chapters(transcript_data, video_path)
        
        chapters_path = video_path.with_suffix(".chapters.json")
        with open(chapters_path, "w") as f:
            json.dump(chapters_data, f, indent=2)
        
        logger.info(f"Saved chapters to: {chapters_path}")
        return chapters_path

    def extract_with_subtitles(
        self,
        video_path: Path,
        burn_format: str = "ass",
    ) -> dict:
        video_path = Path(video_path)
        transcript_path = video_path.with_suffix(".transcript.json")
        
        if not transcript_path.exists():
            raise FileNotFoundError(
                f"Transcript not found: {transcript_path}. "
                "Run 'purah transcribe' first."
            )
        
        with open(transcript_path, "r") as f:
            transcript_data = json.load(f)
        
        segments_path = video_path.with_suffix(".segments.json")
        if not segments_path.exists():
            raise FileNotFoundError(
                f"Segments not found: {segments_path}. "
                "Run 'purah analyze' first."
            )
        
        with open(segments_path, "r") as f:
            segments_data = json.load(f)
        
        logger.info("Extracting segments with subtitles...")
        extracted = []
        subtitle_files = []
        burned_videos = []
        
        for segment in segments_data.get("segments", []):
            try:
                start_seconds = segment["start_seconds"]
                end_seconds = segment["end_seconds"]
                title = segment.get("title", f"segment_{segment['id']}")
                category = segment.get("category", "short")
                
                timestamp_str = (
                    f"{int(start_seconds // 3600):02d}"
                    f"{int((start_seconds % 3600) // 60):02d}"
                    f"{int(start_seconds % 60):02d}"
                )
                safe_category = self.extractor.sanitize_filename(category)
                safe_title = self.extractor.sanitize_filename(title)
                output_name = f"shorts_{timestamp_str}_{safe_category}_{safe_title}"
                
                video_output = self.extractor.extract_segment(
                    video_path,
                    start_seconds,
                    end_seconds,
                    title,
                    category,
                )
                extracted.append(video_output)
                
                seg_subtitles = self.subtitler.generate_segment_subtitles(
                    transcript_data,
                    video_path,
                    start_seconds,
                    end_seconds,
                    output_name,
                )
                subtitle_files.extend(
                    (k, str(v)) for k, v in seg_subtitles.items()
                )
                
                if burn_format == "ass" and "ass" in seg_subtitles:
                    burned = self.subtitler.burn_subtitles(
                        video_output,
                        seg_subtitles["ass"],
                        subtitle_format="ass",
                    )
                    burned_videos.append(str(burned))
                elif burn_format == "srt" and "srt" in seg_subtitles:
                    burned = self.subtitler.burn_subtitles(
                        video_output,
                        seg_subtitles["srt"],
                        subtitle_format="srt",
                    )
                    burned_videos.append(str(burned))
                
            except Exception as e:
                logger.error(f"Failed to extract segment {segment.get('id')}: {e}")
                continue
        
        result = {
            "video": str(video_path),
            "extracted": [str(f) for f in extracted],
            "subtitles": dict(subtitle_files),
            "burned_videos": burned_videos,
            "count": len(extracted),
        }
        
        logger.info(
            f"Extracted {len(extracted)} segments with subtitles, "
            f"{len(burned_videos)} with burned subtitles"
        )
        return result


def create_pipeline(
    watch_folder: Optional[Path] = None,
    output_folder: Optional[Path] = None,
) -> Pipeline:
    return Pipeline(watch_folder=watch_folder, output_folder=output_folder)
