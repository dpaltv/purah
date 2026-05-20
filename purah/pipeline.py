import json
import logging
import re
import subprocess
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
        video_name: Optional[str] = None,
    ):
        self.watch_folder = Path(watch_folder) if watch_folder else config.DEFAULT_WATCH_FOLDER
        self.output_folder = Path(output_folder) if output_folder else config.DEFAULT_OUTPUT_FOLDER
        self.video_name = video_name

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


    def _get_output_base(self, video_path: Path) -> Path:
        if self.video_name:
            return self.output_folder / self.video_name
        return self.output_folder / video_path.stem

    @staticmethod
    def _normalize_word(word: str) -> str:
        return word.lower().strip(".,!?;:'\"-–—()[]{}<>")

    def find_mentions(
        self,
        video_path: Path,
        phrases: list,
        buffer_seconds: float = 0.5,
        group_window: float = 0.5,
        burn_counts: bool = False,
    ) -> dict:
        video_path = Path(video_path).resolve()

        transcript_path = video_path.with_suffix(".transcript.json")
        if transcript_path.exists():
            logger.info(f"Loading existing transcript: {transcript_path}")
            with open(transcript_path, "r") as f:
                transcript_data = json.load(f)
        else:
            logger.info("Transcribing video...")
            transcript_data = self.transcriber.transcribe(video_path, transcript_path)

        video_duration = transcript_data.get("transcription", {}).get("duration", 0)

        all_words = []
        segments = transcript_data.get("transcription", {}).get("segments", [])
        for seg in segments:
            all_words.extend(seg.get("words", []))

        if not all_words:
            return {
                "video": str(video_path),
                "duration_seconds": video_duration,
                "total_mentions": 0,
                "groups": [],
                "clips": [],
            }

        mentions = []

        for phrase in phrases:
            phrase_lower = phrase.lower().strip()
            phrase_words = phrase_lower.split()

            if len(phrase_words) == 1:
                for word in all_words:
                    word_text = word.get("word", "").strip()
                    if re.search(rf'\b{re.escape(phrase_lower)}\b', word_text.lower()):
                        mentions.append({
                            "word": word_text,
                            "start": word["start"],
                            "end": word["end"],
                            "probability": word.get("probability", 0),
                            "phrase": phrase,
                        })
            else:
                n = len(phrase_words)
                i = 0
                while i < len(all_words) - n + 1:
                    match = True
                    for j in range(n):
                        w = self._normalize_word(all_words[i + j].get("word", ""))
                        if w != phrase_words[j]:
                            match = False
                            break
                    if match:
                        mentions.append({
                            "word": " ".join(
                                all_words[i + k].get("word", "") for k in range(n)
                            ),
                            "start": all_words[i]["start"],
                            "end": all_words[i + n - 1]["end"],
                            "probability": min(
                                all_words[i + k].get("probability", 1) for k in range(n)
                            ),
                            "phrase": phrase,
                        })
                        i += n
                    else:
                        i += 1

        mentions.sort(key=lambda m: m["start"])

        logger.info(
            f"Found {len(mentions)} mention(s) in transcript for phrases: {phrases}"
        )

        if not mentions:
            return {
                "video": str(video_path),
                "duration_seconds": video_duration,
                "total_mentions": 0,
                "groups": [],
                "clips": [],
            }

        groups = []
        current_group = [mentions[0]]

        for mention in mentions[1:]:
            gap = mention["start"] - current_group[-1]["end"]
            if gap <= group_window:
                current_group.append(mention)
            else:
                groups.append(current_group)
                current_group = [mention]

        if current_group:
            groups.append(current_group)

        output_dir = self._get_output_base(video_path) / "mentions"
        output_dir.mkdir(parents=True, exist_ok=True)

        clips = []
        group_data = []

        for i, group in enumerate(groups, 1):
            group_start = group[0]["start"]
            group_end = group[-1]["end"]

            clip_start = max(0, group_start - buffer_seconds)
            clip_end = min(video_duration, group_end + buffer_seconds)
            duration = clip_end - clip_start

            unique_phrases = sorted(set(m["phrase"] for m in group))
            phrase_label = "_".join(p.lower().replace(" ", "_") for p in unique_phrases)
            output_name = f"mention_{i:03d}_{phrase_label}"
            output_path = output_dir / f"{output_name}.mp4"

            cmd = [
                config.FFMPEG_PATH,
                "-y",
                "-ss", str(clip_start),
                "-i", str(video_path),
                "-t", str(duration),
            ]

            if burn_counts and config.BUNGEE_FONT_PATH:
                cmd += [
                    "-vf", (
                        f"drawtext=text='{i}':"
                        f"fontfile={config.BUNGEE_FONT_PATH}:"
                        f"fontsize=120:"
                        f"fontcolor=white:"
                        f"borderw=4:"
                        f"bordercolor=black:"
                        f"x=w-tw-40:"
                        f"y=40"
                    ),
                ]

            cmd += [
                "-c:v", "libx264",
                "-preset", "slow",
                "-crf", "16",
                "-c:a", "aac",
                "-b:a", "128k",
                str(output_path),
            ]

            logger.info(
                f"Extracting mention group {i}: "
                f"{clip_start:.1f}s - {clip_end:.1f}s "
                f"(duration: {duration:.1f}s, {len(group)} mention(s), "
                f"phrases: {unique_phrases})"
            )

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"FFmpeg failed for group {i}: {result.stderr}")
                continue

            clips.append(str(output_path))

            group_data.append({
                "group": i,
                "clip": str(output_path),
                "clip_start": clip_start,
                "clip_end": clip_end,
                "duration": duration,
                "mention_count": len(group),
                "phrases": unique_phrases,
                "mentions": [
                    {
                        "word": m["word"],
                        "start": m["start"],
                        "end": m["end"],
                        "phrase": m["phrase"],
                    }
                    for m in group
                ],
            })

        compilation_path = None
        if len(clips) > 1:
            concat_file = output_dir / "concat.txt"
            with open(concat_file, "w") as f:
                for clip in clips:
                    f.write(f"file '{Path(clip).resolve()}'\n")

            compilation_path = str(output_dir / "mentions_compilation.mp4")
            cmd = [
                config.FFMPEG_PATH,
                "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", str(concat_file),
                "-c:v", "libx264",
                "-preset", "slow",
                "-crf", "16",
                "-c:a", "aac",
                "-b:a", "128k",
                compilation_path,
            ]

            logger.info("Stitching all clips into compilation...")
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"Compilation failed: {result.stderr}")
                compilation_path = None
            else:
                concat_file.unlink()

        metadata = {
            "source_video": str(video_path),
            "duration_seconds": video_duration,
            "buffer_seconds": buffer_seconds,
            "group_window_seconds": group_window,
            "phrases": phrases,
            "total_mentions": len(mentions),
            "groups": group_data,
            "compilation": compilation_path,
        }

        metadata_path = output_dir / "mentions.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(
            f"Extracted {len(clips)} clip(s) "
            f"from {len(mentions)} mention(s) in {len(groups)} group(s)"
        )

        return metadata

    def find_ai_mentions(
        self,
        video_path: Path,
        buffer_seconds: float = 0.5,
        group_window: float = 0.5,
        burn_counts: bool = False,
    ) -> dict:
        return self.find_mentions(
            video_path,
            phrases=["AI"],
            buffer_seconds=buffer_seconds,
            group_window=group_window,
            burn_counts=burn_counts,
        )


def create_pipeline(
    watch_folder: Optional[Path] = None,
    output_folder: Optional[Path] = None,
) -> Pipeline:
    return Pipeline(watch_folder=watch_folder, output_folder=output_folder)
