import json
import logging
import re
import subprocess
from pathlib import Path
from typing import Optional

from .transcriber import Transcriber
from .analyzer import (
    Analyzer,
    TopicExtractor,
    BreaksExtractor,
    merge_chapters,
    _get_chunks_from_breaks,
    _sub_split_chunks,
    format_timestamp,
)
from . import config
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
        self.watch_folder = watch_folder or config.DEFAULT_WATCH_FOLDER
        self.output_folder = output_folder or config.DEFAULT_OUTPUT_FOLDER
        self.video_name = video_name

        self.transcriber = Transcriber()
        self.analyzer = Analyzer()
        self.topic_extractor = TopicExtractor()
        self.breaks_extractor = BreaksExtractor()
        self.subtitler = Subtitler()

    def _get_output_base(self, video_path: Path) -> Path:
        if self.video_name:
            return self.output_folder / self.video_name
        return self.output_folder / video_path.stem

    def _get_transcript_path(self, video_path: Path) -> Path:
        base = self._get_output_base(video_path)
        base.mkdir(parents=True, exist_ok=True)
        return base / f"{video_path.stem}.transcript.json"

    def _get_segments_path(self, video_path: Path) -> Path:
        base = self._get_output_base(video_path)
        base.mkdir(parents=True, exist_ok=True)
        return base / f"{video_path.stem}.segments.json"

    def _get_chapters_path(self, video_path: Path) -> Path:
        base = self._get_output_base(video_path)
        base.mkdir(parents=True, exist_ok=True)
        return base / f"{video_path.stem}.chapters.json"

    def _get_subtitle_dir(self, video_path: Path) -> Path:
        base = self._get_output_base(video_path)
        base.mkdir(parents=True, exist_ok=True)
        return base

    def _get_shorts_dir(self, video_path: Path) -> Path:
        base = self._get_output_base(video_path)
        shorts_dir = base / "shorts"
        shorts_dir.mkdir(parents=True, exist_ok=True)
        return shorts_dir

    def _analyze_shorts_with_breaks(self, transcript_data: dict, video_path: Path, segments_path: Path) -> dict:
        """Analyze transcript for shorts segments, using breaks as chunk boundaries."""
        video_duration = transcript_data.get("transcription", {}).get("duration", 0)

        logger.info("Detecting stream breaks for shorts chunking...")
        breaks_result = self.breaks_extractor.extract_breaks(transcript_data, video_path)
        break_events = (breaks_result or {}).get("breaks", [])

        if break_events:
            raw_chunks = _get_chunks_from_breaks(break_events, video_duration)
            max_secs = config.CHAPTER_CHUNK_MINUTES * 60
            overlap_secs = config.CHAPTER_CHUNK_OVERLAP_MINUTES * 60
            chunks = _sub_split_chunks(raw_chunks, max_secs, overlap_secs)
            logger.info(f"Chunking by {len(break_events)} breaks into {len(chunks)} sections")
        else:
            chunks = [(0, video_duration)]
            logger.info("No breaks found, analyzing full transcript")

        all_segments = []
        for i, (cs, ce) in enumerate(chunks):
            logger.info(f"  Section {i + 1}/{len(chunks)} ({format_timestamp(cs)} → {format_timestamp(ce)}): analyzing...")
            result = self.analyzer.analyze(transcript_data, video_path, chunk_start=cs, chunk_end=ce)
            segments = result.get("segments", [])

            seen = {(s["start_seconds"], s["end_seconds"]) for s in all_segments}
            new_segments = [s for s in segments if (s["start_seconds"], s["end_seconds"]) not in seen]
            all_segments.extend(new_segments)
            logger.info(f"  Section {i + 1}/{len(chunks)}: found {len(new_segments)} new segments")

        all_segments.sort(key=lambda x: x["start_seconds"])
        for i, seg in enumerate(all_segments):
            seg["id"] = i + 1

        result = {
            "source_video": str(video_path),
            "duration_seconds": video_duration,
            "segments": all_segments,
        }

        with open(segments_path, "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"Saved {len(all_segments)} segments to: {segments_path}")
        return result

    def process_video(self, video_path: Path) -> dict:
        video_path = Path(video_path).resolve()

        logger.info(f"Starting pipeline for: {video_path}")

        transcript_path = self._get_transcript_path(video_path)

        if transcript_path.exists():
            logger.info(f"Loading existing transcript: {transcript_path}")
            with open(transcript_path, "r") as f:
                transcript_data = json.load(f)
        else:
            logger.info("Step 1: Transcribing video...")
            transcript_data = self.transcriber.transcribe(video_path, transcript_path)

        segments_path = self._get_segments_path(video_path)
        if segments_path.exists():
            logger.info(f"Loading existing segments: {segments_path}")
            with open(segments_path, "r") as f:
                segments_data = json.load(f)
        else:
            logger.info("Step 2: Analyzing transcript for shorts segments...")
            segments_data = self._analyze_shorts_with_breaks(transcript_data, video_path, segments_path)

        self.extractor = Extractor(output_dir=self._get_shorts_dir(video_path))
        self.subtitler = Subtitler(output_dir=self._get_subtitle_dir(video_path))

        logger.info("Step 3: Extracting video segments...")
        extracted = self.extractor.extract_all_segments(video_path, segments_data)

        result = {
            "video": str(video_path),
            "transcript": str(transcript_path),
            "segments": str(segments_path),
            "extracted": [str(f) for f in extracted],
            "count": len(extracted),
            "output_folder": str(self._get_output_base(video_path)),
        }

        logger.info(f"Pipeline complete! Extracted {len(extracted)} segments")
        return result

    def transcribe(self, video_path: Path) -> Path:
        video_path = Path(video_path)
        transcript_path = self._get_transcript_path(video_path)
        self.transcriber.transcribe(video_path, transcript_path)
        return transcript_path

    def analyze(self, video_path: Path) -> Path:
        video_path = Path(video_path)
        transcript_path = self._get_transcript_path(video_path)

        if not transcript_path.exists():
            logger.info("Transcript not found, transcribing first...")
            self.transcribe(video_path)

        with open(transcript_path, "r") as f:
            transcript_data = json.load(f)

        segments_path = self._get_segments_path(video_path)
        if segments_path.exists():
            logger.info(f"Loading existing segments: {segments_path}")
            return segments_path

        self.subtitler = Subtitler(output_dir=self._get_subtitle_dir(video_path))
        self._analyze_shorts_with_breaks(transcript_data, video_path, segments_path)
        return segments_path

    def extract(self, video_path: Path) -> list:
        video_path = Path(video_path)
        segments_path = self._get_segments_path(video_path)

        if not segments_path.exists():
            logger.info("Segments not found, analyzing first...")
            self.analyze(video_path)

        with open(segments_path, "r") as f:
            segments_data = json.load(f)

        self.extractor = Extractor(output_dir=self._get_shorts_dir(video_path))
        extracted = self.extractor.extract_all_segments(video_path, segments_data)
        return extracted

    def generate_subtitles(
        self,
        video_path: Path,
        burn: bool = False,
        burn_format: str = "ass",
    ) -> dict:
        video_path = Path(video_path)
        transcript_path = self._get_transcript_path(video_path)

        if not transcript_path.exists():
            logger.info("Transcript not found, transcribing first...")
            self.transcribe(video_path)

        with open(transcript_path, "r") as f:
            transcript_data = json.load(f)

        self.subtitler = Subtitler(output_dir=self._get_subtitle_dir(video_path))

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
        transcript_path = self._get_transcript_path(video_path)

        if not transcript_path.exists():
            logger.info("Transcript not found, transcribing first...")
            self.transcribe(video_path)

        chapters_path = self._get_chapters_path(video_path)
        if chapters_path.exists():
            logger.info(f"Loading existing chapters: {chapters_path}")
            return chapters_path

        with open(transcript_path, "r") as f:
            transcript_data = json.load(f)

        video_duration = transcript_data.get("transcription", {}).get("duration", 0)

        logger.info("Step 1: Detecting stream breaks...")
        breaks_result = self.breaks_extractor.extract_breaks(transcript_data, video_path)
        break_events = (breaks_result or {}).get("breaks", [])
        logger.info(f"Found {len(break_events)} stream breaks")

        logger.info("Step 2: Extracting topic chapters (using breaks as boundaries)...")
        topic_result = self.topic_extractor.extract_chapters(
            transcript_data, video_path, breaks=break_events
        )

        if topic_result is None:
            logger.error("Topic extraction failed, no chapters to merge")
            chapters_data = {
                "source_video": str(video_path),
                "duration_seconds": video_duration,
                "suggested_title": None,
                "chapters": [],
            }
            chapters_path = self._get_chapters_path(video_path)
            with open(chapters_path, "w") as f:
                json.dump(chapters_data, f, indent=2)
            logger.info(f"Saved chapters (empty) to: {chapters_path}")
            return chapters_path

        topic_chapters = topic_result.get("chapters", [])

        logger.info(
            f"Merging {len(topic_chapters)} topic chapters with {len(break_events)} breaks..."
        )
        merged = merge_chapters(topic_chapters, break_events, video_duration)

        chapters_data = {
            "source_video": str(video_path),
            "duration_seconds": video_duration,
            "suggested_title": topic_result.get("suggested_title"),
            "chapters": merged,
        }

        chapters_path = self._get_chapters_path(video_path)
        with open(chapters_path, "w") as f:
            json.dump(chapters_data, f, indent=2)

        logger.info(f"Saved {len(merged)} chapters to: {chapters_path}")
        return chapters_path

    def extract_with_subtitles(
        self,
        video_path: Path,
        burn_format: str = "ass",
    ) -> dict:
        video_path = Path(video_path)
        transcript_path = self._get_transcript_path(video_path)

        if not transcript_path.exists():
            logger.info("Transcript not found, transcribing first...")
            self.transcribe(video_path)

        with open(transcript_path, "r") as f:
            transcript_data = json.load(f)

        segments_path = self._get_segments_path(video_path)
        if not segments_path.exists():
            logger.info("Segments not found, analyzing first...")
            self.analyze(video_path)

        with open(segments_path, "r") as f:
            segments_data = json.load(f)

        self.extractor = Extractor(output_dir=self._get_shorts_dir(video_path))
        self.subtitler = Subtitler(output_dir=self._get_subtitle_dir(video_path))

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


    def find_ai_mentions(
        self,
        video_path: Path,
        buffer_seconds: float = 0.5,
        group_window: float = 0.5,
        burn_counts: bool = False,
    ) -> dict:
        video_path = Path(video_path).resolve()

        transcript_path = self._get_transcript_path(video_path)
        if transcript_path.exists():
            logger.info(f"Loading existing transcript: {transcript_path}")
            with open(transcript_path, "r") as f:
                transcript_data = json.load(f)
        else:
            logger.info("Transcribing video...")
            transcript_data = self.transcriber.transcribe(video_path, transcript_path)

        video_duration = transcript_data.get("transcription", {}).get("duration", 0)

        mentions = []
        segments = transcript_data.get("transcription", {}).get("segments", [])
        for seg in segments:
            for word in seg.get("words", []):
                word_text = word.get("word", "").strip()
                if re.search(r'\bai\b', word_text.lower()):
                    mentions.append({
                        "word": word_text,
                        "start": word["start"],
                        "end": word["end"],
                        "probability": word.get("probability", 0),
                    })

        logger.info(f"Found {len(mentions)} AI mention(s) in transcript")

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

        output_dir = self._get_output_base(video_path) / "ai_clips"
        output_dir.mkdir(parents=True, exist_ok=True)

        clips = []
        group_data = []

        for i, group in enumerate(groups, 1):
            group_start = group[0]["start"]
            group_end = group[-1]["end"]

            clip_start = max(0, group_start - buffer_seconds)
            clip_end = min(video_duration, group_end + buffer_seconds)
            duration = clip_end - clip_start

            output_name = f"ai_mention_{i:03d}"
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
                f"Extracting AI mention group {i}: "
                f"{clip_start:.1f}s - {clip_end:.1f}s "
                f"(duration: {duration:.1f}s, {len(group)} mention(s))"
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
                "mentions": [
                    {
                        "word": m["word"],
                        "start": m["start"],
                        "end": m["end"],
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

            compilation_path = str(output_dir / "ai_compilation.mp4")
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
            "total_mentions": len(mentions),
            "groups": group_data,
            "compilation": compilation_path,
        }

        metadata_path = output_dir / "ai_mentions.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(
            f"Extracted {len(clips)} AI mention clip(s) "
            f"from {len(mentions)} mention(s) in {len(groups)} group(s)"
        )

        return metadata


def create_pipeline(
    watch_folder: Optional[Path] = None,
    output_folder: Optional[Path] = None,
    video_name: Optional[str] = None,
) -> Pipeline:
    return Pipeline(
        watch_folder=watch_folder,
        output_folder=output_folder,
        video_name=video_name,
    )