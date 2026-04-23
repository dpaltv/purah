import logging
import subprocess
from pathlib import Path
from typing import Optional, List, Tuple

from . import config

logger = logging.getLogger(__name__)


def format_timestamp_srt(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def format_timestamp_vtt(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def format_timestamp_ass(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centisecs = int((seconds % 1) * 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{centisecs:02d}"


def get_word_segments(segments: List[dict]) -> List[dict]:
    all_words = []
    for seg in segments:
        if "words" in seg:
            for word_data in seg["words"]:
                all_words.append({
                    "word": word_data.get("word", "").strip(),
                    "start": word_data.get("start", 0),
                    "end": word_data.get("end", 0),
                    "probability": word_data.get("probability", 1.0),
                })
    return all_words


def chunk_words(words: List[dict], max_words: int = 3) -> List[List[dict]]:
    """Split word list into groups of max_words for smaller subtitle chunks."""
    chunks = []
    for i in range(0, len(words), max_words):
        chunks.append(words[i:i + max_words])
    return chunks


def transcript_to_srt(transcript_data: dict, offset: float = 0) -> str:
    segments = transcript_data.get("transcription", {}).get("segments", [])
    
    if not segments:
        return ""

    srt_lines = []
    for i, seg in enumerate(segments, start=1):
        start = seg.get("start", 0) + offset
        end = seg.get("end", 0) + offset
        text = seg.get("text", "").strip()

        if not text:
            continue

        srt_lines.append(f"{i}")
        srt_lines.append(f"{format_timestamp_srt(start)} --> {format_timestamp_srt(end)}")
        srt_lines.append(text)
        srt_lines.append("")

    return "\n".join(srt_lines)


def transcript_to_vtt(transcript_data: dict, offset: float = 0) -> str:
    segments = transcript_data.get("transcription", {}).get("segments", [])
    
    if not segments:
        return ""

    vtt_lines = ["WEBVTT", ""]
    
    for seg in segments:
        start = seg.get("start", 0) + offset
        end = seg.get("end", 0) + offset
        text = seg.get("text", "").strip()

        if not text:
            continue

        vtt_lines.append(f"{format_timestamp_vtt(start)} --> {format_timestamp_vtt(end)}")
        vtt_lines.append(text)
        vtt_lines.append("")

    return "\n".join(vtt_lines)


def format_timestamp_sbv(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def transcript_to_sbv(transcript_data: dict, offset: float = 0) -> str:
    segments = transcript_data.get("transcription", {}).get("segments", [])
    
    if not segments:
        return ""

    sbv_lines = []
    for seg in segments:
        start = seg.get("start", 0) + offset
        end = seg.get("end", 0) + offset
        text = seg.get("text", "").strip()

        if not text:
            continue

        sbv_lines.append(f"{format_timestamp_sbv(start)},{format_timestamp_sbv(end)}")
        sbv_lines.append(text)
        sbv_lines.append("")

    return "\n".join(sbv_lines)


def build_ass_header() -> str:
    return f"""[Script Info]
Title: Purah Subtitles
ScriptType: v4.00+
WrapStyle: 0
PlayResX: 384
PlayResY: 288
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,BUNGEE,20,&H00FFFFFF,&H00FFFF00,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def build_ass_line(
    start: float,
    end: float,
    text: str,
    style: str = "Default",
) -> str:
    start_str = format_timestamp_ass(start)
    end_str = format_timestamp_ass(end)
    return f"Dialogue: 0,{start_str},{end_str},{style},,0,0,0,,{text}"


def build_ass_karaoke_line(
    start: float,
    words: List[dict],
    style: str = "Default",
    max_words: int = 3,
) -> List[str]:
    """Build ASS karaoke lines for word groups with smaller chunks.
    
    Splits words into chunks of max_words for better readability.
    Uses word's own duration (word_end - word_start) for timing.
    """
    if not words:
        return []

    lines = []
    word_chunks = chunk_words(words, max_words)

    for chunk in word_chunks:
        chunk_start = chunk[0]["start"]
        chunk_end = chunk[-1]["end"]

        karaoke_text = ""
        for word_data in chunk:
            word = word_data.get("word", "").strip()
            word_start = word_data.get("start", 0)
            word_end = word_data.get("end", 0)

            if not word:
                continue

            duration = word_end - word_start
            duration_centisecs = max(1, int(duration * 100))

            karaoke_text += f"{{\\k{duration_centisecs}}}{word} "

        karaoke_text = karaoke_text.strip()

        start_str = format_timestamp_ass(chunk_start)
        end_str = format_timestamp_ass(chunk_end)

        lines.append(f"Dialogue: 0,{start_str},{end_str},{style},,0,0,0,,{karaoke_text}")

    return lines


def transcript_to_ass(transcript_data: dict, offset: float = 0) -> str:
    segments = transcript_data.get("transcription", {}).get("segments", [])
    
    if not segments:
        return ""

    ass_lines = [build_ass_header()]
    
    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            continue

        start = seg.get("start", 0) + offset
        end = seg.get("end", 0) + offset

        if "words" in seg and seg["words"]:
            words = [
                {
                    "word": w.get("word", "").strip(),
                    "start": w.get("start", 0) + offset,
                    "end": w.get("end", 0) + offset,
                }
                for w in seg["words"]
                if w.get("word", "").strip()
            ]
            if words:
                karaoke_lines = build_ass_karaoke_line(start, words, max_words=3)
                ass_lines.extend(karaoke_lines)
                continue

        ass_lines.append(build_ass_line(start, end, text))

    return "\n".join(ass_lines)


class Subtitler:
    def __init__(
        self,
        output_dir: Optional[Path] = None,
        subtitle_format: str = "both",
    ):
        self.output_dir = output_dir or config.DEFAULT_OUTPUT_FOLDER
        self.subtitle_format = subtitle_format

    def generate_subtitle_files(
        self,
        transcript_data: dict,
        video_path: Path,
        offset: float = 0,
    ) -> dict:
        video_path = Path(video_path)
        base_name = video_path.stem
        
        output_files = {}

        srt_content = transcript_to_srt(transcript_data, offset)
        if srt_content:
            srt_path = video_path.with_suffix(".srt")
            with open(srt_path, "w", encoding="utf-8") as f:
                f.write(srt_content)
            output_files["srt"] = srt_path
            logger.info(f"Generated SRT: {srt_path}")

        sbv_content = transcript_to_sbv(transcript_data, offset)
        if sbv_content:
            sbv_path = video_path.with_suffix(".sbv")
            with open(sbv_path, "w", encoding="utf-8") as f:
                f.write(sbv_content)
            output_files["sbv"] = sbv_path
            logger.info(f"Generated SBV (YouTube): {sbv_path}")

        vtt_content = transcript_to_vtt(transcript_data, offset)
        if vtt_content:
            vtt_path = video_path.with_suffix(".vtt")
            with open(vtt_path, "w", encoding="utf-8") as f:
                f.write(vtt_content)
            output_files["vtt"] = vtt_path
            logger.info(f"Generated VTT: {vtt_path}")

        ass_content = transcript_to_ass(transcript_data, offset)
        if ass_content:
            ass_path = video_path.with_suffix(".ass")
            with open(ass_path, "w", encoding="utf-8") as f:
                f.write(ass_content)
            output_files["ass"] = ass_path
            logger.info(f"Generated ASS (with word highlighting): {ass_path}")

        return output_files

    def burn_subtitles(
        self,
        video_path: Path,
        subtitle_path: Path,
        output_path: Optional[Path] = None,
        subtitle_format: str = "srt",
    ) -> Path:
        video_path = Path(video_path)
        subtitle_path = Path(subtitle_path)

        if output_path is None:
            stem = video_path.stem
            output_path = video_path.parent / f"{stem}_with_subtitles.mp4"

        output_path = Path(output_path)

        if subtitle_format == "ass":
            filter_str = f"ass={subtitle_path}"
        else:
            filter_str = f"subtitles={subtitle_path}"

        cmd = [
            config.FFMPEG_PATH,
            "-y",
            "-i", str(video_path),
            "-vf", filter_str,
            "-c:v", "libx264",
            "-preset", "slow",
            "-crf", "16",
            "-c:a", "copy",
            str(output_path),
        ]

        logger.info(f"Burning {subtitle_format.upper()} subtitles into {video_path.name}...")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg subtitle burn failed: {result.stderr}")

        logger.info(f"Burned subtitles saved to: {output_path}")
        return output_path

    def get_segment_transcript(
        self,
        transcript_data: dict,
        start_seconds: float,
        end_seconds: float,
    ) -> dict:
        segments = transcript_data.get("transcription", {}).get("segments", [])
        
        filtered_segments = []
        for seg in segments:
            seg_start = seg.get("start", 0)
            seg_end = seg.get("end", 0)
            
            if seg_end >= start_seconds and seg_start <= end_seconds:
                clip_start = max(0, start_seconds - seg_start)
                clip_end = seg_end - seg_start
                
                text = seg.get("text", "").strip()
                
                if "words" in seg:
                    words = []
                    for w in seg["words"]:
                        w_start = w.get("start", 0)
                        w_end = w.get("end", 0)
                        
                        if w_end >= start_seconds and w_start <= end_seconds:
                            words.append({
                                "word": w.get("word", ""),
                                "start": w_start - start_seconds,
                                "end": w_end - start_seconds,
                                "probability": w.get("probability", 1.0),
                            })
                    
                    if words:
                        filtered_segments.append({
                            "start": clip_start,
                            "end": clip_end,
                            "text": text,
                            "words": words,
                        })
                else:
                    if text:
                        filtered_segments.append({
                            "start": clip_start,
                            "end": clip_end,
                            "text": text,
                        })

        return {
            "transcription": {
                "segments": filtered_segments,
                "duration": end_seconds - start_seconds,
            }
        }

    def generate_segment_subtitles(
        self,
        transcript_data: dict,
        video_path: Path,
        start_seconds: float,
        end_seconds: float,
        output_name: str,
    ) -> dict:
        segment_transcript = self.get_segment_transcript(
            transcript_data, start_seconds, end_seconds
        )

        video_path = Path(video_path)
        
        output_files = {}
        
        srt_content = transcript_to_srt(segment_transcript, 0)
        if srt_content:
            srt_path = video_path.parent / f"{output_name}.srt"
            with open(srt_path, "w", encoding="utf-8") as f:
                f.write(srt_content)
            output_files["srt"] = srt_path
            logger.info(f"Generated segment SRT: {srt_path}")

        sbv_content = transcript_to_sbv(segment_transcript, 0)
        if sbv_content:
            sbv_path = video_path.parent / f"{output_name}.sbv"
            with open(sbv_path, "w", encoding="utf-8") as f:
                f.write(sbv_content)
            output_files["sbv"] = sbv_path
            logger.info(f"Generated segment SBV (YouTube): {sbv_path}")

        vtt_content = transcript_to_vtt(segment_transcript, 0)
        if vtt_content:
            vtt_path = video_path.parent / f"{output_name}.vtt"
            with open(vtt_path, "w", encoding="utf-8") as f:
                f.write(vtt_content)
            output_files["vtt"] = vtt_path
            logger.info(f"Generated segment VTT: {vtt_path}")

        ass_content = transcript_to_ass(segment_transcript, 0)
        if ass_content:
            ass_path = video_path.parent / f"{output_name}.ass"
            with open(ass_path, "w", encoding="utf-8") as f:
                f.write(ass_content)
            output_files["ass"] = ass_path
            logger.info(f"Generated segment ASS: {ass_path}")

        return output_files

    def burn_segment_subtitles(
        self,
        video_path: Path,
        subtitle_path: Path,
        output_path: Path,
        subtitle_format: str = "ass",
    ) -> Path:
        return self.burn_subtitles(
            video_path,
            subtitle_path,
            output_path,
            subtitle_format,
        )
