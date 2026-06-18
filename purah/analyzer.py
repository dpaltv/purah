import json
import logging
import re
from pathlib import Path
from typing import Optional
import requests

from . import config

logger = logging.getLogger(__name__)


def format_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def parse_timestamp(ts: str) -> float:
    parts = ts.split(":")
    if len(parts) == 3:
        return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
    elif len(parts) == 2:
        return float(parts[0]) * 60 + float(parts[1])
    return float(ts)


def _get_chunks(video_duration: float) -> list:
    chunk_secs = config.CHAPTER_CHUNK_MINUTES * 60
    overlap_secs = config.CHAPTER_CHUNK_OVERLAP_MINUTES * 60
    chunks = []
    start = 0.0
    while start < video_duration:
        end = min(start + chunk_secs, video_duration)
        chunks.append((start, end))
        if end >= video_duration:
            break
        start = end - overlap_secs
    return chunks


def extract_transcript_window(transcript_data: dict, start_seconds: float = 0, end_seconds: Optional[float] = None) -> str:
    segments = transcript_data.get("transcription", {}).get("segments", [])
    if end_seconds is None:
        end_seconds = transcript_data.get("transcription", {}).get("duration", 0)
    formatted = []
    for seg in segments:
        s = seg.get("start", 0)
        if s < start_seconds or s >= end_seconds:
            continue
        text = seg.get("text", "").strip()
        if text:
            formatted.append(f"[{format_timestamp(s)}] {text}")
    return " ".join(formatted)


def get_transcript_segment(
    transcript_data: dict,
    start_seconds: float,
    end_seconds: float,
) -> str:
    segments = transcript_data.get("transcription", {}).get("segments", [])
    
    relevant_segments = [
        seg.get("text", "").strip()
        for seg in segments
        if seg.get("start", 0) >= start_seconds and seg.get("end", 0) <= end_seconds
    ]
    
    return " ".join(relevant_segments)


def get_transcript_before(
    transcript_data: dict,
    start_seconds: float,
    buffer_seconds: float = 180,
) -> str:
    before_start = max(0, start_seconds - buffer_seconds)
    segments = transcript_data.get("transcription", {}).get("segments", [])
    
    relevant_segments = [
        seg.get("text", "").strip()
        for seg in segments
        if seg.get("start", 0) >= before_start and seg.get("end", 0) <= start_seconds
    ]
    
    return " ".join(relevant_segments)


def get_transcript_after(
    transcript_data: dict,
    end_seconds: float,
    buffer_seconds: float = 180,
    video_duration: float = 0,
) -> str:
    after_end = min(video_duration, end_seconds + buffer_seconds)
    segments = transcript_data.get("transcription", {}).get("segments", [])
    
    relevant_segments = [
        seg.get("text", "").strip()
        for seg in segments
        if seg.get("start", 0) >= end_seconds and seg.get("end", 0) <= after_end
    ]
    
    return " ".join(relevant_segments)


class Analyzer:
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_retries: int = 3,
    ):
        self.base_url = base_url or config.LM_STUDIO_BASE_URL
        self.api_key = api_key or config.LM_STUDIO_API_KEY
        self.model = model or config.LM_STUDIO_MODEL
        self.temperature = temperature
        self.max_retries = max_retries
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        })

    def check_connection(self) -> bool:
        try:
            response = self._session.get(
                f"{self.base_url}/models",
                timeout=5
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to connect to LM Studio: {e}")
            return False

    def _build_prompt(self, transcript_text: str) -> str:
        return f"""You are analyzing a Twitch stream transcript to find segments suitable for YouTube Shorts.

CRITICAL INSTRUCTIONS:
1. The transcript below contains timestamps in format "[MM:SS] text" showing WHEN each segment was spoken
2. You MUST use ONLY timestamps that appear in the transcript text below
3. NEVER guess or estimate timestamps - only use timestamps that are explicitly shown
4. If you cannot find a timestamp in the transcript, do NOT include that segment

The stream is technical content (coding, hardware builds, programming tutorials).

For each potential short segment, identify:
- start_time: When the segment starts (HH:MM:SS format from the beginning of the video) - MUST match a timestamp from the transcript
- end_time: When the segment ends (HH:MM:SS format) - MUST match a timestamp from the transcript
- category: One of: coding_idea, funny_moment, technical_detail, demo, tip
- title: A short, catchy title (max 60 characters)
- description: 1-2 sentence summary of what makes this segment interesting
- confidence: Score from 0.0 to 1.0 indicating how good this segment is for a short

Look for:
- Coding ideas: unique solutions, cool tricks, architecture decisions
- Funny moments: jokes, mishaps, unexpected outcomes
- Technical details: complex explanations, deep dives, important concepts
- Demos: live coding demonstrations, project showcases
- Tips: quick tips, shortcuts, best practices

Return a JSON array of segments. If no good segments found, return an empty array [].

Transcript to analyze (timestamps are in brackets like [00:15:30]):
{transcript_text}

Respond ONLY with JSON array, no other text:"""

    def analyze(self, transcript_data: dict, video_path: Path) -> dict:
        video_duration = transcript_data.get("transcription", {}).get("duration", 0)
        
        transcript_text = self._extract_text_from_transcript(transcript_data)
        
        if not transcript_text.strip():
            logger.warning("No transcript text found")
            return {"source_video": str(video_path), "duration_seconds": video_duration, "segments": []}
        
        logger.info(f"Analyzing full transcript ({(video_duration/60):.0f} minutes)...")
        
        chunk_prompt = self._build_prompt(transcript_text)
        
        messages = [
            {"role": "system", "content": "You are an expert video editor specializing in finding engaging content for YouTube Shorts. Always respond with valid JSON."},
            {"role": "user", "content": chunk_prompt},
        ]
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": 16384,
        }
        
        try:
            response = self._session.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                timeout=None,
            )
            
            if response.status_code != 200:
                logger.error(f"Request failed: {response.status_code} - {response.text}")
                return {"source_video": str(video_path), "duration_seconds": video_duration, "segments": []}
            
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            all_segments = self._parse_llm_response(
                content,
                transcript_data,
                video_duration,
            )
            
            logger.info(f"Analysis complete: found {len(all_segments)} potential shorts")
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return {"source_video": str(video_path), "duration_seconds": video_duration, "segments": []}
        
        return {
            "source_video": str(video_path),
            "duration_seconds": video_duration,
            "segments": all_segments,
        }

    def _split_into_chunks(self, transcript_data: dict, video_duration: float) -> list:
        chunk_duration = config.TRANSCRIPT_CHUNK_MINUTES * 60
        chunk_overlap = config.TRANSCRIPT_CHUNK_OVERLAP_MINUTES * 60
        chunks = []
        
        all_segments = transcript_data.get("transcription", {}).get("segments", [])
        
        chunk_start = 0
        while chunk_start < video_duration:
            chunk_end = min(chunk_start + chunk_duration, video_duration)
            
            chunk_segments = [
                seg for seg in all_segments
                if seg.get("start", 0) >= chunk_start and seg.get("end", 0) <= chunk_end
            ]
            
            chunk_text = " ".join(seg.get("text", "").strip() for seg in chunk_segments)
            
            chunks.append({
                "start_seconds": chunk_start,
                "end_seconds": chunk_end,
                "text": chunk_text,
            })
            
            chunk_start = chunk_end - chunk_overlap
            
            if chunk_start <= chunks[-1]["start_seconds"]:
                break
        
        logger.info(f"Split transcript into {len(chunks)} chunks of {config.TRANSCRIPT_CHUNK_MINUTES} minutes each with {config.TRANSCRIPT_CHUNK_OVERLAP_MINUTES} minutes overlap")
        return chunks

    def _deduplicate_segments(self, segments: list) -> list:
        if not segments:
            return segments
        
        seen = set()
        unique_segments = []
        
        for seg in segments:
            key = (seg.get("start_seconds"), seg.get("end_seconds"))
            if key not in seen:
                seen.add(key)
                unique_segments.append(seg)
        
        if len(unique_segments) < len(segments):
            logger.info(f"Deduplicated {len(segments) - len(unique_segments)} duplicate segments")
        
        return unique_segments

    def _extract_text_from_transcript(self, transcript_data: dict) -> str:
        segments = transcript_data.get("transcription", {}).get("segments", [])
        formatted = []
        for seg in segments:
            start = seg.get("start", 0)
            text = seg.get("text", "").strip()
            if text:
                formatted.append(f"[{format_timestamp(start)}] {text}")
        return " ".join(formatted)

    def _parse_llm_response(
        self,
        content: str,
        transcript_data: dict,
        video_duration: float,
        chunk_start_offset: float = 0,
    ) -> list:
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        try:
            segments = json.loads(content)
        except json.JSONDecodeError:
            json_match = re.search(r'\[[\s\S]*\]', content)
            if json_match:
                segments = json.loads(json_match.group())
            else:
                logger.error(f"Failed to parse LLM response as JSON: {content[:500]}")
                return []

        if not isinstance(segments, list):
            segments = [segments]

        enriched_segments = []
        for i, seg in enumerate(segments):
            if not isinstance(seg, dict):
                continue

            start_seconds = self._parse_time_field(seg.get("start_time"))
            end_seconds = self._parse_time_field(seg.get("end_time"))
            
            if start_seconds is None or end_seconds is None:
                continue
            
            start_seconds += chunk_start_offset
            end_seconds += chunk_start_offset
            
            transcript = get_transcript_segment(
                transcript_data,
                start_seconds,
                end_seconds,
            )
            
            before_transcript = get_transcript_before(
                transcript_data,
                start_seconds,
                config.BUFFER_SECONDS,
            )
            
            after_transcript = get_transcript_after(
                transcript_data,
                end_seconds,
                config.BUFFER_SECONDS,
                video_duration,
            )

            enriched_seg = {
                "id": i + 1,
                "start_time": format_timestamp(start_seconds),
                "end_time": format_timestamp(end_seconds),
                "start_seconds": start_seconds,
                "end_seconds": end_seconds,
                "category": seg.get("category", "technical_detail"),
                "title": seg.get("title", f"Short {i + 1}"),
                "description": seg.get("description", ""),
                "confidence": float(seg.get("confidence", 0.5)),
                "transcript": transcript,
                "before_buffer_transcript": before_transcript,
                "after_buffer_transcript": after_transcript,
            }
            
            enriched_segments.append(enriched_seg)

        return enriched_segments

    def _parse_time_field(self, value) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            return parse_timestamp(value)
        return None


class TopicExtractor:
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_retries: int = 3,
    ):
        self.base_url = base_url or config.LM_STUDIO_BASE_URL
        self.api_key = api_key or config.LM_STUDIO_API_KEY
        self.model = model or config.LM_STUDIO_MODEL
        self.temperature = temperature
        self.max_retries = max_retries
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        })

    def _build_prompt(
        self,
        transcript_text: str,
        video_duration: float,
        chunk_start: float = 0,
        chunk_end: Optional[float] = None,
        is_first: bool = True,
        previous_chapters: Optional[list] = None,
    ) -> str:
        if chunk_end is None:
            chunk_end = video_duration

        if is_first:
            header = """You are analyzing a Twitch stream transcript to identify chapter markers and suggest a title for a YouTube video.

This is the BEGINNING of the video. Your first task is to suggest a title."""
            first_chapter_note = "2. The FIRST chapter MUST start at 00:00:00"
        else:
            header = f"""You are analyzing a section of a Twitch stream transcript ({format_timestamp(chunk_start)} to {format_timestamp(chunk_end)}) to identify chapter markers.

The content before {format_timestamp(chunk_start)} has already been analyzed."""
            first_chapter_note = "2. Do NOT force a chapter at the start of this section. Only add a chapter if there is an actual topic transition at that point."

        previous_context = ""
        if previous_chapters:
            prev_lines = "\n".join(
                f"- {c['timestamp']}: {c['title']}"
                for c in previous_chapters
            )
            previous_context = f"""
Previous chapters already identified earlier in this video:
{prev_lines}

These chapters are already set. Do NOT include them again in your response. Only identify NEW topic changes in this section.
"""

        return f"""{header}

The transcript contains timestamps showing when each segment was spoken.

For each new chapter, identify:
- timestamp: When the chapter/topic starts (HH:MM:SS format from the beginning of the video)
- title: A short, descriptive chapter title (max 60 characters)

IMPORTANT:
1. Return chapters at natural topic boundaries (typically every 3-10 minutes)
{first_chapter_note}
3. Chapters should represent meaningful topic changes, not just time intervals
4. Topic chapter titles should lead with the keyword and be concise (e.g., "Python Setup" not "Setting up Python")
5. Do NOT add chapters for stream breaks — those are handled separately
{previous_context}
Transcript to analyze (timestamps in brackets like [00:15:30]):
{transcript_text}

Respond ONLY with JSON object with "title" (string in first section) and "chapters" (array) fields, no other text:"""

    def _call_llm(self, prompt: str) -> Optional[str]:
        messages = [
            {"role": "system", "content": "You are an expert video editor specializing in identifying chapter markers for YouTube videos. Always respond with valid JSON."},
            {"role": "user", "content": prompt},
        ]

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": 16384,
        }

        try:
            response = self._session.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                timeout=None,
            )

            if response.status_code != 200:
                logger.error(f"Chapter extraction request failed: {response.status_code} - {response.text}")
                return None

            result = response.json()
            return result["choices"][0]["message"]["content"]

        except Exception as e:
            logger.error(f"Chapter extraction request failed: {e}")
            return None

    def extract_chapters(self, transcript_data: dict, video_path: Path) -> dict:
        video_duration = transcript_data.get("transcription", {}).get("duration", 0)
        chunks = _get_chunks(video_duration)

        logger.info(f"Splitting video into {len(chunks)} chunks of {config.CHAPTER_CHUNK_MINUTES} min each with {config.CHAPTER_CHUNK_OVERLAP_MINUTES} min overlap")

        all_chapters = []
        suggested_title = None

        for i, (chunk_start, chunk_end) in enumerate(chunks):
            is_first = (i == 0)
            chunk_text = extract_transcript_window(transcript_data, chunk_start, chunk_end)

            if not chunk_text.strip():
                logger.info(f"  Chunk {i + 1}/{len(chunks)}: empty, skipping")
                continue

            previous = [c for c in all_chapters if c["seconds"] < chunk_start]

            prompt = self._build_prompt(
                transcript_text=chunk_text,
                video_duration=video_duration,
                chunk_start=chunk_start,
                chunk_end=chunk_end,
                is_first=is_first,
                previous_chapters=previous,
            )

            logger.info(f"  Chunk {i + 1}/{len(chunks)} ({format_timestamp(chunk_start)} → {format_timestamp(chunk_end)}): analyzing...")

            content = self._call_llm(prompt)
            if content is None:
                logger.warning(f"  Chunk {i + 1}/{len(chunks)}: LLM call failed, skipping")
                continue

            chunk_title, chapters = self._parse_chapters_response(content, video_duration)

            if is_first and chunk_title:
                suggested_title = chunk_title

            seen_seconds = {c["seconds"] for c in all_chapters}
            added = 0
            for ch in chapters:
                if ch["seconds"] < chunk_start:
                    continue
                if ch["seconds"] in seen_seconds:
                    continue
                seen_seconds.add(ch["seconds"])
                all_chapters.append(ch)
                added += 1

            logger.info(f"  Chunk {i + 1}/{len(chunks)}: found {added} new topic chapters")

        all_chapters.sort(key=lambda x: x["seconds"])

        has_first = any(c["seconds"] == 0 for c in all_chapters)
        if not has_first and all_chapters:
            all_chapters.insert(0, {
                "timestamp": "00:00:00",
                "seconds": 0,
                "title": "Start",
            })

        logger.info(f"Chapter extraction complete: found {len(all_chapters)} total chapters")
        return {
            "source_video": str(video_path),
            "duration_seconds": video_duration,
            "suggested_title": suggested_title,
            "chapters": all_chapters,
        }

    def _parse_chapters_response(self, content: str, video_duration: float) -> tuple:
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        suggested_title = None
        chapters = []

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                try:
                    parsed = json.loads(json_match.group())
                except json.JSONDecodeError:
                    json_match = re.search(r'\[[\s\S]*\]', content)
                    if json_match:
                        parsed = json.loads(json_match.group())
                    else:
                        logger.error(f"Failed to parse chapters as JSON: {content[:500]}")
                        return None, []
            else:
                logger.error(f"Failed to parse chapters as JSON: {content[:500]}")
                return None, []

        if isinstance(parsed, dict):
            suggested_title = parsed.get("title")
            chapters = parsed.get("chapters", [])
        elif isinstance(parsed, list):
            chapters = parsed

        if not isinstance(chapters, list):
            return suggested_title, []

        if suggested_title and isinstance(suggested_title, str):
            suggested_title = suggested_title[:100]
        else:
            suggested_title = None

        valid_chapters = []
        seen_timestamps = set()

        for chapter in chapters:
            if not isinstance(chapter, dict):
                continue

            ts = chapter.get("timestamp")
            title = chapter.get("title", "Untitled")

            if ts is None:
                continue

            seconds = parse_timestamp(ts)
            if seconds > video_duration:
                continue
            if seconds in seen_timestamps:
                continue

            seen_timestamps.add(seconds)
            valid_chapters.append({
                "timestamp": ts,
                "seconds": seconds,
                "title": title[:60],
            })

        valid_chapters.sort(key=lambda x: x["seconds"])

        return suggested_title, valid_chapters


class BreaksExtractor:
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_retries: int = 3,
    ):
        self.base_url = base_url or config.LM_STUDIO_BASE_URL
        self.api_key = api_key or config.LM_STUDIO_API_KEY
        self.model = model or config.LM_STUDIO_MODEL
        self.temperature = temperature
        self.max_retries = max_retries
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        })

    def _build_prompt(
        self,
        transcript_text: str,
        chunk_start: float = 0,
        chunk_end: Optional[float] = None,
        is_first: bool = True,
        previous_breaks: Optional[list] = None,
    ) -> str:
        if chunk_end is not None and chunk_end > 0:
            end_str = format_timestamp(chunk_end)
        else:
            end_str = "the end"

        if is_first:
            section_info = "This is the BEGINNING of the video."
        else:
            section_info = f"This section covers {format_timestamp(chunk_start)} to {end_str}."

        previous_context = ""
        if previous_breaks:
            prev_lines = "\n".join(
                f"- {b['start']} to {b['end']}"
                for b in previous_breaks
            )
            previous_context = f"""
Previous breaks already identified earlier in this video:
{prev_lines}

These breaks are already recorded. Do NOT include them again. Only identify NEW breaks in this section.
"""

        return f"""You are analyzing a Twitch stream transcript to identify when the streamer goes on break and when they return.

{section_info}
The transcript contains timestamps showing when each segment was spoken, like [HH:MM:SS] text.

Identify stream breaks by looking for:
1. Gaps in the transcript — long periods where there is no speech detected
2. Break phrases spoken just before a gap, such as "be right back", "brb", "stepping away", "going on break", "taking a break", "afk", "back in a bit", "step away"
3. Return phrases spoken when the streamer comes back, such as "i'm back", "hello", "welcome back", "back again", "we're back", "hey everyone", "hello again"

For each break, provide:
- start: The timestamp (HH:MM:SS) when the break begins. Use the timestamp of the last relevant speech before the silence, or where a break phrase is spoken.
- end: The timestamp (HH:MM:SS) when the streamer returns. Use the timestamp of the first speech after the silence, or where a return phrase is spoken.

IMPORTANT:
- Only identify breaks where the streamer actually steps away (there will be a significant gap in the transcript)
- A short pause of a few seconds is NOT a break
- If you cannot find a clear return timestamp, set end equal to start
- Gaps in the transcript are the primary signal of a break
{previous_context}
Transcript to analyze (timestamps in brackets like [00:15:30]):
{transcript_text}

Respond ONLY with a JSON object with a single "breaks" array field, no other text:"""

    def _call_llm(self, prompt: str) -> Optional[str]:
        messages = [
            {"role": "system", "content": "You are an expert at identifying stream breaks from transcript data. Always respond with valid JSON."},
            {"role": "user", "content": prompt},
        ]

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": 4096,
        }

        try:
            response = self._session.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                timeout=None,
            )

            if response.status_code != 200:
                logger.error(f"Break extraction request failed: {response.status_code} - {response.text}")
                return None

            result = response.json()
            return result["choices"][0]["message"]["content"]

        except Exception as e:
            logger.error(f"Break extraction request failed: {e}")
            return None

    def extract_breaks(self, transcript_data: dict, video_path: Path) -> dict:
        video_duration = transcript_data.get("transcription", {}).get("duration", 0)
        chunks = _get_chunks(video_duration)

        logger.info(f"Splitting break detection into {len(chunks)} chunks")

        all_breaks = []

        for i, (chunk_start, chunk_end) in enumerate(chunks):
            is_first = (i == 0)
            chunk_text = extract_transcript_window(transcript_data, chunk_start, chunk_end)

            if not chunk_text.strip():
                logger.info(f"  Breaks chunk {i + 1}/{len(chunks)}: empty, skipping")
                continue

            previous = [b for b in all_breaks if b["start_seconds"] < chunk_start]

            prompt = self._build_prompt(
                transcript_text=chunk_text,
                chunk_start=chunk_start,
                chunk_end=chunk_end,
                is_first=is_first,
                previous_breaks=previous,
            )

            logger.info(f"  Breaks chunk {i + 1}/{len(chunks)} ({format_timestamp(chunk_start)} → {format_timestamp(chunk_end)}): analyzing...")

            content = self._call_llm(prompt)
            if content is None:
                logger.warning(f"  Breaks chunk {i + 1}/{len(chunks)}: LLM call failed, skipping")
                continue

            break_events = self._parse_breaks_response(content, video_duration)

            seen_starts = {b["start_seconds"] for b in all_breaks}
            added = 0
            for brk in break_events:
                if brk["start_seconds"] < chunk_start:
                    continue
                if brk["start_seconds"] in seen_starts:
                    continue
                seen_starts.add(brk["start_seconds"])
                all_breaks.append(brk)
                added += 1

            logger.info(f"  Breaks chunk {i + 1}/{len(chunks)}: found {added} new breaks")

        all_breaks.sort(key=lambda x: x["start_seconds"])

        logger.info(f"Break extraction complete: found {len(all_breaks)} total breaks")
        return {
            "source_video": str(video_path),
            "duration_seconds": video_duration,
            "breaks": all_breaks,
        }

    def _parse_breaks_response(self, content: str, video_duration: float) -> list:
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                try:
                    parsed = json.loads(json_match.group())
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse breaks as JSON: {content[:500]}")
                    return []
            else:
                logger.error(f"Failed to parse breaks as JSON: {content[:500]}")
                return []

        break_events = parsed.get("breaks", []) if isinstance(parsed, dict) else (parsed if isinstance(parsed, list) else [])

        if not isinstance(break_events, list):
            return []

        valid_breaks = []
        for event in break_events:
            if not isinstance(event, dict):
                continue

            start = event.get("start")
            end = event.get("end")

            if not start:
                continue

            start_seconds = parse_timestamp(start)
            end_seconds = parse_timestamp(end) if end else start_seconds

            if start_seconds > video_duration:
                continue
            if end_seconds > video_duration:
                end_seconds = video_duration
            if end_seconds < start_seconds:
                end_seconds = start_seconds

            valid_breaks.append({
                "start": start,
                "end": end if end else start,
                "start_seconds": start_seconds,
                "end_seconds": end_seconds,
            })

        valid_breaks.sort(key=lambda x: x["start_seconds"])
        return valid_breaks


def merge_chapters(
    topic_chapters: list,
    break_events: list,
    video_duration: float,
) -> list:
    """Merge topic chapters and break events into a single sorted chapter list.

    If a break starts and ends within the same topic, a "{topic} cont'd" chapter
    is inserted at the resume point. If the topic changed during the break (the
    resume point falls in a new topic), no cont'd is needed — the break itself
    served as a topic transition.
    """
    result = [t.copy() for t in topic_chapters]

    for brk in sorted(break_events, key=lambda x: x["start_seconds"]):
        bs = brk["start_seconds"]
        be = brk["end_seconds"]

        containing_topic = None
        for i, ch in enumerate(topic_chapters):
            next_ts = topic_chapters[i + 1]["seconds"] if i + 1 < len(topic_chapters) else video_duration
            if ch["seconds"] <= bs < next_ts:
                containing_topic = ch
                break
        if containing_topic is None and topic_chapters:
            containing_topic = topic_chapters[-1]

        topic_at_be = None
        for ch in reversed(topic_chapters):
            if ch["seconds"] <= be:
                topic_at_be = ch
                break

        same_topic = (
            containing_topic is not None
            and topic_at_be is not None
            and containing_topic["seconds"] == topic_at_be["seconds"]
        )

        result.append({
            "timestamp": format_timestamp(bs),
            "seconds": bs,
            "title": "Break",
        })

        if same_topic:
            result.append({
                "timestamp": format_timestamp(be),
                "seconds": be,
                "title": f"{containing_topic['title']} cont'd",
            })

    result.sort(key=lambda x: x["seconds"])
    seen = set()
    deduped = []
    for ch in result:
        if ch["seconds"] not in seen:
            seen.add(ch["seconds"])
            deduped.append(ch)

    has_first = any(c["seconds"] == 0 for c in deduped)
    if not has_first:
        deduped.insert(0, {"timestamp": "00:00:00", "seconds": 0, "title": "Start"})

    return deduped


# Backward compatibility alias
ChapterExtractor = TopicExtractor
